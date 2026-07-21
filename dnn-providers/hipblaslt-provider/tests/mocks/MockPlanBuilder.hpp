// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "engines/plans/PlanBuilderInterface.hpp"

namespace hipblaslt_plugin
{

class MockPlanBuilder : public IPlanBuilder
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 HipdnnEnginePluginExecutionContext& executionContext),
                (const, override));
};

}

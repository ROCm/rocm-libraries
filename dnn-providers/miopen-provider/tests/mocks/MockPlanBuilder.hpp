// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"

namespace miopen_plugin
{

class MockPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                               HipdnnMiopenSettings,
                                                               HipdnnMiopenContext>
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const HipdnnMiopenSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionSettings,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnMiopenSettings& executionSettings),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnMiopenContext& executionContext),
                (const, override));

    MOCK_METHOD((std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>),
                getCustomKnobs,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
};

}

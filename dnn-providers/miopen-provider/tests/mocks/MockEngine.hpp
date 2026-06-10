/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"

namespace miopen_plugin
{

class MockEngine : public hipdnn_plugin_sdk::
                       IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool,
                isApplicable,
                (HipdnnMiopenHandle & handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(void,
                getDetails,
                (HipdnnMiopenHandle & handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 hipdnnPluginConstData_t& detailsOut),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionContext,
                (const HipdnnMiopenHandle& handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnMiopenContext& executionContext),
                (const, override));
};

} // namespace miopen_plugin

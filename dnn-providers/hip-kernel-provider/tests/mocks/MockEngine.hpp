/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"
#include "hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

namespace hip_kernel_plugin
{

class MockEngine : public hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                     HipdnnHipKernelSettings,
                                                     HipdnnHipKernelContext>
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool,
                isApplicable,
                (HipdnnHipKernelHandle & handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(void,
                getDetails,
                (HipdnnHipKernelHandle & handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 hipdnnPluginConstData_t& detailsOut),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionContext,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                 HipdnnHipKernelContext& executionContext),
                (const, override));
};

} // namespace hip_kernel_plugin

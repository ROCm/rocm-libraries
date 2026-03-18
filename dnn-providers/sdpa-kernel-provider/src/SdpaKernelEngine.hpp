// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "SdpaKernelContext.hpp"
#include "SdpaKernelHandle.hpp"
#include "SdpaKernelSettings.hpp"

namespace sdpa_kernel_provider
{

class SdpaKernelEngine
    : public hipdnn_plugin_sdk::IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>
{
public:
    static constexpr const char* engineName()
    {
        return "SdpaKernelPlugin";
    }

    int64_t id() const override;

    bool isApplicable(SdpaKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(SdpaKernelHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(const SdpaKernelHandle& handle,
                            const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&
                                engineConfig) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void initializeExecutionContext(
        const SdpaKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        SdpaKernelContext& executionContext) const override;
};

}

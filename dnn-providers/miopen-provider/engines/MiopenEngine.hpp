// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

#include "HipdnnMiopenHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace miopen_plugin
{

/**
 * @brief MIOpen implementation of the IEngine interface.
 *
 * This class implements the templated IEngine interface using MIOpen-specific types.
 * It manages a collection of plan builders and delegates operations to them.
 */
class MiopenEngine : public hipdnn_plugin_sdk::
                         IEngine<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    MiopenEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(
        HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(HipdnnMiopenHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const HipdnnMiopenHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnMiopenContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                        HipdnnMiopenSettings,
                                                        HipdnnMiopenContext>> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                                HipdnnMiopenSettings,
                                                                HipdnnMiopenContext>>>
        _planBuilders;
};

}

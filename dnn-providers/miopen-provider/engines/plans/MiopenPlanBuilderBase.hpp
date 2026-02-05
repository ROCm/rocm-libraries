// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"

namespace miopen_plugin
{

class MiopenPlanBuilderBase : public IPlanBuilder
{
public:
    ~MiopenPlanBuilderBase() override = default;

    size_t getMaxWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

} // namespace miopen_plugin

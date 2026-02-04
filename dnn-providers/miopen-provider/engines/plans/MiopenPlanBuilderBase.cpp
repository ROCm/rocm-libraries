// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenPlanBuilderBase.hpp"

namespace miopen_plugin
{

size_t MiopenPlanBuilderBase::getMaxWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return 0;
}

} // namespace miopen_plugin

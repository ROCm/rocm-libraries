// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlanBuilderInterface.hpp"

namespace miopen_legacy_plugin
{

size_t IPlanBuilder::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                      [[maybe_unused]] const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    return 0;
}

} // namespace miopen_legacy_plugin

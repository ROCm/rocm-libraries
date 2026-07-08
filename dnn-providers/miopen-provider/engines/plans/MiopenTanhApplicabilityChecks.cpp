// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/plans/MiopenTanhApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationChecks.hpp"

namespace miopen_plugin::tanh_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

void checkTanhModeSupported(const PointwiseAttributes& attrs)
{
    if(attrs.operation() != PointwiseMode::TANH_FWD)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Tanh plan builder: unsupported pointwise mode. "
            "Supported mode: TANH_FWD");
    }
}

bool isTanhSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    return unary_activation_applicability::isSupported(opGraph, "Tanh", checkTanhModeSupported);
}

} // namespace miopen_plugin::tanh_applicability

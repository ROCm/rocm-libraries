// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/plans/MiopenSigmoidApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationChecks.hpp"

namespace miopen_plugin::sigmoid_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

void checkSigmoidModeSupported(const PointwiseAttributes& attrs)
{
    if(attrs.operation() != PointwiseMode::SIGMOID_FWD)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Sigmoid plan builder: unsupported pointwise mode. "
            "Supported mode: SIGMOID_FWD");
    }
}

bool isSigmoidSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    return unary_activation_applicability::isSupported(
        opGraph, "Sigmoid", checkSigmoidModeSupported);
}

} // namespace miopen_plugin::sigmoid_applicability

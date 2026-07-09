// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/plans/MiopenReluApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationChecks.hpp"

namespace miopen_plugin::relu_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// NOTE: Keep this check's branch order identical to MiopenUtils::mapPointwiseModeToMiopenActivation.
// If this check accepts a parameter combination that the mapping cannot represent, MIOpen may
// silently execute the op as a standard ReLU, leading to incorrect results.
void checkReluModeSupported(const PointwiseAttributes& attrs)
{
    if(attrs.operation() != PointwiseMode::RELU_FWD)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Relu plan builder: unsupported pointwise mode. "
            "Supported mode: RELU_FWD");
    }

    const auto lowerClip = attrs.relu_lower_clip();
    const auto upperClip = attrs.relu_upper_clip();
    const auto lowerClipSlope = attrs.relu_lower_clip_slope();

    if(lowerClip && upperClip)
    {
        return; // Clamp
    }
    if(upperClip)
    {
        return; // Clipped ReLU
    }
    if(lowerClipSlope)
    {
        return; // Leaky ReLU
    }
    if(lowerClip && *lowerClip != 0.f)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Relu plan builder: standard relu with a non-zero lower_clip is not supported");
    }

    // Standard ReLU (including lower_clip == 0.0, which is a no-op lower clip).
}

bool isReluSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    return unary_activation_applicability::isSupported(opGraph, "Relu", checkReluModeSupported);
}

} // namespace miopen_plugin::relu_applicability

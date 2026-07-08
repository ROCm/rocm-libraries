// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/plans/MiopenReluApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationChecks.hpp"

namespace miopen_plugin::relu_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// NOTE: this mirrors the branch order/fallthrough of
// MiopenUtils::mapPointwiseModeToMiopenActivation exactly.
// function accepts it and executes it as a Standard ReLU. Keeping this check structurally
// identical to the mapping function prevents that kind of drift.
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

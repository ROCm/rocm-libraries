// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <string>
#include <unordered_map>

namespace hipdnn_frontend::autotune::detail
{

// Validates an engine sweep spec's axes and fixed settings against the engine's available knobs.
inline Error validateSweepSpec(const EngineSweepSpec& sweepSpec,
                               const std::unordered_map<KnobType_t, Knob>& knobLookup)
{
    // Validate axis knob IDs exist and each value is within range
    for(const auto& axis : sweepSpec.axes)
    {
        if(axis.knobId == autotune::detail::BENCHMARKING_KNOB_NAME)
        {
            HIPDNN_FE_LOG_WARN("Stripping internal knob '"
                               << autotune::detail::BENCHMARKING_KNOB_NAME
                               << "' from add_engine_sweep() sweep axis. "
                               << "This knob is managed by autotune() in EXHAUSTIVE mode.");
            continue;
        }
        if(axis.values.empty())
        {
            HIPDNN_FE_LOG_WARN("Dropping sweep axis for knob '"
                               << axis.knobId << "' on engine " << sweepSpec.engineId
                               << ": no values provided; knob will take its engine default "
                               << "(axis not swept).");
            continue;
        }
        auto knobIt = knobLookup.find(axis.knobId);
        if(knobIt == knobLookup.end())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Sweep axis knob '" + axis.knobId + "' is not available for engine "
                        + std::to_string(sweepSpec.engineId)};
        }
        for(const auto& val : axis.values)
        {
            const KnobSetting testSetting(axis.knobId, val);
            auto valErr = knobIt->second.validate(testSetting);
            if(valErr.is_bad())
            {
                return {ErrorCode::INVALID_VALUE,
                        "Sweep axis knob '" + axis.knobId + "' has invalid value for engine "
                            + std::to_string(sweepSpec.engineId) + ": " + valErr.get_message()};
            }
        }
    }

    // Validate fixed settings
    for(const auto& [knobId, value] : sweepSpec.fixedSettings)
    {
        if(knobId == autotune::detail::BENCHMARKING_KNOB_NAME)
        {
            HIPDNN_FE_LOG_WARN("Stripping internal knob '"
                               << autotune::detail::BENCHMARKING_KNOB_NAME
                               << "' from add_engine_sweep() fixed settings. "
                               << "This knob is managed by autotune() in EXHAUSTIVE mode.");
            continue;
        }
        auto knobIt = knobLookup.find(knobId);
        if(knobIt == knobLookup.end())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Fixed setting knob '" + knobId + "' is not available for engine "
                        + std::to_string(sweepSpec.engineId)};
        }
        const KnobSetting testSetting(knobId, value);
        auto valErr = knobIt->second.validate(testSetting);
        if(valErr.is_bad())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Fixed setting knob '" + knobId + "' has invalid value for engine "
                        + std::to_string(sweepSpec.engineId) + ": " + valErr.get_message()};
        }
    }

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::autotune::detail

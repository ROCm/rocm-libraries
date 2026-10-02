// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/// @file IngestorPlanRefusal.hpp
/// The one mapping from an ingestor plan save or load refusal onto a plugin status.
///
/// The backend reports both statuses as one hipDNN status. The message prefix therefore
/// names the category, and the rest of the message names the cause.
namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

enum class IngestorPlanRefusal
{
    /// The plan is valid, but this provider cannot save or load it here.
    INCOMPATIBLE,
    /// The plan data is truncated, corrupt or inconsistent.
    DAMAGED,
};

inline constexpr std::string_view INGESTOR_PLAN_INCOMPATIBLE_PREFIX
    = "ingestor plan is valid but cannot be saved or loaded here: ";
inline constexpr std::string_view INGESTOR_PLAN_DAMAGED_PREFIX = "ingestor plan data is damaged: ";

/// Throws `HipdnnPluginException`. `INCOMPATIBLE` gives `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`,
/// and `DAMAGED` gives `HIPDNN_PLUGIN_STATUS_INVALID_VALUE`.
[[noreturn]] inline void refuseIngestorPlan(IngestorPlanRefusal refusal, const std::string& message)
{
    switch(refusal)
    {
    case IngestorPlanRefusal::INCOMPATIBLE:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            std::string(INGESTOR_PLAN_INCOMPATIBLE_PREFIX) + message);
    case IngestorPlanRefusal::DAMAGED:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       std::string(INGESTOR_PLAN_DAMAGED_PREFIX)
                                                           + message);
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "unknown ingestor plan refusal: " + message);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

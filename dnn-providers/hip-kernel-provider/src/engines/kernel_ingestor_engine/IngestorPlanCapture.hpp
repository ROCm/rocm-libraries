// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Collects everything a saved form of @p plan holds: the selected kernel's identity and
/// code object, its dispatch handler's launch inputs, and the plan's own workspace size.
///
/// For a benchmarking plan, the plan saved is the candidate that the first execute()
/// chose. A plan restored from a saved payload is refused. Reads the plan only through
/// const accessors and an acquire load, so it is safe while other threads execute the
/// plan.
///
/// @param engineId The id of the engine that built @p plan.
/// @param engineName The name of that engine. Its hash must equal @p engineId.
/// @throws HipdnnPluginException with a save refusal when the plan cannot be saved, and
///         with INTERNAL_ERROR when the provider is inconsistent.
serialization::IngestorPlanPayload captureIngestorPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                                       const Handle& handle,
                                                       int64_t engineId,
                                                       const std::string& engineName);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Collects everything a saved form of @p plan holds: for each built kernel, its identity and
/// code objects, its dispatch handler's launch inputs, and the user-supplied scalar UIDs it
/// reads; and the plan's own workspace size.
///
/// An unslotted plan served by one launch is a kind 1 payload. A slot program, and an unslotted
/// plan that launches several kernels, is a kind 2 payload: one stage per slot (one stage with
/// no slot name for an unslotted plan), each code object stored once. A kernel with several
/// launches is saved only through a handler that opts in with supportsSavingAllLaunches().
///
/// For a benchmarking plan, the plan saved is the candidate that the first execute()
/// chose, and for a benchmarking slot program the program it served. A plan restored from a
/// saved payload is never captured: hipDNN saves a loaded plan from the bytes it was loaded
/// from, so only a direct call of the provider hook sees a restored plan, and the hook refuses
/// it. A plan that no ingestor engine built is refused. Reads the plan only through const
/// accessors and an acquire load, so it is safe while other threads execute the plan.
///
/// @throws HipdnnPluginException with a save refusal when the plan cannot be saved, and
///         with INTERNAL_ERROR when the provider is inconsistent.
serialization::IngestorSavedPlan captureIngestorPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                                     const Handle& handle);

/// Refuses to save a plan that no ingestor engine built.
/// @throws HipdnnPluginException with HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE.
[[noreturn]] void refuseNonIngestorPlanSave();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

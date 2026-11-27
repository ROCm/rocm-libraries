// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// What a restore asks of the provider apart from the payload.
struct IngestorPlanRestoreEnvironment
{
    /// Resolves the handle's device and its properties. Must not be null.
    const hipdnn_plugin_sdk::ingestor::IDeviceResolver<Handle>* deviceResolver = nullptr;
};

/// Rebuilds an executable plan from a saved ingestor plan payload.
///
/// A payload holds GPU code, and the restored plan executes that code. Restore only
/// payloads from a trusted source. The checks below detect damage and incompatibility.
/// They do not detect a payload made to look valid.
///
/// The steps run in this order, and the first failure refuses the payload. The RFC 0009 load
/// checks each step makes are named beside it; RFC 0009 runs its check 10 last, as step 9 does.
///   1. The header: marker, format version and plan kind (RFC 0009 checks 1 to 5).
///   2. The SHA-256 of the body (RFC 0009 checks 6 and 7).
///   3. The FlatBuffers verifier and the field structure (RFC 0009 checks 8 and 9).
///   4. The handle's device can run the target of every code object (RFC 0009 check 11). A
///      kind 2 payload is checked for every code object before any handler restores a stage.
///   5. A dispatch handler is registered under the saved dispatch name of each stage, and it
///      restores (RFC 0009 checks 12 and 13). A handler that opted in to several launches
///      restores every stage of a kind 2 payload, one kernel included. Any other handler
///      restores a stage of one kernel.
///   6. The recorded kernel signature of each kernel matches the arguments the handler
///      launches, and the handler launches exactly the saved set of launches (RFC 0009
///      check 14).
///   7. The launch values meet the contract of the dispatch name (RFC 0009 check 15).
///   8. The driver loads each code object on the handle's device and finds its symbol (RFC
///      0009 check 16).
///   9. The runtime pass-by-value UIDs the plan lists are exactly the UIDs the restored
///      handlers bind, over all stages (RFC 0009 check 10). A restored dispatch that no
///      ingestor pack built binds none.
///
/// Damage gives HIPDNN_PLUGIN_STATUS_INVALID_VALUE. A valid payload that cannot be
/// restored here gives HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE. Restore reads no descriptor,
/// no archive and no benchmark result.
///
/// @throws HipdnnPluginException on any refusal, and with INTERNAL_ERROR when the handle
///         names no device.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data,
                        size_t size,
                        const Handle& handle,
                        const IngestorPlanRestoreEnvironment& environment);

/// Rebuilds an executable plan from decoded payload contents: a RestoredIngestorPlan for a
/// kind 1 payload, and a RestoredSlotProgramPlan for every kind 2 payload, an unslotted one
/// included. Runs steps 4 to 9 of restoreIngestorPlan(); the decoder has run steps 1 to 3.
/// The rvalue overload takes the code bytes out of @p contents instead of copying them.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorSavedPlan(serialization::IngestorSavedPlan&& contents,
                             const Handle& handle,
                             const IngestorPlanRestoreEnvironment& environment);

/// restoreIngestorSavedPlan() on a copy of @p contents.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorSavedPlan(const serialization::IngestorSavedPlan& contents,
                             const Handle& handle,
                             const IngestorPlanRestoreEnvironment& environment);

/// restoreIngestorPlan() with this provider's device resolver.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data, size_t size, const Handle& handle);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

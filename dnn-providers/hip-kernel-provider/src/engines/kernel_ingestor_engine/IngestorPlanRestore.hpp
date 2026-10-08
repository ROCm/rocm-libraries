// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// What a restore asks of the provider apart from the payload.
struct IngestorPlanRestoreEnvironment
{
    /// The registered name of the loaded engine with this id. No value when no engine with
    /// this id is loaded, or when the engine-name registry has no name for this id.
    std::function<std::optional<std::string>(int64_t)> loadedEngineName;
    /// Resolves the handle's device and its properties. Must not be null.
    const hipdnn_plugin_sdk::ingestor::IDeviceResolver<Handle>* deviceResolver = nullptr;
};

/// A restored plan and the ID of the engine that its payload names.
struct IngestorPlanRestoreResult
{
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> plan;
    int64_t engineId = 0;
};

/// Rebuilds an executable plan from a saved ingestor plan payload.
///
/// A payload holds GPU code, and the restored plan executes that code. Restore only
/// payloads from a trusted source. The checks below detect damage and incompatibility.
/// They do not detect a payload made to look valid.
///
/// The checks run in this order, and the first failure refuses the payload:
///   1. The header: marker, format version and plan kind.
///   2. The SHA-256 of the body.
///   3. The FlatBuffers verifier and the field structure.
///   4. An engine with the id of the saved engine name is loaded, and the engine-name
///      registry maps that id to that name.
///   5. The handle's device can run the code object's target.
///   6. A dispatch handler is registered under the saved dispatch name, and it restores.
///   7. The recorded kernel signature matches the arguments the handler launches.
///   8. The launch values meet the contract of the dispatch name.
///   9. The driver loads the code object on the handle's device and finds the symbol.
///
/// Damage gives HIPDNN_PLUGIN_STATUS_INVALID_VALUE. A valid payload that cannot be
/// restored here gives HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE. Restore reads no descriptor,
/// no archive and no benchmark result.
///
/// The result holds the ID of the saved engine name, which check 4 found loaded.
///
/// @throws HipdnnPluginException on any refusal, and with INTERNAL_ERROR when the handle
///         names no device.
IngestorPlanRestoreResult restoreIngestorPlan(const uint8_t* data,
                                              size_t size,
                                              const Handle& handle,
                                              const IngestorPlanRestoreEnvironment& environment);

/// restoreIngestorPlan() with this provider's loaded engines and device resolver.
IngestorPlanRestoreResult
    restoreIngestorPlan(const uint8_t* data, size_t size, const Handle& handle);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

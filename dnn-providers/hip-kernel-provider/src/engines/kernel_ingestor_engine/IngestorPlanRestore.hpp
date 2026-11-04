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
/// The checks run in this order, and the first failure refuses the payload:
///   1. The header: marker, format version and plan kind.
///   2. The SHA-256 of the body.
///   3. The FlatBuffers verifier and the field structure.
///   4. The handle's device can run the code object's target.
///   5. A dispatch handler is registered under the saved dispatch name, and it restores.
///   6. The recorded kernel signature matches the arguments the handler launches.
///   7. The launch values meet the contract of the dispatch name.
///   8. The driver loads the code object on the handle's device and finds the symbol.
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

/// restoreIngestorPlan() with this provider's device resolver.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data, size_t size, const Handle& handle);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

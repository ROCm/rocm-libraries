// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief Registers every ingestor pack's native matchers, scorers, and dispatch
 *        handlers, exactly once for the life of the process.
 *
 * Called from Container's constructor, before any descriptor-backed engine resolves the
 * symbols its descriptors name. Idempotent.
 *
 * Does not throw on a pack failing: that pack is logged, excluded from
 * discoverDescriptorSets(), and the rest still register.
 */
void registerNativeIngestorSymbols();

/**
 * @brief Every descriptor set this provider serves.
 *
 * **The one function ALMIOPEN-2401 replaces.** Its body is the C++ stand-in for a
 * descriptor-file scan; the return type is already what a loader produces.
 *
 * Safe to call for enumeration alone, before registerNativeIngestorSymbols(): it
 * builds plain data and touches no registry. Container::copyEngineIds depends on that,
 * being static and running before any Container exists.
 *
 * Reads the inventory once per process and memoizes, so every caller sees the same
 * set.
 */
std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> discoverDescriptorSets();

/**
 * @brief Hashes @p name into hipDNN's engine-id space, registering it on first call.
 *
 * A descriptor-backed engine is defined by data, so its name is registered at run time
 * rather than by EngineNames.hpp's compile-time macro. Idempotent and thread-safe;
 * never called at static-init, where the registrar's throw would be fatal.
 */
int64_t registerEngineName(const std::string& name);

/// @brief The device resolver every descriptor-backed engine in this provider shares.
///
/// Process-lifetime: a device-property cache with no engine-specific state.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

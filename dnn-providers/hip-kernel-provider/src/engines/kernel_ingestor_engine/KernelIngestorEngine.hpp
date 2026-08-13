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

/// Registers every ingestor pack's native matchers, scorers, and dispatch handlers,
/// once for the process. A pack that fails to register is logged and excluded.
void registerNativeIngestorSymbols();

/// Every descriptor set this provider serves. Registers symbols first: the backend's
/// first call arrives via the static engine-id path before any Container exists, so
/// an unregistered pack must be excluded here, not later. Memoized.
std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> discoverDescriptorSets();

/// Hashes @p name into hipDNN's engine-id space, registering it on first call.
/// Never call at static-init: the registrar's throw would be fatal.
int64_t registerEngineName(const std::string& name);

/// The device resolver every descriptor-backed engine in this provider shares.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

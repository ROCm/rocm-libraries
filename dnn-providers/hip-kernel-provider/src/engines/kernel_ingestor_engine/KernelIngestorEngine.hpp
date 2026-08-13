// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Registers every ingestor pack's native matchers, scorers, and dispatch handlers,
/// once for the process. A pack that fails to register is logged and excluded.
void registerNativeIngestorSymbols();

/// The directory discoverDescriptorSets() reads descriptor files from: the build-tree copy
/// if it exists, the installed copy otherwise, with HIPDNN_DESCRIPTOR_DIR overriding both.
/// Declared here so a test loads exactly what the provider loads -- restating the fallback
/// order in a test is how the two silently drift apart.
std::filesystem::path descriptorSearchDirectory();

/// Every descriptor set this provider serves, read from installed files. Registers symbols
/// first, because validation asks the registry whether each descriptor's symbol exists: a
/// set is returned only if it can actually be built, which is what lets
/// Container::copyEngineIds advertise ids before any engine is constructed. A malformed or
/// unresolvable descriptor costs its pack or its engine, never the provider. Memoized, so
/// two scans can never disagree.
const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets();

/// Hashes @p name into hipDNN's engine-id space, registering it on first call.
/// Never call at static-init: the registrar's throw would be fatal.
int64_t registerEngineName(const std::string& name);

/// The device resolver every descriptor-backed engine in this provider shares.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

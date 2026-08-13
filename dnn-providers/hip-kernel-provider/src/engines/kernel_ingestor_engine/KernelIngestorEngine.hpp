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

/// The directory discoverDescriptorSets() reads descriptor files from: HIPDNN_DESCRIPTOR_DIR
/// if set, the installed copy otherwise. Only the install path is compiled in -- a baked
/// build-tree path would ship inside the plugin and win over the installed files on any host
/// where it happened to exist, so nothing would ever exercise the installed ones. Tests and
/// run-from-build-dir set the variable. Declared here so a test loads exactly what the
/// provider loads: restating the order in a test is how the two silently drift apart.
std::filesystem::path descriptorSearchDirectory();

/// Every descriptor set this provider serves, read from installed files. Registers symbols
/// first, because validation asks the registry whether each descriptor's symbol exists: a
/// set is returned only if it can actually be built, which is what lets
/// Container::copyEngineIds advertise ids before any engine is constructed. A malformed or
/// unresolvable descriptor costs its pack or its engine, never the provider. Memoized, so
/// two scans can never disagree.
const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets();

/// The device resolver every descriptor-backed engine in this provider shares.
/// Process-lifetime: a device-property cache with no engine-specific state.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

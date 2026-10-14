// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Registers every ingestor pack's native matchers, scorers, and dispatch handlers,
/// once for the process. A pack that fails to register is logged and excluded.
void registerNativeIngestorSymbols();

/// The provider's own descriptor tree, from the first of three sources that answers:
/// HIPDNN_DESCRIPTOR_DIR if it names a real directory, else the copy sitting beside this
/// module (resolved through dladdr, so a relocated or DESTDIR install finds its own
/// files), else the configure-time install prefix. Only the last is compiled in -- a baked
/// build-tree path would ship inside the plugin and shadow the installed files. Declared
/// here so tests resolve exactly what the provider resolves.
std::filesystem::path descriptorSearchDirectory();

/// Every directory discoverDescriptorSets() reads, in order: the tree above, then
/// HIPDNN_DESCRIPTOR_RUNTIME_DIR when it names one. The runtime tree is additive rather
/// than an override -- it can add descriptors beside the shipped ones, but a file
/// redefining a shipped id is refused and the shipped definition stands.
std::vector<std::filesystem::path> descriptorSearchDirectories();

/// Every descriptor set this provider serves. Registers symbols first so validation can
/// check each descriptor's symbol exists -- a set returns only if buildable, which lets
/// Container::copyEngineIds advertise ids before any engine is constructed. A malformed
/// descriptor costs its pack, never the provider. Memoized, so two scans can't disagree.
const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets();

/// The state manager discovery built while validating the discoverDescriptorSets() entry
/// whose engine id is @p engineId, or null once taken or when no set has that id.
/// One-shot: the first engine constructed for that set takes it, and any later one -- a
/// Container created after an earlier one was destroyed -- builds its own from the set, so
/// no state manager is shared between two engines. One no engine takes stays alive beside
/// its set until the provider unloads, so a process that never constructs a Container
/// holds one extra state manager per set.
std::unique_ptr<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>>
    takeDiscoveredStateManager(const hipdnn_plugin_sdk::ingestor::DescriptorId& engineId);

/// Test-only. Rescans the descriptor tree through the same path discovery takes and
/// refills every takeDiscoveredStateManager() slot with the state manager that scan built.
/// Thread-safe against concurrent takes.
/// @throws std::logic_error the rescan does not reproduce discoverDescriptorSets() in
///         order (e.g. the search directories changed); the slots are left unchanged.
void rediscoverStateManagersForTesting();

/// The device resolver every descriptor-backed engine in this provider shares.
/// Process-lifetime: a device-property cache with no engine-specific state.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

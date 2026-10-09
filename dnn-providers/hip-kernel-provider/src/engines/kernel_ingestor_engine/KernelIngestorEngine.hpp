// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
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
/// HIPDNN_DESCRIPTOR_RUNTIME_DIR when it names one, then each HIPDNN_DESCRIPTOR_PATH entry.
/// Those are additive rather than an override -- they can add descriptors beside the
/// shipped ones, but a file redefining a shipped id is refused and the shipped definition
/// stands. Reads the environment once, so each unusable value is warned about once.
std::vector<std::filesystem::path> descriptorSearchDirectories();

/// Every descriptor file under those directories, parsed once. Shared with opaque engines
/// resolving their declared UHD id so the tree is walked only once.
const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& descriptorCatalog();

/// Every descriptor set this provider serves. Registers symbols first so validation can
/// check each descriptor's symbol exists -- a set returns only if buildable, which lets
/// Container::copyEngineIds advertise ids before any engine is constructed. A malformed
/// descriptor costs its pack, never the provider. Memoized, so two scans can't disagree.
const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets();

/// The immutable kernel-ingestor state discovery built while validating the
/// discoverDescriptorSets() entry whose engine id is @p engineId, or null when no set has
/// that id. Built once per process from the descriptor tree and shared, never copied: every
/// engine constructed for that set -- one per Container, so one per hipDNN handle -- wraps
/// it in a state manager of its own (own catalog and winner caches), so constructing a
/// Container costs no per-kernel work. Callers holding the pointer keep the state alive
/// across a rediscoverSharedStatesForTesting(). Thread-safe.
std::shared_ptr<const hipdnn_plugin_sdk::ingestor::KernelIngestorSharedState<Handle>>
    discoveredSharedState(const hipdnn_plugin_sdk::ingestor::DescriptorId& engineId);

/// Test-only. Re-validates descriptorCatalog() through the same path discovery takes and
/// replaces every discoveredSharedState() with the state that pass built; the next
/// Container serves the new states, while engines already constructed keep the ones they
/// hold. Thread-safe against concurrent lookups.
/// @throws std::logic_error the pass does not reproduce discoverDescriptorSets() in order;
///         the states are left unchanged.
void rediscoverSharedStatesForTesting();

/// The device resolver every descriptor-backed engine in this provider shares.
/// Process-lifetime: a device-property cache with no engine-specific state.
const HandleDeviceResolver& deviceResolver();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @file PointwiseAddPack.hpp
 * @brief The pointwise-add descriptor set, built in memory.
 *
 * Stands in for what a loader will produce from installed files: one engine (UED) with
 * its metadata schema (KMD) and heuristic (UHD), two matchers (UMDs), one dispatch
 * descriptor (UDD), and one pack (KDP) binding them over three kernels (UKDs).
 *
 * Nothing here is loaded, parsed, or validated against a schema — that is ALMIOPEN-2401
 * and the packaging follow-ups. What this demonstrates is that once descriptors exist in
 * memory, everything downstream of them works: matching, pruning, ranking, knob
 * reporting, workspace sizing, and launch are all driven by this data.
 *
 * The three kernels are chosen so each of those steps has something real to do:
 * two differ only in block size, so ranking has an order to produce and the knob has a
 * value set of two; the third targets a different dtype, so the kernel-scoped matcher
 * has something to prune.
 */

/// A pack plus every descriptor it references by id, ready to construct a state manager.
struct PointwiseAddDescriptorSet
{
    hipdnn_plugin_sdk::ingestor::EngineDescriptor engine;
    hipdnn_plugin_sdk::ingestor::MetadataSchema schema;
    std::vector<hipdnn_plugin_sdk::ingestor::MatchDescriptor> matchers;
    std::vector<hipdnn_plugin_sdk::ingestor::DispatchDescriptor> dispatches;
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDescriptorPack> packs;
    hipdnn_plugin_sdk::ingestor::HeuristicDescriptor heuristic;
};

/// @brief Builds this pack's descriptor set.
PointwiseAddDescriptorSet buildPointwiseAddDescriptorSet();

/**
 * @brief This engine's hipDNN engine id, registering its name on first call.
 *
 * A descriptor-backed engine's id is its UED name hashed into hipDNN's engine-id space,
 * the same derivation a hand-written engine's registered name goes through. Because the
 * engine is defined by data rather than by a compile-time registration macro, the name
 * is registered at run time.
 *
 * Registration is deliberately reached through a function-local static rather than a
 * namespace-scope object. The registrar throws on a name or hash collision, and a throw
 * from a namespace-scope initializer would escape a global constructor during dlopen()
 * and terminate the process. Reached from here — called by the engine table, which runs
 * inside the plugin's create entry point — the same collision surfaces as a failed
 * plugin creation the host can report.
 */
int64_t pointwiseAddEngineId();

/// @brief Builds the state manager backing this pack.
///
/// Carries only what selection needs: the KMD, the matchers and dispatch descriptors the
/// packs reference, the packs themselves, and the resolved heuristic. The UED is not
/// among them, because a UED is 1:1 with a hipDNN engine and is owned by the engine.
std::unique_ptr<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>>
    makePointwiseAddStateManager();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

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
 * Stands in for what a loader will produce from installed files: one engine (UED),
 * its metadata schema (KMD) and heuristic (UHD), two matchers (UMDs), one dispatch
 * descriptor (UDD), and one pack (KDP) binding them over three kernels (UKDs).
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
 * The id is the UED name hashed into hipDNN's engine-id space, registered at run time
 * since the engine is defined by data. A function-local static, not a namespace-scope
 * object: the registrar throws on a collision, which from a global constructor during
 * dlopen() would terminate the process.
 */
int64_t pointwiseAddEngineId();

/**
 * @brief Builds the state manager backing this pack, from an already-built
 *        descriptor set.
 *
 * Takes @p set by value so a caller building both the engine and its state manager
 * calls buildPointwiseAddDescriptorSet() once. @p set's UED is ignored; the engine
 * owns it.
 */
std::unique_ptr<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>>
    makePointwiseAddStateManager(PointwiseAddDescriptorSet set);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

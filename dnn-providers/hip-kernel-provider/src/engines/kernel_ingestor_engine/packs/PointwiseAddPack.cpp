// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/PointwiseAddPack.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;

namespace
{

// Descriptor ids are stable GUIDs, minted once for this pack and never regenerated.
// Authored descriptors will carry the same values as text, parsed with parseUuid().
const DescriptorId ENGINE_ID = parseUuid("8f1a6c30-7d24-4a0e-9b51-2c6d84f0a911");
const DescriptorId SCHEMA_ID = parseUuid("b27e4d15-3f89-4c62-8a07-51de93b4c7a2");
const DescriptorId HEURISTIC_ID = parseUuid("4c9b0e72-16a5-4d38-b6f4-8e20a7c15d63");
const DescriptorId GRAPH_MATCHER_ID = parseUuid("e5d38a04-9c71-4b2f-83a6-7f14b0d29e58");
const DescriptorId KERNEL_MATCHER_ID = parseUuid("1a7f52c8-b036-4e91-95d2-6c83af407b14");
const DescriptorId DISPATCH_ID = parseUuid("93b6c1e7-2058-4fa3-8d17-4b95e2c60fa8");
const DescriptorId PACK_ID = parseUuid("7d024fb9-5e13-4c86-a92b-08f37d1c4e50");

KernelDescriptor makeKernel(const DescriptorId& id,
                            const std::string& variant,
                            int64_t blockSize,
                            const std::string& dtype,
                            int64_t priority)
{
    KernelDescriptor kernel;
    kernel.id = id;
    kernel.name = "pointwise_add." + variant;
    // The only KernelSourceKind this pack implements; see KernelSource's doc for the others.
    kernel.source.sourceFile = "PointwiseAdd.cpp";
    kernel.source.entryPoint = "PointwiseAdd";
    kernel.metadata = {{std::string(BLOCK_SIZE_FIELD), MetadataValue{blockSize}},
                       {std::string(DTYPE_FIELD), MetadataValue{dtype}}};
    kernel.priority = priority;
    return kernel;
}

} // namespace

PointwiseAddDescriptorSet buildPointwiseAddDescriptorSet()
{
    PointwiseAddDescriptorSet set;

    set.schema.id = SCHEMA_ID;
    set.schema.name = "pointwise add variant fields";
    // block_size defaults; dtype does not, since an omitted one would silently inherit
    // another kernel's.
    set.schema.fields = {{std::string(BLOCK_SIZE_FIELD), MetadataType::INT, int64_t{64}},
                         {std::string(DTYPE_FIELD), MetadataType::STRING, std::nullopt}};

    set.heuristic.id = HEURISTIC_ID;
    set.heuristic.name = "pointwise add selector";
    set.heuristic.payload = SCORE_SYMBOL;

    set.engine.id = ENGINE_ID;
    set.engine.name = ENGINE_NAME;
    set.engine.heuristicId = HEURISTIC_ID;
    set.engine.metadataSchemaId = SCHEMA_ID;
    // block_size is exposed to the caller; dtype is pinned by the graph, not chosen.
    set.engine.knobs = {std::string(BLOCK_SIZE_FIELD)};
    // This engine's dispatch handler compiles through hiprtc at plan build; a pack
    // shipping prebuilt kernels would not declare this.
    set.engine.behaviorNotes = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};

    set.matchers = {
        {GRAPH_MATCHER_ID,
         "single-node pointwise add over 1-element tensors",
         MatchScope::GRAPH,
         std::string(GRAPH_MATCHER_SYMBOL)},
        {KERNEL_MATCHER_ID,
         "kernel dtype matches the graph's dtype",
         MatchScope::KERNEL,
         std::string(KERNEL_MATCHER_SYMBOL)},
    };

    set.dispatches = {
        {DISPATCH_ID, "pointwise add dispatch", std::string(DISPATCH_SYMBOL)},
    };

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "hipkernel:pointwise_add";
    pack.matcherIds = {GRAPH_MATCHER_ID, KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    // The two FLOAT entries differ only in block size, giving ranking an order; the
    // HALF entry is what the kernel-scoped matcher prunes on a FLOAT graph.
    pack.kernels
        = {makeKernel(
               parseUuid("2f8c17d6-4a90-4b53-9e18-c05a63b782d4"), "f32_block64", 64, "FLOAT", 0),
           makeKernel(
               parseUuid("a41b93e5-6d72-4f08-b3c9-15e847d0629f"), "f32_block256", 256, "FLOAT", 0),
           makeKernel(
               parseUuid("c6e20a48-8b31-4d97-a052-9f4173e8b5c1"), "f16_block64", 64, "HALF", 0)};
    set.packs = {std::move(pack)};

    return set;
}

int64_t pointwiseAddEngineId()
{
    // Registers the engine name against its id on first call; see the header for why
    // this is a function-local static.
    static const hipdnn_data_sdk::utilities::EngineRegistrar s_registrar{ENGINE_NAME};
    return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
}

std::unique_ptr<KernelIngestorStateManager<Handle>>
    makePointwiseAddStateManager(PointwiseAddDescriptorSet set)
{
    return std::make_unique<KernelIngestorStateManager<Handle>>(std::move(set.schema),
                                                                std::move(set.matchers),
                                                                std::move(set.dispatches),
                                                                std::move(set.packs),
                                                                makeKernelHeuristic(set.heuristic));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

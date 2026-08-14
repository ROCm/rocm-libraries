// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/// @file PointwiseAddDescriptors.cpp
/// The pointwise-add descriptor set, built in memory: stands in for what a loader
/// will produce from installed files (one engine, schema, heuristic, two matchers,
/// one dispatch descriptor, one pack over three kernels).
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;

namespace
{

constexpr std::string_view ENGINE_NAME = "hipkernel:PointwiseAdd";

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.pointwise_add.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.pointwise_add.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.pointwise_add.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.pointwise_add.dispatch";

constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";

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
    kernel.source.sourceFile = "PointwiseAdd.cpp";
    kernel.source.entryPoint = "PointwiseAdd";
    kernel.metadata = {{std::string(BLOCK_SIZE_FIELD), MetadataValue{blockSize}},
                       {std::string(DTYPE_FIELD), MetadataValue{dtype}}};
    kernel.priority = priority;
    return kernel;
}

} // namespace

hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseAddDescriptorSet()
{
    hipdnn_plugin_sdk::ingestor::DescriptorSet set;

    set.schema.id = SCHEMA_ID;
    set.schema.name = "pointwise add variant fields";
    // dtype has no default, unlike block_size, to avoid inheriting another kernel's.
    set.schema.fields = {{std::string(BLOCK_SIZE_FIELD), MetadataType::INT, int64_t{64}},
                         {std::string(DTYPE_FIELD), MetadataType::STRING, std::nullopt}};

    HeuristicDescriptor heuristic;
    heuristic.id = HEURISTIC_ID;
    heuristic.name = "pointwise add selector";
    heuristic.payload = SCORE_SYMBOL;
    set.heuristic = std::move(heuristic);

    set.engine.id = ENGINE_ID;
    set.engine.name = ENGINE_NAME;
    set.engine.heuristicId = HEURISTIC_ID;
    set.engine.metadataSchemaId = SCHEMA_ID;
    // dtype is pinned by the graph, not chosen.
    set.engine.knobs = {std::string(BLOCK_SIZE_FIELD)};
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
    // HALF is pruned by the kernel-scoped matcher on a FLOAT graph.
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

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

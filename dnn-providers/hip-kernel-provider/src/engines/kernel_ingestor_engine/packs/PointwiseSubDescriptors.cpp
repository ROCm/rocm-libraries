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

/// @file PointwiseSubDescriptors.cpp
/// The pointwise-subtract descriptor set, built in memory. A second engine, distinct
/// ids from the add pack's: descriptors reference each other by id, and both sets
/// load into one provider.
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;

namespace
{

constexpr std::string_view ENGINE_NAME = "hipkernel:PointwiseSub";

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.pointwise_sub.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.pointwise_sub.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.pointwise_sub.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.pointwise_sub.dispatch";

constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";

const DescriptorId ENGINE_ID = parseUuid("d49308c3-5e9e-4fea-8bc5-4ee6aec001fe");
const DescriptorId SCHEMA_ID = parseUuid("7859ec6c-b65f-44e6-9732-b95bdadedb77");
const DescriptorId HEURISTIC_ID = parseUuid("f02bf944-c856-44fe-99f5-b0fe563bac6b");
const DescriptorId GRAPH_MATCHER_ID = parseUuid("dc67dde9-b2a1-470e-80c9-4ee84c1e3b25");
const DescriptorId KERNEL_MATCHER_ID = parseUuid("024f1138-2f1b-4658-bf3e-d21ee1b2397c");
const DescriptorId DISPATCH_ID = parseUuid("8bd42efc-96dc-4de3-84a9-9cbeed5fc420");
const DescriptorId PACK_ID = parseUuid("530216cb-d708-4097-9391-d81882633b3b");

KernelDescriptor makeKernel(const DescriptorId& id,
                            const std::string& variant,
                            int64_t blockSize,
                            const std::string& dtype,
                            int64_t priority)
{
    KernelDescriptor kernel;
    kernel.id = id;
    kernel.name = "pointwise_sub." + variant;
    kernel.source.sourceFile = "PointwiseSub.cpp";
    kernel.source.entryPoint = "PointwiseSub";
    kernel.metadata = {{std::string(BLOCK_SIZE_FIELD), MetadataValue{blockSize}},
                       {std::string(DTYPE_FIELD), MetadataValue{dtype}}};
    kernel.priority = priority;
    return kernel;
}

} // namespace

hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseSubDescriptorSet()
{
    hipdnn_plugin_sdk::ingestor::DescriptorSet set;

    set.schema.id = SCHEMA_ID;
    set.schema.name = "pointwise sub variant fields";
    // dtype has no default, to avoid inheriting another kernel's.
    set.schema.fields = {{std::string(BLOCK_SIZE_FIELD), MetadataType::INT, int64_t{64}},
                         {std::string(DTYPE_FIELD), MetadataType::STRING, std::nullopt}};

    set.heuristic.id = HEURISTIC_ID;
    set.heuristic.name = "pointwise sub selector";
    set.heuristic.payload = SCORE_SYMBOL;

    set.engine.id = ENGINE_ID;
    set.engine.name = ENGINE_NAME;
    set.engine.heuristicId = HEURISTIC_ID;
    set.engine.metadataSchemaId = SCHEMA_ID;
    // dtype is pinned by the graph, not chosen.
    set.engine.knobs = {std::string(BLOCK_SIZE_FIELD)};
    set.engine.behaviorNotes = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};

    set.matchers = {
        {GRAPH_MATCHER_ID,
         "single-node pointwise sub over 1-element tensors",
         MatchScope::GRAPH,
         std::string(GRAPH_MATCHER_SYMBOL)},
        {KERNEL_MATCHER_ID,
         "kernel dtype matches the graph's dtype",
         MatchScope::KERNEL,
         std::string(KERNEL_MATCHER_SYMBOL)},
    };

    set.dispatches = {
        {DISPATCH_ID, "pointwise sub dispatch", std::string(DISPATCH_SYMBOL)},
    };

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "hipkernel:pointwise_sub";
    pack.matcherIds = {GRAPH_MATCHER_ID, KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    // HALF is pruned by the kernel-scoped matcher on a FLOAT graph.
    pack.kernels
        = {makeKernel(
               parseUuid("f8da0a47-fed7-453d-97ff-f5d3e0f382e1"), "f32_block64", 64, "FLOAT", 0),
           makeKernel(
               parseUuid("70cd25c3-f20c-4036-868f-ceffb938231f"), "f32_block256", 256, "FLOAT", 0),
           makeKernel(
               parseUuid("32c65a6b-06ec-4fab-aacc-75b00f531350"), "f16_block64", 64, "HALF", 0)};
    set.packs = {std::move(pack)};

    return set;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ingestor_poc/PointwiseAddPack.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "ingestor_poc/NativeSymbolNames.hpp"

namespace hip_kernel_provider::ingestor_poc
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

// Descriptor ids. Real descriptors carry GUIDs so any author can mint one locally
// without colliding with another's; readable names serve the same role for a pack that
// is built in memory and never shared.
constexpr const char* ENGINE_ID = "poc.ued.pointwise_add";
constexpr const char* SCHEMA_ID = "poc.kmd.pointwise_add";
constexpr const char* HEURISTIC_ID = "poc.uhd.pointwise_add";
constexpr const char* GRAPH_MATCHER_ID = "poc.umd.pointwise_add.graph";
constexpr const char* KERNEL_MATCHER_ID = "poc.umd.pointwise_add.kernel";
constexpr const char* DISPATCH_ID = "poc.udd.pointwise_add";
constexpr const char* PACK_ID = "poc.kdp.pointwise_add";

KernelDescriptor
    makeKernel(const std::string& id, int64_t blockSize, const std::string& dtype, int64_t priority)
{
    KernelDescriptor kernel;
    kernel.id = id;
    kernel.name = "pointwise add (" + dtype + ", block " + std::to_string(blockSize) + ")";
    kernel.sourceFile = "PointwiseAdd.cpp";
    kernel.entryPoint = "PointwiseAdd";
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    kernel.priority = priority;
    return kernel;
}

} // namespace

PointwiseAddDescriptorSet buildPointwiseAddDescriptorSet()
{
    PointwiseAddDescriptorSet set;

    set.schema.id = SCHEMA_ID;
    set.schema.name = "pointwise add variant fields";
    set.schema.fields = {{std::string(BLOCK_SIZE_FIELD), int64_t{64}},
                         {std::string(DTYPE_FIELD), std::string{"FLOAT"}}};

    set.heuristic.id = HEURISTIC_ID;
    set.heuristic.name = "pointwise add selector";
    set.heuristic.scoreSymbol = SCORE_SYMBOL;

    set.engine.id = ENGINE_ID;
    set.engine.name = ENGINE_NAME;
    set.engine.heuristicId = HEURISTIC_ID;
    set.engine.metadataSchemaId = SCHEMA_ID;
    // block_size is exposed to the caller; dtype is not, because it is pinned by the
    // graph rather than chosen. Exposing it would offer a choice nothing can serve.
    set.engine.knobs = {std::string(BLOCK_SIZE_FIELD)};

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
    pack.name = "kernel_ingestor_poc:pointwise_add";
    pack.matcherIds = {GRAPH_MATCHER_ID, KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    // Three kernels, each earning its place: the two FLOAT entries differ only in block
    // size, so ranking has an order to produce and the block_size knob has a real value
    // set; the HALF entry is what the kernel-scoped matcher prunes on a FLOAT graph.
    pack.kernels = {makeKernel("poc.ukd.add_f32_block64", 64, "FLOAT", 0),
                    makeKernel("poc.ukd.add_f32_block256", 256, "FLOAT", 0),
                    makeKernel("poc.ukd.add_f16_block64", 64, "HALF", 0)};
    set.packs = {std::move(pack)};

    return set;
}

int64_t pointwiseAddEngineId()
{
    // Registers the engine name against its id on first call, so diagnostics can print
    // a name instead of a hex value and a colliding descriptor is named. See the header
    // for why this is a function-local static rather than a namespace-scope object.
    static const hipdnn_data_sdk::utilities::EngineRegistrar s_registrar{ENGINE_NAME};
    return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
}

std::shared_ptr<KernelIngestorStateManager<Handle>> makePointwiseAddStateManager()
{
    auto set = buildPointwiseAddDescriptorSet();

    return std::make_shared<KernelIngestorStateManager<Handle>>(
        std::move(set.engine),
        std::move(set.schema),
        std::move(set.matchers),
        std::move(set.dispatches),
        std::move(set.packs),
        std::make_shared<NativeKernelHeuristic>(ScoreRegistry::resolve(std::string(SCORE_SYMBOL))));
}

} // namespace hip_kernel_provider::ingestor_poc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

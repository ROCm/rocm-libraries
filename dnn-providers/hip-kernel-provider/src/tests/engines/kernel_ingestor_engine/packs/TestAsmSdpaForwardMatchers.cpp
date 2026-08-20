// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#ifdef HIPDNN_ENGINE_ASM_SDPA

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestAsmSdpaForwardMatchers.cpp
 * @brief The ASM SDPA forward engine's matchers, run against a *declared* gfx942 rather
 *        than a real one.
 *
 * This engine's packs arch-prune everywhere but gfx942 and gfx950, so on any other
 * machine its matchers never execute and every test of them passes vacuously. That is
 * how a matcher that declines every graph reached hardware: the engine loaded, validated
 * and enumerated perfectly, and the first thing that actually ran it was an MI300X.
 *
 * `MatchContext` takes DeviceProperties by value, so the arch a matcher sees is data, not
 * a property of this host. These tests supply gfx942 and call the registered symbols
 * directly, which puts the matchers under test on every machine that builds the provider.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* GRAPH_MATCHER_SYMBOL = "hipkernel.asm_sdpa_fwd.graph_match";
constexpr const char* KERNEL_MATCHER_SYMBOL = "hipkernel.asm_sdpa_fwd.kernel_match";

/// What a gfx942 device reports, feature suffix and all.
DeviceProperties gfx942Properties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942:sramecc+:xnack-";
    properties.warpSize = 64;
    properties.multiProcessorCount = 304;
    return properties;
}

/// A bf16 forward graph of the shape the CSV's first gfx942 row serves: hd128, no mask,
/// batch mode. If anything at all is servable by this engine, this is.
flatbuffers::FlatBufferBuilder makeServableGraph()
{
    const std::vector<int64_t> dims{1, 1, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               data_objects::DataType::BFLOAT16,
                                                               data_objects::DataType::FLOAT);
}

/// Registers the provider's native symbols, exactly as plugin construction does.
void ensureSymbolsRegistered()
{
    static const bool s_registered = [] {
        static_cast<void>(hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets());
        return true;
    }();
    static_cast<void>(s_registered);
}

} // namespace

TEST(TestAsmSdpaForwardMatchers, AcceptsAServableGraphOnGfx942)
{
    ensureSymbolsRegistered();

    auto builder = makeServableGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto properties = gfx942Properties();
    const MatchContext context{graph, 0, properties};

    const auto& matcher = GraphMatchRegistry::resolve(GRAPH_MATCHER_SYMBOL);

    // The claim the engine's whole existence rests on. A refusal here is the engine
    // declining every SDPA graph on the one architecture it ships kernels for.
    EXPECT_TRUE(matcher(context).has_value()) << "graph matcher declined a graph the CSV serves";
}

TEST(TestAsmSdpaForwardMatchers, BindsTheOperandUidsDispatchReads)
{
    ensureSymbolsRegistered();

    auto builder = makeServableGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto properties = gfx942Properties();
    const MatchContext context{graph, 0, properties};

    const auto& matcher = GraphMatchRegistry::resolve(GRAPH_MATCHER_SYMBOL);
    const auto bound = matcher(context);
    ASSERT_TRUE(bound.has_value());

    // prepare() re-reads these rather than re-deriving them, so a match that binds
    // nothing throws at plan build instead of declining here.
    EXPECT_TRUE(tryGetBoundInt(*bound, "asm_sdpa_fwd.q.uid").has_value());
    EXPECT_TRUE(tryGetBoundInt(*bound, "asm_sdpa_fwd.k.uid").has_value());
    EXPECT_TRUE(tryGetBoundInt(*bound, "asm_sdpa_fwd.v.uid").has_value());
    EXPECT_TRUE(tryGetBoundInt(*bound, "asm_sdpa_fwd.o.uid").has_value());
}

TEST(TestAsmSdpaForwardMatchers, KeepsAKernelWhoseShapeTheGraphAsksFor)
{
    ensureSymbolsRegistered();

    auto builder = makeServableGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto properties = gfx942Properties();
    const MatchContext context{graph, 0, properties};

    // The gfx942 pack, as installed. Its first row is hd128/no-mask, which is what
    // makeServableGraph() asks for.
    const auto& sets = hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets();
    const auto engine = std::find_if(sets.begin(), sets.end(), [](const auto& set) {
        return set.engine.name == "hipkernel:AsmSdpaForward";
    });
    ASSERT_NE(engine, sets.end()) << "the ASM SDPA forward engine is not installed";

    const auto pack = std::find_if(engine->packs.begin(), engine->packs.end(), [](const auto& p) {
        return !p.arch.empty() && p.arch.front() == "gfx942";
    });
    ASSERT_NE(pack, engine->packs.end()) << "no gfx942 pack";

    const auto& kernelMatcher = KernelMatcherRegistry::resolve(KERNEL_MATCHER_SYMBOL);
    const auto& graphMatcher = GraphMatchRegistry::resolve(GRAPH_MATCHER_SYMBOL);
    const auto bound = graphMatcher(context);
    ASSERT_TRUE(bound.has_value());

    bool anyKept = false;
    for(const auto& kernel : pack->kernels)
    {
        const KernelDefinition definition{kernel.id,
                                          pack->id,
                                          pack->dispatchId,
                                          kernel.source,
                                          kernel.metadata,
                                          kernel.priority,
                                          kernel.arch};
        anyKept |= kernelMatcher(context, *bound, definition);
    }

    EXPECT_TRUE(anyKept) << "every kernel in the gfx942 pack was pruned for a graph the CSV serves";
}

/// A backward graph with FLOAT compute, built by hand.
///
/// createValidSdpaBwdGraph() stamps the node's compute type from the tensor dtype, so it
/// cannot express bf16 tensors with fp32 compute -- which is the only shape this engine
/// serves. The builder this replaced hand-built its graphs for the same reason.
// NOLINTNEXTLINE(misc-use-internal-linkage)
static flatbuffers::FlatBufferBuilder makeServableBackwardGraph()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;

    const std::vector<int64_t> dims{1, 1, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto add = [&](const char* name,
                         DataType type,
                         const std::vector<int64_t>& d,
                         const std::vector<int64_t>& s) {
        const auto thisUid = uid++;
        tensors.push_back(CreateTensorAttributesDirect(builder, thisUid, name, type, &s, &d));
        return thisUid;
    };

    const auto qUid = add("q", DataType::BFLOAT16, dims, strides);
    const auto kUid = add("k", DataType::BFLOAT16, dims, strides);
    const auto vUid = add("v", DataType::BFLOAT16, dims, strides);
    const auto oUid = add("o", DataType::BFLOAT16, dims, strides);
    const auto doUid = add("do", DataType::BFLOAT16, dims, strides);

    const std::vector<int64_t> statsDims{dims[0], dims[1], dims[2], 1};
    const std::vector<int64_t> statsStrides{dims[1] * dims[2], dims[2], 1, 1};
    const auto statsUid = add("stats", DataType::FLOAT, statsDims, statsStrides);

    const auto dqUid = add("dq", DataType::BFLOAT16, dims, strides);
    const auto dkUid = add("dk", DataType::BFLOAT16, dims, strides);
    const auto dvUid = add("dv", DataType::BFLOAT16, dims, strides);

    auto attributes = CreateSdpaBackwardAttributes(
        builder, qUid, kUid, vUid, oUid, doUid, statsUid, dqUid, dkUid, dvUid);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     attributes.Union()));

    builder.Finish(CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, &tensors, &nodes));
    return builder;
}

TEST(TestAsmSdpaBackwardMatchers, AcceptsAServableGraphOnGfx942)
{
    ensureSymbolsRegistered();

    // Outputs (o, do, dq, dk, dv) are inferred by the frontend and may be unshaped when
    // matching runs, which is the point: a matcher requiring a rank-4 shape of them
    // declines every graph before shape inference, and only on the arch whose pack is
    // not pruned -- invisible on every other machine.
    auto builder = makeServableBackwardGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto properties = gfx942Properties();
    const MatchContext context{graph, 0, properties};

    const auto& matcher = GraphMatchRegistry::resolve("hipkernel.asm_sdpa_bwd.graph_match");

    EXPECT_TRUE(matcher(context).has_value())
        << "backward graph matcher declined a graph the CSV serves";
}

#endif // HIPDNN_ENGINE_ASM_SDPA
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

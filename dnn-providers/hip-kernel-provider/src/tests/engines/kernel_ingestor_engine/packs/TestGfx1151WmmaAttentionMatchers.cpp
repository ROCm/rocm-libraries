// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention.score";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t RAGGED_OFFSET_UID = 6;

struct GraphSpec
{
    int64_t batch = 2;
    int64_t queryHeads = 8;
    int64_t kvHeads = 2;
    int64_t queryLength = 33;
    int64_t kvLength = 65;
    int64_t headSize = 96;
    data_objects::DataType dtype = data_objects::DataType::HALF;
    bool bhsd = false;
    bool returnLse = true;
    data_objects::DataType lseDtype = data_objects::DataType::FLOAT;
    int64_t leftBound = 7;
    int64_t rightBound = 3;
    std::optional<int64_t> outputHeadSize;
    // Operand UID that carries a ragged (THD) offset table, if any.
    std::optional<int64_t> raggedOperand;
    bool causalMask = false;
    bool causalMaskBottomRight = false;
};

DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx1151";
    properties.warpSize = 32;
    return properties;
}

std::vector<int64_t> denseStrides(bool bhsd, int64_t heads, int64_t sequence, int64_t headSize)
{
    if(bhsd)
    {
        return {heads * sequence * headSize, sequence * headSize, headSize, 1};
    }
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> qDims{spec.batch, spec.queryHeads, spec.queryLength, spec.headSize};
    const std::vector<int64_t> kDims{spec.batch, spec.kvHeads, spec.kvLength, spec.headSize};
    const std::vector<int64_t>& vDims = kDims;
    const std::vector<int64_t> oDims{
        spec.batch, spec.queryHeads, spec.queryLength, spec.outputHeadSize.value_or(spec.headSize)};
    const std::vector<int64_t> lseDims{spec.batch, spec.queryHeads, spec.queryLength, 1};

    const auto qStrides = denseStrides(spec.bhsd, spec.queryHeads, spec.queryLength, spec.headSize);
    const auto kStrides = denseStrides(spec.bhsd, spec.kvHeads, spec.kvLength, spec.headSize);
    const auto& vStrides = kStrides;
    const auto oStrides = denseStrides(
        spec.bhsd, spec.queryHeads, spec.queryLength, spec.outputHeadSize.value_or(spec.headSize));
    const auto lseStrides = denseStrides(spec.bhsd, spec.queryHeads, spec.queryLength, 1);

    const auto tensorFor = [&](int64_t uid,
                               data_objects::DataType dtype,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dims) {
        const ::flatbuffers::Optional<int64_t> ragged
            = spec.raggedOperand == uid ? ::flatbuffers::Optional<int64_t>(RAGGED_OFFSET_UID)
                                        : ::flatbuffers::nullopt;
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          dtype,
                                                          &strides,
                                                          &dims,
                                                          false,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          false,
                                                          ragged);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        tensorFor(Q_UID, spec.dtype, qStrides, qDims),
        tensorFor(K_UID, spec.dtype, kStrides, kDims),
        tensorFor(V_UID, spec.dtype, vStrides, vDims),
        tensorFor(O_UID, spec.dtype, oStrides, oDims),
    };
    if(spec.returnLse)
    {
        tensors.push_back(tensorFor(LSE_UID, spec.lseDtype, lseStrides, lseDims));
    }
    if(spec.raggedOperand.has_value())
    {
        const std::vector<int64_t> offsetDims{spec.batch + 1, 1, 1, 1};
        const std::vector<int64_t> offsetStrides{1, 1, 1, 1};
        tensors.push_back(
            tensorFor(RAGGED_OFFSET_UID, data_objects::DataType::INT32, offsetStrides, offsetDims));
    }

    data_objects::SdpaAttributesBuilder attributes(builder);
    attributes.add_q_tensor_uid(Q_UID);
    attributes.add_k_tensor_uid(K_UID);
    attributes.add_v_tensor_uid(V_UID);
    attributes.add_o_tensor_uid(O_UID);
    if(spec.returnLse)
    {
        attributes.add_stats_tensor_uid(LSE_UID);
        attributes.add_generate_stats(true);
    }
    attributes.add_left_bound(spec.leftBound);
    attributes.add_causal_mask(spec.causalMask);
    attributes.add_causal_mask_bottom_right(spec.causalMaskBottomRight);
    attributes.add_right_bound(spec.rightBound);
    attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
    attributes.add_attn_scale_value(0.125F);
    attributes.add_implementation(data_objects::AttentionImplementation::AUTO);
    attributes.add_mma_core_mode(data_objects::DataType::FLOAT);

    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "sdpa",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::SdpaAttributes,
                                       attributes.Finish().Union())};

    const auto name = builder.CreateString("gfx1151_wmma_attention_test");
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graph(builder);
    graph.add_name(name);
    graph.add_tensors(tensorVector);
    graph.add_nodes(nodeVector);
    builder.Finish(graph.Finish());
    return builder;
}

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    auto builder = buildSdpaGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    return matcher(MatchContext{graph, 0, properties});
}

struct KernelSpec
{
    std::string dtype = "FP16";
    int64_t headSize = 96;
    std::string maskMode = "window";
    int64_t transposedQk = 0;
    int64_t queryTail = 1;
    int64_t kvTail = 1;
    int64_t blockN = 32;
    int64_t numWaves = 1;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000003");
    kernel.metadata = {{"dtype", spec.dtype},
                       {"head_size", spec.headSize},
                       {"mask_mode", spec.maskMode},
                       {"transposed_qk", spec.transposedQk},
                       {"query_tail", spec.queryTail},
                       {"kv_tail", spec.kvTail},
                       {"block_n", spec.blockN},
                       {"num_waves", spec.numWaves}};
    return kernel;
}

bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
    auto builder = buildSdpaGraph(graphSpec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const auto bound = graphMatcher(context);
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(context, *bound, makeKernel(kernelSpec));
}

TEST(TestGfx1151WmmaAttentionMatchers, FixtureConstructsGfx1151DeviceByValue)
{
    const auto properties = testDeviceProperties();
    EXPECT_EQ(properties.gcnArchName, "gfx1151");
    EXPECT_EQ(properties.warpSize, 32);
}

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsD96BhsdWindowAndLse)
{
    GraphSpec spec;
    spec.bhsd = true;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsBf16WithoutLse)
{
    GraphSpec spec;
    spec.dtype = data_objects::DataType::BFLOAT16;
    spec.returnLse = false;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx1151WmmaAttentionMatchers, DeclinesMismatchedOutputShape)
{
    GraphSpec spec;
    spec.outputHeadSize = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx1151WmmaAttentionMatchers, DeclinesHalfPrecisionLse)
{
    GraphSpec spec;
    spec.lseDtype = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// A ragged operand addresses each batch from a device offset table. The engine
// binds dense strides only, so binding one would silently read it as dense.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesRaggedOperands)
{
    ASSERT_TRUE(matchGraph(GraphSpec{}).has_value());
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID, LSE_UID})
    {
        GraphSpec spec;
        spec.raggedOperand = uid;
        EXPECT_FALSE(matchGraph(spec).has_value()) << "ragged operand uid " << uid;
    }
}

// The deprecated causal flags override the bounds and alignment, as in the
// hipDNN reference; otherwise a left bound would drop the causal limit.
TEST(TestGfx1151WmmaAttentionMatchers, DeprecatedCausalFlagsOverrideBounds)
{
    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    for(const bool bottomRight : {false, true})
    {
        GraphSpec spec;
        spec.causalMask = !bottomRight;
        spec.causalMaskBottomRight = bottomRight;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.mask"), 1); // causal
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_left"), -1);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_right"), 0);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bottom_right"),
                  bottomRight ? 1 : 0);
        auto causal = KernelSpec{};
        causal.maskMode = "causal";
        auto window = KernelSpec{};
        window.maskMode = "window";
        EXPECT_TRUE(matchesKernel(spec, causal));
        EXPECT_FALSE(matchesKernel(spec, window));
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, KernelMatcherUsesRuntimeMaskDtypeAndHeadSize)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));

    auto wrongMask = KernelSpec{};
    wrongMask.maskMode = "causal";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, wrongMask));

    auto wrongHeadSize = KernelSpec{};
    wrongHeadSize.headSize = 128;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, wrongHeadSize));

    GraphSpec bf16Graph;
    bf16Graph.dtype = data_objects::DataType::BFLOAT16;
    bf16Graph.headSize = 64;
    bf16Graph.queryLength = 32;
    bf16Graph.kvLength = 64;
    bf16Graph.leftBound = -1;
    bf16Graph.rightBound = 0;
    KernelSpec bf16Kernel;
    bf16Kernel.dtype = "BF16";
    bf16Kernel.headSize = 64;
    bf16Kernel.maskMode = "causal";
    bf16Kernel.transposedQk = 1;
    bf16Kernel.queryTail = 0;
    bf16Kernel.kvTail = 0;
    EXPECT_TRUE(matchesKernel(bf16Graph, bf16Kernel));
}

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersFastTransposedKernel)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    auto builder = buildSdpaGraph(GraphSpec{});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const KernelSpec standard;
    KernelSpec fast;
    fast.transposedQk = 1;
    EXPECT_GT(scorer(context, BoundTokens{}, makeKernel(fast)),
              scorer(context, BoundTokens{}, makeKernel(standard)));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

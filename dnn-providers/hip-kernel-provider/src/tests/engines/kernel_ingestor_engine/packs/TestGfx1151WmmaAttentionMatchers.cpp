// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
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
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention.score";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t RAGGED_OFFSET_UID = 6;
constexpr int64_t SCALE_UID = 7;
constexpr int64_t BIAS_UID = 8;

enum class ScaleForm
{
    VALUE, // attn_scale_value = 0.125
    ABSENT, // neither form: hipDNN default 1.0
    CONSTANT_TENSOR, // pass-by-value tensor holding 0.25
    RUNTIME_TENSOR, // runtime pass-by-value host scalar
    DEVICE_TENSOR, // ordinary device tensor (unsupported)
    VALUE_AND_TENSOR, // both forms at once (ill-formed)
};

// An attn_mask tensor: an additive bias over [B|1, H|1, Sq|1, Sk].
struct BiasSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype = data_objects::DataType::FLOAT;
    bool passByValue = false;
};

// Packed strides for @p dims, innermost axis last.
std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

BiasSpec packedBias(std::vector<int64_t> dims,
                    data_objects::DataType dtype = data_objects::DataType::FLOAT)
{
    auto strides = packedStrides(dims);
    return BiasSpec{std::move(dims), std::move(strides), dtype, false};
}

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
    // V head count and head width; K's when absent.
    std::optional<int64_t> vHeads;
    std::optional<int64_t> vHeadSize;
    std::optional<BiasSpec> bias;
    // Operand UID that carries a ragged (THD) offset table, if any.
    std::optional<int64_t> raggedOperand;
    bool causalMask = false;
    bool causalMaskBottomRight = false;
    ScaleForm scale = ScaleForm::VALUE;
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
    const int64_t vHeads = spec.vHeads.value_or(spec.kvHeads);
    const int64_t vHeadSize = spec.vHeadSize.value_or(spec.headSize);
    const int64_t oHeadSize = spec.outputHeadSize.value_or(vHeadSize);
    const std::vector<int64_t> qDims{spec.batch, spec.queryHeads, spec.queryLength, spec.headSize};
    const std::vector<int64_t> kDims{spec.batch, spec.kvHeads, spec.kvLength, spec.headSize};
    const std::vector<int64_t> vDims{spec.batch, vHeads, spec.kvLength, vHeadSize};
    const std::vector<int64_t> oDims{spec.batch, spec.queryHeads, spec.queryLength, oHeadSize};
    const std::vector<int64_t> lseDims{spec.batch, spec.queryHeads, spec.queryLength, 1};

    const auto qStrides = denseStrides(spec.bhsd, spec.queryHeads, spec.queryLength, spec.headSize);
    const auto kStrides = denseStrides(spec.bhsd, spec.kvHeads, spec.kvLength, spec.headSize);
    const auto vStrides = denseStrides(spec.bhsd, vHeads, spec.kvLength, vHeadSize);
    const auto oStrides = denseStrides(spec.bhsd, spec.queryHeads, spec.queryLength, oHeadSize);
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
    if(spec.bias.has_value())
    {
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       BIAS_UID,
                                                       nullptr,
                                                       spec.bias->dtype,
                                                       &spec.bias->strides,
                                                       &spec.bias->dims,
                                                       false,
                                                       data_objects::TensorValue::NONE,
                                                       0,
                                                       spec.bias->passByValue));
    }
    const std::vector<int64_t> scalarDims{1};
    switch(spec.scale)
    {
    case ScaleForm::CONSTANT_TENSOR:
    case ScaleForm::VALUE_AND_TENSOR:
    {
        const data_objects::Float32Value value(0.25F);
        const auto valueOffset = builder.CreateStruct(value).Union();
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       SCALE_UID,
                                                       nullptr,
                                                       data_objects::DataType::FLOAT,
                                                       &scalarDims,
                                                       &scalarDims,
                                                       false,
                                                       data_objects::TensorValue::Float32Value,
                                                       valueOffset));
        break;
    }
    case ScaleForm::RUNTIME_TENSOR:
    case ScaleForm::DEVICE_TENSOR:
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       SCALE_UID,
                                                       nullptr,
                                                       data_objects::DataType::FLOAT,
                                                       &scalarDims,
                                                       &scalarDims,
                                                       false,
                                                       data_objects::TensorValue::NONE,
                                                       0,
                                                       spec.scale == ScaleForm::RUNTIME_TENSOR));
        break;
    default:
        break;
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
    if(spec.scale == ScaleForm::VALUE || spec.scale == ScaleForm::VALUE_AND_TENSOR)
    {
        attributes.add_attn_scale_value(0.125F);
    }
    if(spec.scale != ScaleForm::VALUE && spec.scale != ScaleForm::ABSENT)
    {
        attributes.add_scale_tensor_uid(SCALE_UID);
    }
    if(spec.bias.has_value())
    {
        attributes.add_attn_mask_tensor_uid(BIAS_UID);
    }
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
    int64_t runtimeHeadDims = 0;
    int64_t useAttnBias = 0;
    std::string biasDtype = "f32";
};

// A standard-path object compiled for head widths up to @p bucket, as shipped.
KernelSpec runtimeKernel(int64_t bucket,
                         std::string maskMode = "window",
                         std::optional<std::string> biasDtype = std::nullopt)
{
    KernelSpec spec;
    spec.headSize = bucket;
    spec.maskMode = std::move(maskMode);
    spec.runtimeHeadDims = 1;
    if(biasDtype.has_value())
    {
        spec.useAttnBias = 1;
        spec.biasDtype = *biasDtype;
    }
    return spec;
}

// An exact-size transposed-QK object, as shipped (32-key, single wave).
KernelSpec transposedKernel(int64_t headSize, std::string maskMode)
{
    KernelSpec spec;
    spec.headSize = headSize;
    spec.maskMode = std::move(maskMode);
    spec.transposedQk = 1;
    spec.queryTail = 0;
    spec.kvTail = 0;
    return spec;
}

KernelDefinition makeKernel(const KernelSpec& spec)
{
    KernelDefinition kernel;
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
                       {"num_waves", spec.numWaves},
                       {"runtime_head_dims", spec.runtimeHeadDims},
                       {"use_attn_bias", spec.useAttnBias},
                       {"bias_dtype", spec.biasDtype}};
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

double scoreOf(const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    auto builder = buildSdpaGraph(GraphSpec{});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    return scorer(MatchContext{graph, 0, properties}, BoundTokens{}, makeKernel(kernelSpec));
}

// GraphSpec{} with no mask (left/right bounds unbounded).
GraphSpec unmaskedGraph()
{
    GraphSpec spec;
    spec.leftBound = -1;
    spec.rightBound = -1;
    return spec;
}

// GraphSpec{} with a causal mask expressed through bounds.
GraphSpec causalGraph()
{
    GraphSpec spec;
    spec.leftBound = -1;
    spec.rightBound = 0;
    return spec;
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
        // A causal request is served by a causal object or, through runtime
        // bounds, by the band ("window") object; never by a no-mask object.
        auto causal = KernelSpec{};
        causal.maskMode = "causal";
        auto none = KernelSpec{};
        none.maskMode = "none";
        EXPECT_TRUE(matchesKernel(spec, causal));
        EXPECT_TRUE(matchesKernel(spec, KernelSpec{}));
        EXPECT_FALSE(matchesKernel(spec, none));
    }
}

// hipDNN scale semantics: a value, a pass-by-value tensor (constant or read at
// launch) or neither (1.0). Device tensors and both forms at once are declined.
TEST(TestGfx1151WmmaAttentionMatchers, ResolvesEveryHipdnnScaleForm)
{
    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    const auto bitsOf = [](float value) {
        int32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<int64_t>(bits);
    };
    const auto bound = [](ScaleForm form) {
        GraphSpec spec;
        spec.scale = form;
        return matchGraph(spec);
    };

    const auto value = bound(ScaleForm::VALUE);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(tryGetBoundInt(*value, "gfx1151_wmma_attention.scale_bits"), bitsOf(0.125F));

    const auto absent = bound(ScaleForm::ABSENT);
    ASSERT_TRUE(absent.has_value());
    EXPECT_EQ(tryGetBoundInt(*absent, "gfx1151_wmma_attention.scale_bits"), bitsOf(1.0F));

    const auto constant = bound(ScaleForm::CONSTANT_TENSOR);
    ASSERT_TRUE(constant.has_value());
    EXPECT_EQ(tryGetBoundInt(*constant, "gfx1151_wmma_attention.scale_bits"), bitsOf(0.25F));
    EXPECT_FALSE(tryGetBoundInt(*constant, "gfx1151_wmma_attention.scale.uid").has_value());

    const auto runtime = bound(ScaleForm::RUNTIME_TENSOR);
    ASSERT_TRUE(runtime.has_value());
    EXPECT_EQ(tryGetBoundInt(*runtime, "gfx1151_wmma_attention.scale.uid"), SCALE_UID);

    EXPECT_FALSE(bound(ScaleForm::DEVICE_TENSOR).has_value());
    EXPECT_FALSE(bound(ScaleForm::VALUE_AND_TENSOR).has_value());
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

    // The strongest standard object: the smallest bucket with its mask compiled in.
    const auto standard = runtimeKernel(64, "none");
    const auto fast = transposedKernel(64, "none");
    EXPECT_GT(scorer(context, BoundTokens{}, makeKernel(fast)),
              scorer(context, BoundTokens{}, makeKernel(standard)));
}

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersNarrowTransposedGeometryForLongSequences)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    GraphSpec longSequence;
    longSequence.queryLength = 1024;
    longSequence.kvLength = 1024;
    auto builder = buildSdpaGraph(longSequence);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    KernelSpec narrow;
    narrow.transposedQk = 1;
    KernelSpec wide = narrow;
    wide.blockN = 64;
    wide.numWaves = 2;
    EXPECT_GT(scorer(context, BoundTokens{}, makeKernel(narrow)),
              scorer(context, BoundTokens{}, makeKernel(wide)));
}

// -----------------------------------------------------------------------------
// Runtime head widths and V heads
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsEveryServedHeadWidth)
{
    for(const int64_t headDim : {16, 80, 112, 192, 256})
    {
        GraphSpec spec;
        spec.headSize = headDim;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value()) << "d" << headDim;
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.head_dim_q"), headDim);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.head_dim_v"), headDim);
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsDistinctValueHeadWidth)
{
    GraphSpec spec;
    spec.headSize = 64;
    spec.vHeadSize = 128;
    const auto bound = matchGraph(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.head_dim_q"), 64);
    EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.head_dim_v"), 128);
}

TEST(TestGfx1151WmmaAttentionMatchers, DeclinesUnservedHeadWidths)
{
    for(const int64_t headDim : {8, 72, 272})
    {
        GraphSpec qk;
        qk.headSize = headDim;
        EXPECT_FALSE(matchGraph(qk).has_value()) << "Q/K d" << headDim;
        GraphSpec vo;
        vo.vHeadSize = headDim;
        EXPECT_FALSE(matchGraph(vo).has_value()) << "V/O d" << headDim;
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsValueHeadsDividingQueryHeads)
{
    // Eight query heads, two K heads.
    for(const int64_t vHeads : {1, 4, 8})
    {
        GraphSpec spec;
        spec.vHeads = vHeads;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value()) << "Hv " << vHeads;
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.v_heads"), vHeads);
    }
    for(const int64_t vHeads : {3, 16})
    {
        GraphSpec spec;
        spec.vHeads = vHeads;
        EXPECT_FALSE(matchGraph(spec).has_value()) << "Hv " << vHeads;
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, RuntimeObjectServesWidthsUpToItsBucket)
{
    GraphSpec d80;
    d80.headSize = 80;
    EXPECT_FALSE(matchesKernel(d80, runtimeKernel(64)));
    EXPECT_TRUE(matchesKernel(d80, runtimeKernel(128)));
    EXPECT_TRUE(matchesKernel(d80, runtimeKernel(256)));

    GraphSpec d256;
    d256.headSize = 256;
    EXPECT_FALSE(matchesKernel(d256, runtimeKernel(128)));
    EXPECT_TRUE(matchesKernel(d256, runtimeKernel(256)));

    // The wider of the two widths picks the bucket.
    GraphSpec mixed;
    mixed.headSize = 64;
    mixed.vHeadSize = 128;
    EXPECT_FALSE(matchesKernel(mixed, runtimeKernel(64)));
    EXPECT_TRUE(matchesKernel(mixed, runtimeKernel(128)));
}

TEST(TestGfx1151WmmaAttentionMatchers, ExactObjectNeedsItsOwnWidthAndSharedKvHeads)
{
    const KernelSpec exact; // d96, runtime_head_dims 0
    EXPECT_TRUE(matchesKernel(GraphSpec{}, exact));

    GraphSpec mixed;
    mixed.vHeadSize = 64;
    EXPECT_FALSE(matchesKernel(mixed, exact));

    GraphSpec ownValueHeads;
    ownValueHeads.vHeads = 1;
    EXPECT_FALSE(matchesKernel(ownValueHeads, exact));
    EXPECT_TRUE(matchesKernel(ownValueHeads, runtimeKernel(128)));

    GraphSpec halfValueHeads;
    halfValueHeads.vHeads = 4;
    EXPECT_TRUE(matchesKernel(halfValueHeads, runtimeKernel(128)));
}

TEST(TestGfx1151WmmaAttentionMatchers, MaskModeServesItsRequests)
{
    const auto none = runtimeKernel(128, "none");
    const auto band = runtimeKernel(128, "window");
    const auto causal = runtimeKernel(128, "causal");

    EXPECT_TRUE(matchesKernel(unmaskedGraph(), none));
    EXPECT_TRUE(matchesKernel(unmaskedGraph(), band));
    EXPECT_FALSE(matchesKernel(unmaskedGraph(), causal));

    EXPECT_FALSE(matchesKernel(causalGraph(), none));
    EXPECT_TRUE(matchesKernel(causalGraph(), band));
    EXPECT_TRUE(matchesKernel(causalGraph(), causal));

    EXPECT_FALSE(matchesKernel(GraphSpec{}, none)); // window 7/3
    EXPECT_TRUE(matchesKernel(GraphSpec{}, band));
    EXPECT_FALSE(matchesKernel(GraphSpec{}, causal));
}

// -----------------------------------------------------------------------------
// Additive bias (attn_mask)
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsBiasWithBroadcastStrides)
{
    // B=2, Hq=8, Sq=33, Sk=65.
    struct Case
    {
        BiasSpec bias;
        int64_t kind;
        std::array<int64_t, 3> strides;
    };
    const std::vector<Case> cases{
        {packedBias({2, 8, 33, 65}), 1, {int64_t{8} * 33 * 65, int64_t{33} * 65, 65}},
        {packedBias({1, 1, 33, 65}), 1, {0, 0, 65}},
        {packedBias({2, 1, 1, 65}, data_objects::DataType::HALF), 2, {65, 0, 0}},
    };
    for(const auto& testCase : cases)
    {
        GraphSpec spec;
        spec.bias = testCase.bias;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bias"), testCase.kind);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bias.uid"), BIAS_UID);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bias.stride_batch"),
                  testCase.strides[0]);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bias.stride_head"),
                  testCase.strides[1]);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bias.stride_query"),
                  testCase.strides[2]);
    }

    const auto unbiased = matchGraph(GraphSpec{});
    ASSERT_TRUE(unbiased.has_value());
    EXPECT_EQ(tryGetBoundInt(*unbiased, "gfx1151_wmma_attention.bias"), 0);
    EXPECT_FALSE(tryGetBoundInt(*unbiased, "gfx1151_wmma_attention.bias.uid").has_value());
}

TEST(TestGfx1151WmmaAttentionMatchers, DeclinesUnservedBiasForms)
{
    const auto declines = [](BiasSpec bias, const char* what) {
        GraphSpec spec;
        spec.bias = std::move(bias);
        EXPECT_FALSE(matchGraph(spec).has_value()) << what;
    };
    declines(packedBias({8, 33, 65}), "rank 3");
    declines(BiasSpec{{2, 8, 33, 65},
                      {int64_t{2} * 8 * 33 * 65, int64_t{2} * 33 * 65, int64_t{2} * 65, 2},
                      data_objects::DataType::FLOAT,
                      false},
             "key stride 2");
    declines(packedBias({2, 8, 33, 1}), "broadcast key axis");
    declines(packedBias({2, 2, 33, 65}), "K head count");
    declines(packedBias({2, 8, 33, 65}, data_objects::DataType::BFLOAT16), "BF16 under FP16 Q");
    declines(packedBias({2, 8, 33, 65}, data_objects::DataType::INT32), "INT32");
    auto passByValue = packedBias({1, 1, 1, 65});
    passByValue.passByValue = true;
    declines(passByValue, "pass-by-value");
}

TEST(TestGfx1151WmmaAttentionMatchers, BiasObjectMustMatchTheBiasClass)
{
    GraphSpec f32Bias;
    f32Bias.bias = packedBias({2, 8, 33, 65});
    EXPECT_TRUE(matchesKernel(f32Bias, runtimeKernel(128, "window", "f32")));
    EXPECT_FALSE(matchesKernel(f32Bias, runtimeKernel(128, "window", "q")));
    EXPECT_FALSE(matchesKernel(f32Bias, runtimeKernel(128, "window")));
    EXPECT_FALSE(matchesKernel(f32Bias, runtimeKernel(128, "none")));

    GraphSpec qBias;
    qBias.bias = packedBias({2, 8, 33, 65}, data_objects::DataType::HALF);
    EXPECT_TRUE(matchesKernel(qBias, runtimeKernel(128, "window", "q")));
    EXPECT_FALSE(matchesKernel(qBias, runtimeKernel(128, "window", "f32")));

    EXPECT_FALSE(matchesKernel(GraphSpec{}, runtimeKernel(128, "window", "f32")));

    // Transposed objects never take a bias.
    GraphSpec aligned = causalGraph();
    aligned.headSize = 64;
    aligned.queryLength = 32;
    aligned.kvLength = 64;
    aligned.bias = packedBias({2, 8, 32, 64});
    auto transposedBias = transposedKernel(64, "causal");
    transposedBias.useAttnBias = 1;
    EXPECT_FALSE(matchesKernel(aligned, transposedBias));
    EXPECT_TRUE(matchesKernel(aligned, runtimeKernel(64, "window", "f32")));
}

// -----------------------------------------------------------------------------
// Score
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersTheTightestBucket)
{
    EXPECT_GT(scoreOf(runtimeKernel(64)), scoreOf(runtimeKernel(128)));
    EXPECT_GT(scoreOf(runtimeKernel(128)), scoreOf(runtimeKernel(256)));
    // The bucket outranks the mask specialization.
    EXPECT_GT(scoreOf(runtimeKernel(128, "window")), scoreOf(runtimeKernel(256, "none")));
}

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersTheMaskSpecializedObject)
{
    EXPECT_GT(scoreOf(runtimeKernel(128, "none")), scoreOf(runtimeKernel(128, "window")));
    EXPECT_GT(scoreOf(runtimeKernel(128, "causal")), scoreOf(runtimeKernel(128, "window")));
}

// An exact-size standard object (runtime_head_dims 0), as shipped for 192/256.
KernelSpec exactKernel(int64_t headSize, std::string maskMode)
{
    KernelSpec spec;
    spec.headSize = headSize;
    spec.maskMode = std::move(maskMode);
    return spec;
}

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersAnExactObjectAtTheSameWidth)
{
    // Over the runtime object of the same head_size, whatever the masks.
    EXPECT_GT(scoreOf(exactKernel(256, "window")), scoreOf(runtimeKernel(256, "none")));
    EXPECT_GT(scoreOf(exactKernel(256, "none")), scoreOf(runtimeKernel(256, "none")));
    // Over a larger bucket.
    EXPECT_GT(scoreOf(exactKernel(192, "window")), scoreOf(runtimeKernel(256, "none")));
    // The mask specialization still separates two exact objects.
    EXPECT_GT(scoreOf(exactKernel(256, "causal")), scoreOf(exactKernel(256, "window")));
    // And the transposed path still outranks everything standard.
    EXPECT_GT(scoreOf(transposedKernel(128, "none")), scoreOf(exactKernel(256, "none")));
    EXPECT_GT(scoreOf(transposedKernel(128, "none")), scoreOf(runtimeKernel(64, "none")));
}

using CatalogEntry = std::pair<std::string, KernelSpec>;

// The shipped standard FP16 objects, after configs/gfx1151_wmma_attention.yaml,
// under their names without the dtype and arch affixes.
std::vector<CatalogEntry> shippedStandardCatalog()
{
    std::vector<CatalogEntry> catalog;
    for(const int64_t bucket : {64, 96, 128, 160, 256})
    {
        const std::string prefix = "dmax" + std::to_string(bucket) + "_";
        catalog.emplace_back(prefix + "none", runtimeKernel(bucket, "none"));
        if(bucket > 128)
        {
            catalog.emplace_back(prefix + "causal", runtimeKernel(bucket, "causal"));
        }
        catalog.emplace_back(prefix + "window", runtimeKernel(bucket, "window"));
        catalog.emplace_back(prefix + "window_bias_f32", runtimeKernel(bucket, "window", "f32"));
        catalog.emplace_back(prefix + "window_bias_q", runtimeKernel(bucket, "window", "q"));
    }
    for(const int64_t headSize : {192, 256})
    {
        const std::string prefix = "d" + std::to_string(headSize) + "_";
        for(const char* mask : {"none", "causal", "window"})
        {
            catalog.emplace_back(prefix + mask, exactKernel(headSize, mask));
        }
    }
    return catalog;
}

// The shipped standard object the engine selects for @p graphSpec: the highest
// score among the matching ones. A tie names every tied object, joined by '|'.
std::string selectedObject(const GraphSpec& graphSpec)
{
    std::string best;
    double bestScore = 0.0;
    for(const auto& [name, spec] : shippedStandardCatalog())
    {
        if(!matchesKernel(graphSpec, spec))
        {
            continue;
        }
        const double score = scoreOf(spec);
        if(best.empty() || score > bestScore)
        {
            best = name;
            bestScore = score;
        }
        else if(score == bestScore)
        {
            best += "|" + name;
        }
    }
    return best;
}

TEST(TestGfx1151WmmaAttentionMatchers, SelectsTheShippedStandardObject)
{
    GraphSpec d256 = unmaskedGraph();
    d256.headSize = 256;
    EXPECT_EQ(selectedObject(d256), "d256_none");

    GraphSpec d240 = unmaskedGraph();
    d240.headSize = 240;
    EXPECT_EQ(selectedObject(d240), "dmax256_none");

    GraphSpec d192Bias;
    d192Bias.headSize = 192;
    d192Bias.bias = packedBias({2, 8, 33, 65});
    EXPECT_EQ(selectedObject(d192Bias), "dmax256_window_bias_f32");

    GraphSpec d160Causal = causalGraph();
    d160Causal.headSize = 160;
    EXPECT_EQ(selectedObject(d160Causal), "dmax160_causal");

    GraphSpec d96Causal = causalGraph();
    d96Causal.headSize = 96;
    EXPECT_EQ(selectedObject(d96Causal), "dmax96_window");

    GraphSpec d256Causal = causalGraph();
    d256Causal.headSize = 256;
    EXPECT_EQ(selectedObject(d256Causal), "d256_causal");

    GraphSpec d192Window;
    d192Window.headSize = 192;
    EXPECT_EQ(selectedObject(d192Window), "d192_window");

    // An exact object needs V to share the K heads.
    GraphSpec d256OwnValueHeads = unmaskedGraph();
    d256OwnValueHeads.headSize = 256;
    d256OwnValueHeads.vHeads = 1;
    EXPECT_EQ(selectedObject(d256OwnValueHeads), "dmax256_none");

    GraphSpec d80 = unmaskedGraph();
    d80.headSize = 80;
    EXPECT_EQ(selectedObject(d80), "dmax96_none");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

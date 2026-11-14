// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx1151WmmaAttentionTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace gfx1151_wmma_attention;
using hipdnn_plugin_sdk::ingestor::KernelSource;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t RAGGED_OFFSET_UID = 6;
constexpr int64_t SCALE_UID = 7;
constexpr int64_t BIAS_UID = 8;
constexpr int64_t SINK_UID = 9;

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
    bool bottomRight = false;
    ScaleForm scale = ScaleForm::VALUE;
    // The VALUE form's attn_scale_value and the CONSTANT_TENSOR form's baked value, when
    // not their defaults (0.125 and 0.25).
    std::optional<float> scaleValue;
    bool overrideShape = false;
    // Elements of padding after each Q row: a row stride that is not a multiple of 8
    // elements leaves rows off the kernels' 16-byte vector alignment.
    int64_t qRowPad = 0;
    // Attention sinks: sink_token [1, Hq, 1, 1] in the Q data type.
    bool sinks = false;
    // Declared device-pointer alignment in bytes, by tensor uid (16 when absent).
    std::map<int64_t, int64_t> alignment;
    // Further SDPA attributes (sequence lengths, FP8, dropout and other features).
    AttributeEdit extraAttributes;
};

// The default device is gfx12: the dispatcher's rule keeps a small-query graph with a
// short key window on prefill there, where on gfx11 it sends it to decode, and most
// tests below match a prefill object against such a graph. Tests of the rule itself, and
// of what only one family does, name their device.
struct Device
{
    std::string arch = "gfx1201";
    int multiprocessors = 32; // RDNA WGPs, as HIP reports them
};

DeviceProperties testDeviceProperties(const Device& device = {})
{
    return deviceProperties(device.arch, device.multiprocessors);
}

/// Dense strides with @p rowPad elements after every row of @p headSize.
std::vector<int64_t>
    denseStrides(bool bhsd, int64_t heads, int64_t sequence, int64_t headSize, int64_t rowPad = 0)
{
    if(bhsd)
    {
        const int64_t row = headSize + rowPad;
        return {heads * sequence * row, sequence * row, row, 1};
    }
    const int64_t token = heads * headSize + rowPad;
    return {sequence * token, headSize, token, 1};
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

    const auto qStrides
        = denseStrides(spec.bhsd, spec.queryHeads, spec.queryLength, spec.headSize, spec.qRowPad);
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
        const int64_t alignment = declaredAlignment(spec.alignment, uid);
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
                                                          ragged,
                                                          alignment);
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
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            BIAS_UID,
            nullptr,
            spec.bias->dtype,
            &spec.bias->strides,
            &spec.bias->dims,
            false,
            data_objects::TensorValue::NONE,
            0,
            spec.bias->passByValue,
            ::flatbuffers::nullopt,
            declaredAlignment(spec.alignment, BIAS_UID)));
    }
    if(spec.sinks)
    {
        const std::vector<int64_t> sinkDims{1, spec.queryHeads, 1, 1};
        tensors.push_back(tensorFor(SINK_UID, spec.dtype, packedStrides(sinkDims), sinkDims));
    }
    const std::vector<int64_t> scalarDims{1};
    switch(spec.scale)
    {
    case ScaleForm::CONSTANT_TENSOR:
    case ScaleForm::VALUE_AND_TENSOR:
    {
        const data_objects::Float32Value value(spec.scaleValue.value_or(0.25F));
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
    attributes.add_diagonal_alignment(spec.bottomRight
                                          ? data_objects::DiagonalAlignment::BOTTOM_RIGHT
                                          : data_objects::DiagonalAlignment::TOP_LEFT);
    if(spec.scale == ScaleForm::VALUE || spec.scale == ScaleForm::VALUE_AND_TENSOR)
    {
        attributes.add_attn_scale_value(spec.scaleValue.value_or(0.125F));
    }
    if(spec.scale != ScaleForm::VALUE && spec.scale != ScaleForm::ABSENT)
    {
        attributes.add_scale_tensor_uid(SCALE_UID);
    }
    if(spec.bias.has_value())
    {
        attributes.add_attn_mask_tensor_uid(BIAS_UID);
    }
    if(spec.sinks)
    {
        attributes.add_sink_token_tensor_uid(SINK_UID);
    }
    attributes.add_implementation(data_objects::AttentionImplementation::AUTO);
    attributes.add_mma_core_mode(data_objects::DataType::FLOAT);
    if(spec.extraAttributes)
    {
        spec.extraAttributes(attributes);
    }

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
    graph.add_is_override_shape_enabled(spec.overrideShape);
    builder.Finish(graph.Finish());
    return builder;
}

/// A graph and the device it is matched for.
class Fixture : public EngineGraphFixture
{
public:
    explicit Fixture(const GraphSpec& spec, const Device& device = {})
        : EngineGraphFixture(buildSdpaGraph(spec), testDeviceProperties(device))
    {
    }
};

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    return Fixture(spec).bound();
}

/// A shipped prefill object: the causal class, block_m 128 (64 over a wave pair).
KernelSpec prefillKernel(int64_t headSize,
                         int64_t headSizeV,
                         std::string bias = "none",
                         std::string maskMode = "causal")
{
    KernelSpec spec;
    spec.headSize = headSize;
    spec.headSizeV = headSizeV;
    spec.bias = std::move(bias);
    spec.maskMode = std::move(maskMode);
    spec.blockM = std::max(headSize, headSizeV) > 128 ? 64 : 128;
    return spec;
}

/// A shipped decode object: four waves, runtime masks.
KernelSpec decodeKernel(int64_t bucket, std::string bias = "none", int64_t dvSplit = 1)
{
    KernelSpec spec;
    spec.kind = "decode";
    spec.headSize = bucket;
    spec.headSizeV = bucket;
    spec.maskMode = "runtime";
    spec.bias = std::move(bias);
    spec.blockSize = 128;
    spec.blockM = 16;
    spec.dvSplit = dvSplit;
    return spec;
}

/// Whether the engine owning @p kernelSpec's kind matches it; the other engine's kernel
/// matcher never sees it.
bool matchesKernel(const GraphSpec& graphSpec,
                   const KernelSpec& kernelSpec,
                   const Device& device = {})
{
    const Fixture fixture(graphSpec, device);
    const auto bound = fixture.bound();
    if(!bound.has_value())
    {
        return false;
    }
    return matchesKernel(symbolsFor(kernelSpec), fixture.context(), makeKernel(kernelSpec), *bound);
}

double scoreOf(const KernelSpec& kernelSpec, const GraphSpec& graphSpec = {})
{
    const Fixture fixture(graphSpec);
    const auto bound = fixture.bound();
    if(!bound.has_value())
    {
        throw std::logic_error("the scored graph does not match");
    }
    return scoreKernel(symbolsFor(kernelSpec), fixture.context(), makeKernel(kernelSpec), *bound);
}

size_t workspaceOf(const GraphSpec& graphSpec, const KernelSpec& kernelSpec, const Device& device)
{
    const Fixture fixture(graphSpec, device);
    const auto bound = fixture.bound();
    if(!bound.has_value())
    {
        throw std::logic_error("the dispatched graph does not match");
    }
    const auto kernel = makeKernel(kernelSpec);
    return dispatchHandlerOf(kernel).workspaceBytes(fixture.context(), *bound, kernel);
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

/// One decode step: Sq = 1 against a long KV cache, BSHD, GQA 32/8, D128, no mask.
GraphSpec decodeGraph()
{
    GraphSpec spec = unmaskedGraph();
    spec.batch = 1;
    spec.queryHeads = 32;
    spec.kvHeads = 8;
    spec.queryLength = 1;
    spec.kvLength = 4096;
    spec.headSize = 128;
    return spec;
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

// Only the paged prefill objects ship with attention sinks: a dense-layout graph carrying
// a well-formed sink_token is declined (TestGfx1151WmmaAttentionPaged covers the paged
// layout), and an object claiming sinks outside the paged layout serves nothing.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesAttentionSinksOnTheDenseLayout)
{
    GraphSpec spec;
    spec.sinks = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
    spec.dtype = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx1151WmmaAttentionMatchers, ADenseObjectClaimingSinksServesNothing)
{
    auto prefill = prefillKernel(96, 96);
    ASSERT_TRUE(matchesKernel(GraphSpec{}, prefill));
    prefill.sinks = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, prefill));
}

// Launch arguments are bound from the graph's shapes at prepare, so a graph whose
// shapes may be overridden at execute is declined (the engine declares schema
// 1.2.0 for runtime scales, which would otherwise admit it).
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesOverrideShapeGraphs)
{
    GraphSpec spec;
    spec.scale = ScaleForm::RUNTIME_TENSOR;
    ASSERT_TRUE(matchGraph(spec).has_value());
    spec.overrideShape = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// The graph's left_bound follows cuDNN (L keys ending at the diagonal, diagonal
// included); the kernel's window_left follows FlashAttention (keys before the
// diagonal), so the same window binds as L - 1. left_bound = 0 is outside hipDNN's
// band domain (positive or -1, SEM-005), so it is declined.
TEST(TestGfx1151WmmaAttentionMatchers, BindsCudnnLeftBoundAsFlashAttentionWindowLeft)
{
    GraphSpec spec;
    spec.leftBound = 7;
    spec.rightBound = 3;
    const auto bound = matchGraph(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.mask"), 2); // window
    EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_left"), 6);
    EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_right"), 3);

    spec.leftBound = 1;
    spec.rightBound = 0;
    const auto diagonalOnly = matchGraph(spec);
    ASSERT_TRUE(diagonalOnly.has_value());
    EXPECT_EQ(tryGetBoundInt(*diagonalOnly, "gfx1151_wmma_attention.mask"), 2); // window
    EXPECT_EQ(tryGetBoundInt(*diagonalOnly, "gfx1151_wmma_attention.window_left"), 0);

    spec.leftBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// The deprecated causal flags compose with the bounds the way cuDNN's setters do
// (resolveDiagonalBand, the hipDNN references): right bound 0 unless one is set,
// bottom-right for causal_mask_bottom_right. window_left is the kernel's count of
// keys left of the diagonal, so cuDNN's left_bound = 7 becomes 6 (SEM-001).
TEST(TestGfx1151WmmaAttentionMatchers, DeprecatedCausalFlagsComposeWithBounds)
{
    for(const bool bottomRight : {false, true})
    {
        GraphSpec windowed;
        windowed.causalMask = !bottomRight;
        windowed.causalMaskBottomRight = bottomRight;
        windowed.rightBound = -1;
        const auto window = matchGraph(windowed);
        ASSERT_TRUE(window.has_value());
        EXPECT_EQ(tryGetBoundInt(*window, "gfx1151_wmma_attention.mask"), 2); // window
        EXPECT_EQ(tryGetBoundInt(*window, "gfx1151_wmma_attention.window_left"), 6);
        EXPECT_EQ(tryGetBoundInt(*window, "gfx1151_wmma_attention.window_right"), 0);
        EXPECT_EQ(tryGetBoundInt(*window, "gfx1151_wmma_attention.bottom_right"),
                  bottomRight ? 1 : 0);

        GraphSpec spec;
        spec.causalMask = !bottomRight;
        spec.causalMaskBottomRight = bottomRight;
        spec.leftBound = -1;
        spec.rightBound = -1;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.mask"), 1); // causal
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_left"), -1);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.window_right"), 0);
        EXPECT_EQ(tryGetBoundInt(*bound, "gfx1151_wmma_attention.bottom_right"),
                  bottomRight ? 1 : 0);
        // A causal request is served by the causal class through runtime bounds;
        // never by a no-mask object.
        EXPECT_TRUE(matchesKernel(spec, prefillKernel(96, 96)));
        EXPECT_FALSE(matchesKernel(spec, prefillKernel(96, 96, "none", "none")));
    }

    // An explicit right bound overrides the deprecated flag's right bound 0, so the
    // band is a window with window_right 3, not a causal mask.
    GraphSpec explicitRight;
    explicitRight.causalMaskBottomRight = true;
    explicitRight.leftBound = -1;
    explicitRight.rightBound = 3;
    const auto widened = matchGraph(explicitRight);
    ASSERT_TRUE(widened.has_value());
    EXPECT_EQ(tryGetBoundInt(*widened, "gfx1151_wmma_attention.mask"), 2); // window
    EXPECT_EQ(tryGetBoundInt(*widened, "gfx1151_wmma_attention.window_left"), -1);
    EXPECT_EQ(tryGetBoundInt(*widened, "gfx1151_wmma_attention.window_right"), 3);
    EXPECT_EQ(tryGetBoundInt(*widened, "gfx1151_wmma_attention.bottom_right"), 1);

    // causal_mask sets no alignment of its own, so a stored BOTTOM_RIGHT alignment
    // stays bottom-right.
    GraphSpec aligned;
    aligned.causalMask = true;
    aligned.bottomRight = true;
    aligned.leftBound = -1;
    aligned.rightBound = -1;
    const auto causal = matchGraph(aligned);
    ASSERT_TRUE(causal.has_value());
    EXPECT_EQ(tryGetBoundInt(*causal, "gfx1151_wmma_attention.mask"), 1); // causal
    EXPECT_EQ(tryGetBoundInt(*causal, "gfx1151_wmma_attention.window_right"), 0);
    EXPECT_EQ(tryGetBoundInt(*causal, "gfx1151_wmma_attention.bottom_right"), 1);
}

// hipDNN scale semantics: a value, a pass-by-value tensor (constant or read at
// launch) or neither (1.0). Device tensors and both forms at once are declined.
TEST(TestGfx1151WmmaAttentionMatchers, ResolvesEveryHipdnnScaleForm)
{
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

// The kernels take scale_log2 = log2(e) * scale and require it positive (rdna_attention.py
// precondition); at 0 or below they return all-NaN or all-zero output with a success
// status (measured on gfx1100 and gfx1101), so the matcher declines such constants as
// the Python route does.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesScalesThatAreNotFiniteAndPositive)
{
    for(const ScaleForm form : {ScaleForm::VALUE, ScaleForm::CONSTANT_TENSOR})
    {
        for(const float scale : {0.0F,
                                 -0.0F,
                                 -0.125F,
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()})
        {
            GraphSpec spec;
            spec.scale = form;
            spec.scaleValue = scale;
            EXPECT_FALSE(matchGraph(spec).has_value())
                << "scale " << scale << " form " << static_cast<int>(form);
        }
        GraphSpec tiny;
        tiny.scale = form;
        tiny.scaleValue = std::numeric_limits<float>::denorm_min();
        EXPECT_TRUE(matchGraph(tiny).has_value()) << "form " << static_cast<int>(form);
    }
}

// -----------------------------------------------------------------------------
// Head widths, rows and head counts
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsEveryMultipleOfEightHeadWidth)
{
    for(const int64_t headDim : {8, 16, 40, 72, 80, 112, 136, 192, 248, 256})
    {
        GraphSpec qk;
        qk.headSize = headDim;
        EXPECT_TRUE(matchGraph(qk).has_value()) << "Q/K d" << headDim;
        GraphSpec vo;
        vo.vHeadSize = headDim;
        EXPECT_TRUE(matchGraph(vo).has_value()) << "V/O d" << headDim;
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, DeclinesUnservedHeadWidths)
{
    for(const int64_t headDim : {4, 12, 100, 260, 264})
    {
        GraphSpec qk;
        qk.headSize = headDim;
        EXPECT_FALSE(matchGraph(qk).has_value()) << "Q/K d" << headDim;
        GraphSpec vo;
        vo.vHeadSize = headDim;
        EXPECT_FALSE(matchGraph(vo).has_value()) << "V/O d" << headDim;
    }
}

// Rows load as 16-byte vectors, so every outer stride is a multiple of 8 elements.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesRowsOffTheVectorAlignment)
{
    for(const bool bhsd : {false, true})
    {
        GraphSpec aligned;
        aligned.bhsd = bhsd;
        aligned.qRowPad = 8;
        EXPECT_TRUE(matchGraph(aligned).has_value()) << "bhsd " << bhsd;
        GraphSpec misaligned = aligned;
        misaligned.qRowPad = 4;
        EXPECT_FALSE(matchGraph(misaligned).has_value()) << "bhsd " << bhsd;
    }
}

/// The two engines and the names their decline lines carry. Both run the shared dense
/// matcher, so each must decline a dense graph under its own name.
struct EngineUnderTest
{
    const PackSymbols* symbols;
    std::string_view name;
};
constexpr std::array<EngineUnderTest, 2> BOTH_ENGINES{
    EngineUnderTest{&PREFILL_SYMBOLS, PREFILL_ENGINE},
    EngineUnderTest{&DECODE_SYMBOLS, DECODE_ENGINE}};

/// What @p engine logged at INFO when it declined @p spec.
LoggedDecline loggedDecline(const GraphSpec& spec, const EngineUnderTest& engine)
{
    return recordDecline([&] { return Fixture(spec).bound(*engine.symbols).has_value(); });
}

GraphSpec withAttributes(AttributeEdit extra)
{
    GraphSpec spec;
    spec.extraAttributes = std::move(extra);
    return spec;
}

// Rows load as 16-byte vectors: a graph declaring a lower Q / K / V / O pointer alignment
// (engine API 1.3.0, which the engine's 1.4.0 admits) is declined with its cause rather
// than failing at execute.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesOperandsDeclaringLessThanSixteenBytes)
{
    GraphSpec wider;
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        wider.alignment[uid] = 32;
    }
    EXPECT_TRUE(matchGraph(wider).has_value());
    for(const auto& [role, uid] : std::vector<std::pair<std::string, int64_t>>{
            {"Q", Q_UID}, {"K", K_UID}, {"V", V_UID}, {"O", O_UID}})
    {
        for(const int64_t bytes : {8, 24})
        {
            GraphSpec spec;
            spec.alignment[uid] = bytes;
            for(const auto& engine : BOTH_ENGINES)
            {
                SCOPED_TRACE(role + ", " + std::to_string(bytes) + " bytes, "
                             + std::string(engine.name));
                expectDecline(loggedDecline(spec, engine),
                              engine.name,
                              "alignment",
                              role + " declares a " + std::to_string(bytes) + "-byte alignment");
            }
        }
    }
}

// The stats are stored as FP32 words and the additive bias loaded in its own element
// type (FP32 or the Q type): a graph declaring less is declined with its cause.
TEST(TestGfx1151WmmaAttentionMatchers, DeclinesStatsAndBiasDeclaringLessThanTheirElements)
{
    struct Case
    {
        std::string what;
        GraphSpec spec;
        int64_t uid;
        int64_t served; // the least alignment the kernels need
        std::string role; // as logged
    };
    GraphSpec f32Bias;
    f32Bias.bias = packedBias({2, 8, 33, 65});
    GraphSpec halfBias;
    halfBias.bias = packedBias({2, 1, 1, 65}, data_objects::DataType::HALF);
    const std::vector<Case> cases{{"stats", GraphSpec{}, LSE_UID, 4, "Stats"},
                                  {"an FP32 bias", f32Bias, BIAS_UID, 4, "attn_mask"},
                                  {"a HALF bias", halfBias, BIAS_UID, 2, "attn_mask"}};
    for(const auto& c : cases)
    {
        auto served = c.spec;
        served.alignment[c.uid] = c.served;
        EXPECT_TRUE(matchGraph(served).has_value()) << c.what;
        for(int64_t bytes = 1; bytes < c.served; ++bytes)
        {
            auto spec = c.spec;
            spec.alignment[c.uid] = bytes;
            for(const auto& engine : BOTH_ENGINES)
            {
                SCOPED_TRACE(c.what + ", " + std::to_string(bytes) + " bytes, "
                             + std::string(engine.name));
                expectDecline(loggedDecline(spec, engine),
                              engine.name,
                              "alignment",
                              c.role + " declares a " + std::to_string(bytes) + "-byte alignment");
            }
        }
    }
}

// Sequence lengths, max_seq_len_kv and the padding mask belong to the paged graph, sinks
// to the paged objects, and FP8, dropout and the other features below to no layout of
// this engine: each dense graph carrying one is declined with its cause and reason, by
// either engine under its own name.
TEST(TestGfx1151WmmaAttentionMatchers, LogsTheCauseOfEachDenseDecline)
{
    using Builder = data_objects::SdpaAttributesBuilder;
    struct Case
    {
        const char* what;
        GraphSpec spec;
        const char* cause;
        const char* reason; // a part of the logged reason
    };
    GraphSpec sinks;
    sinks.sinks = true;
    const char* paddedBatch = "need a paged graph";
    const std::vector<Case> cases{
        {"seq_len_q",
         withAttributes(settingUid(&Builder::add_seq_len_q_tensor_uid)),
         "padding",
         paddedBatch},
        {"seq_len_kv",
         withAttributes(settingUid(&Builder::add_seq_len_kv_tensor_uid)),
         "padding",
         paddedBatch},
        {"max_seq_len_kv",
         withAttributes([](Builder& a) { a.add_max_seq_len_kv(64); }),
         "padding",
         paddedBatch},
        {"padding_mask",
         withAttributes([](Builder& a) { a.add_padding_mask(true); }),
         "padding",
         paddedBatch},
        {"sinks", sinks, "sinks", "only the paged prefill objects take attention sinks"},
        {"descale_k",
         withAttributes(settingUid(&Builder::add_descale_k_tensor_uid)),
         "fp8",
         "FP8 descale / amax operands"},
        {"amax_o",
         withAttributes(settingUid(&Builder::add_amax_o_tensor_uid)),
         "fp8",
         "FP8 descale / amax operands"},
        {"seed",
         withAttributes(settingUid(&Builder::add_seed_tensor_uid)),
         "dropout",
         "no dropout"},
        {"dropout probability",
         withAttributes([](Builder& a) { a.add_dropout_probability(0.1F); }),
         "dropout",
         "no dropout"},
        {"block_mask",
         withAttributes(settingUid(&Builder::add_block_mask_tensor_uid)),
         "feature",
         "a block mask"},
        {"sum_exp",
         withAttributes(settingUid(&Builder::add_sum_exp_tensor_uid)),
         "feature",
         "max / sum_exp outputs"},
        {"a forced COMPOSITE implementation",
         withAttributes([](Builder& a) {
             a.add_implementation(data_objects::AttentionImplementation::COMPOSITE);
         }),
         "feature",
         "a forced implementation"},
        {"a forced UNIFIED implementation",
         withAttributes([](Builder& a) {
             a.add_implementation(data_objects::AttentionImplementation::UNIFIED);
         }),
         "feature",
         "a forced implementation"},
        {"alibi",
         withAttributes([](Builder& a) { a.add_alibi_mask(true); }),
         "alibi",
         "no per-head slopes tensor"},
    };
    for(const auto& c : cases)
    {
        for(const auto& engine : BOTH_ENGINES)
        {
            SCOPED_TRACE(std::string(c.what) + ", " + std::string(engine.name));
            expectDecline(loggedDecline(c.spec, engine), engine.name, c.cause, c.reason);
        }
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, AcceptsValueHeadsDividingQueryHeads)
{
    // Eight query heads, two K heads.
    for(const int64_t vHeads : {1, 2, 4, 8})
    {
        GraphSpec spec;
        spec.vHeads = vHeads;
        EXPECT_TRUE(matchGraph(spec).has_value()) << "Hv " << vHeads;
        EXPECT_TRUE(matchesKernel(spec, prefillKernel(96, 96))) << "Hv " << vHeads;
    }
    for(const int64_t vHeads : {3, 16})
    {
        GraphSpec spec;
        spec.vHeads = vHeads;
        EXPECT_FALSE(matchGraph(spec).has_value()) << "Hv " << vHeads;
    }
}

// -----------------------------------------------------------------------------
// Prefill objects
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, PrefillObjectServesWidthsUpToItsBuckets)
{
    GraphSpec d80;
    d80.headSize = 80;
    EXPECT_FALSE(matchesKernel(d80, prefillKernel(64, 64)));
    EXPECT_TRUE(matchesKernel(d80, prefillKernel(96, 96)));
    EXPECT_TRUE(matchesKernel(d80, prefillKernel(256, 256)));

    // Q/K and V/O against their own buckets.
    GraphSpec q192v128;
    q192v128.headSize = 192;
    q192v128.vHeadSize = 128;
    EXPECT_TRUE(matchesKernel(q192v128, prefillKernel(192, 128)));
    EXPECT_FALSE(matchesKernel(q192v128, prefillKernel(128, 128)));
    q192v128.vHeadSize = 136;
    EXPECT_FALSE(matchesKernel(q192v128, prefillKernel(192, 128)));
    EXPECT_TRUE(matchesKernel(q192v128, prefillKernel(256, 256)));
}

TEST(TestGfx1151WmmaAttentionMatchers, CausalClassServesEveryMask)
{
    const auto causal = prefillKernel(128, 128);
    const auto none = prefillKernel(128, 128, "none", "none");
    EXPECT_TRUE(matchesKernel(unmaskedGraph(), causal));
    EXPECT_TRUE(matchesKernel(causalGraph(), causal));
    EXPECT_TRUE(matchesKernel(GraphSpec{}, causal)); // window 7/3

    EXPECT_TRUE(matchesKernel(unmaskedGraph(), none));
    EXPECT_FALSE(matchesKernel(causalGraph(), none));
    EXPECT_FALSE(matchesKernel(GraphSpec{}, none));
}

TEST(TestGfx1151WmmaAttentionMatchers, KernelMatcherReadsTheDtype)
{
    auto bf16 = prefillKernel(128, 128);
    bf16.dtype = "BF16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, bf16));
    GraphSpec bf16Graph;
    bf16Graph.dtype = data_objects::DataType::BFLOAT16;
    EXPECT_TRUE(matchesKernel(bf16Graph, bf16));
    EXPECT_FALSE(matchesKernel(bf16Graph, prefillKernel(128, 128)));
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
    EXPECT_TRUE(matchesKernel(f32Bias, prefillKernel(128, 128, "f32")));
    EXPECT_FALSE(matchesKernel(f32Bias, prefillKernel(128, 128, "q")));
    EXPECT_FALSE(matchesKernel(f32Bias, prefillKernel(128, 128)));

    GraphSpec qBias;
    qBias.bias = packedBias({2, 8, 33, 65}, data_objects::DataType::HALF);
    EXPECT_TRUE(matchesKernel(qBias, prefillKernel(128, 128, "q")));
    EXPECT_FALSE(matchesKernel(qBias, prefillKernel(128, 128, "f32")));

    EXPECT_FALSE(matchesKernel(GraphSpec{}, prefillKernel(128, 128, "f32")));

    GraphSpec decodeBias = decodeGraph();
    decodeBias.bias = packedBias({1, 32, 1, 4096});
    EXPECT_TRUE(matchesKernel(decodeBias, decodeKernel(128, "f32")));
    EXPECT_FALSE(matchesKernel(decodeBias, decodeKernel(128)));
    EXPECT_FALSE(matchesKernel(decodeGraph(), decodeKernel(128, "f32")));
}

// -----------------------------------------------------------------------------
// Decode objects: the rocKE dispatcher's rule and launch plan
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, DecodeObjectServesTheDispatcherDecodeRule)
{
    // gfx1151.py rdna_attention_use_decode, g = Hq / Hkv, num_cus = 2 x multiProcessorCount:
    //   Sq > 64: 4 * B * Hq * ceil(Sq / 128) < num_cus
    //   gfx12:   Sq <= 32 or g == 1 or visible_keys >= 2048
    //   gfx11:   g * Sq < 512 or visible_keys < 64 * num_cus or B * visible_keys < 512 * num_cus
    // Vectors: test_gfx1151_attention_selection.py::test_decode_rule_vectors.
    EXPECT_TRUE(matchesKernel(decodeGraph(), decodeKernel(128)));
    struct Case
    {
        int64_t queryLength;
        int64_t kvLength; // unmasked: every key is visible
        int64_t queryHeads;
        int64_t kvHeads;
        int64_t batch;
        Device device;
        bool decode;
    };
    const Device gfx11{"gfx1101", 30}; // 60 CUs
    const Device gfx12{"gfx1201", 32}; // 64 CUs
    const Device gfx1100{"gfx1100", 48}; // 96 CUs
    for(const Case& c :
        {Case{1, 4096, 32, 8, 1, gfx11, true},     Case{32, 32768, 64, 8, 8, gfx11, true},
         Case{64, 4096, 64, 8, 1, gfx11, true},    Case{64, 4096, 64, 8, 8, gfx11, false},
         Case{64, 2048, 64, 8, 32, gfx11, true},   Case{64, 32768, 64, 8, 1, gfx11, false},
         Case{64, 32768, 32, 8, 8, gfx11, true},   Case{32, 512, 64, 8, 32, gfx12, true},
         Case{64, 512, 8, 8, 32, gfx12, true},     Case{64, 512, 32, 8, 8, gfx12, false},
         Case{64, 1024, 64, 8, 1, gfx12, false},   Case{64, 2048, 64, 8, 8, gfx12, true},
         Case{96, 4096, 8, 8, 1, gfx12, true},     Case{128, 4096, 8, 8, 8, gfx12, false},
         Case{256, 4096, 8, 8, 1, gfx12, false},   Case{128, 4096, 8, 8, 1, gfx11, true},
         Case{96, 4096, 32, 8, 1, gfx11, false},   Case{64, 4096, 64, 8, 8, gfx1100, true},
         Case{64, 32768, 64, 8, 1, gfx1100, true}, Case{64, 32768, 64, 8, 8, gfx1100, false}})
    {
        GraphSpec spec = decodeGraph();
        spec.queryLength = c.queryLength;
        spec.kvLength = c.kvLength;
        spec.queryHeads = c.queryHeads;
        spec.kvHeads = c.kvHeads;
        spec.batch = c.batch;
        EXPECT_EQ(matchesKernel(spec, decodeKernel(128), c.device), c.decode)
            << c.device.arch << " Sq " << c.queryLength << " Sk " << c.kvLength << " Hq "
            << c.queryHeads << " Hkv " << c.kvHeads << " B " << c.batch;
        // The engines split the rule's problem space exactly: prefill serves the rest.
        EXPECT_EQ(matchesKernel(spec, prefillKernel(128, 128), c.device), !c.decode)
            << c.device.arch << " Sq " << c.queryLength << " Sk " << c.kvLength << " Hq "
            << c.queryHeads << " Hkv " << c.kvHeads << " B " << c.batch;
    }
    // visible_keys is the window's span: a bottom-right 1024-key window over 4096 keys
    // sees 1087 keys, under gfx12's 2048-key threshold.
    GraphSpec window = decodeGraph();
    window.queryLength = 64;
    window.queryHeads = 64;
    window.kvHeads = 8;
    EXPECT_TRUE(matchesKernel(window, decodeKernel(128), gfx12));
    window.bottomRight = true;
    window.leftBound = 1024; // 1024 keys including the diagonal: window_left 1023
    window.rightBound = 0;
    EXPECT_FALSE(matchesKernel(window, decodeKernel(128), gfx12));
}

TEST(TestGfx1151WmmaAttentionMatchers, DecodeObjectIsTheDispatcherBucket)
{
    // wmma_fmha_decode_bucket: the smallest of 64/128/256 holding both widths whose
    // narrowest served Q/K width (16/72/136) is at most head_dim_q.
    struct Case
    {
        int64_t headDimQ;
        int64_t headDimV;
        int64_t bucket; // 0: no decode object serves it
    };
    for(const Case& c : {Case{64, 64, 64},
                         Case{16, 64, 64},
                         Case{8, 8, 0},
                         Case{96, 96, 128},
                         Case{72, 128, 128},
                         Case{64, 128, 0},
                         Case{192, 192, 256},
                         Case{136, 64, 256},
                         Case{128, 256, 0}})
    {
        GraphSpec spec = decodeGraph();
        spec.headSize = c.headDimQ;
        spec.vHeadSize = c.headDimV;
        for(const int64_t bucket : {64, 128, 256})
        {
            EXPECT_EQ(matchesKernel(spec, decodeKernel(bucket)), bucket == c.bucket)
                << "Q " << c.headDimQ << " V " << c.headDimV << " bucket " << bucket;
        }
    }
}

TEST(TestGfx1151WmmaAttentionMatchers, DecodeNeedsPackableHeadCounts)
{
    // P = max(Hkv, Hv) pack groups: one of the two head counts divides the other.
    GraphSpec packs = decodeGraph();
    packs.vHeads = 16;
    EXPECT_TRUE(matchesKernel(packs, decodeKernel(128)));
    GraphSpec mixed = decodeGraph();
    mixed.queryHeads = 24;
    mixed.kvHeads = 4;
    mixed.vHeads = 6;
    EXPECT_FALSE(matchesKernel(mixed, decodeKernel(128)));
    EXPECT_TRUE(matchesKernel(mixed, prefillKernel(128, 128)));
}

TEST(TestGfx1151WmmaAttentionMatchers, DecodeWorkspaceMirrorsTheDispatcherPlan)
{
    // decode_launch_plan(...).workspace for these problems (rocKE host helpers, num_cus =
    // 2 x multiProcessorCount) without the WS_CNT counter region: every multi-split
    // launch merges in the merge kernel, so the workspace holds only the split partials,
    // B * Hq * Sq * splits * (Dv + 1) FP32 values.
    const Device gfx1201{"gfx1201", 32};
    // 8 splits.
    EXPECT_EQ(workspaceOf(decodeGraph(), decodeKernel(128), gfx1201), 132096U);
    // A bottom-right 1024-key window: 2 splits (still merged by the merge kernel).
    GraphSpec window = decodeGraph();
    window.bottomRight = true;
    window.leftBound = 1024; // 1024 keys including the diagonal: window_left 1023
    window.rightBound = 0;
    EXPECT_EQ(workspaceOf(window, decodeKernel(128), gfx1201), 33024U);
    // Batch 4: 2 splits.
    GraphSpec batch4 = decodeGraph();
    batch4.batch = 4;
    EXPECT_EQ(workspaceOf(batch4, decodeKernel(128), gfx1201), 132096U);
    // Batch 8 fills the device with one split: no workspace.
    GraphSpec batch8 = decodeGraph();
    batch8.batch = 8;
    EXPECT_EQ(workspaceOf(batch8, decodeKernel(128), gfx1201), 0U);
    // gfx11 D256 (dv_split 2), Sq 4, 48 WGPs: 12 splits.
    GraphSpec d256 = decodeGraph();
    d256.queryHeads = 16;
    d256.kvHeads = 2;
    d256.queryLength = 4;
    d256.kvLength = 8192;
    d256.headSize = 256;
    EXPECT_EQ(workspaceOf(d256, decodeKernel(256, "none", 2), Device{"gfx1100", 48}), 789504U);
    // Prefill takes no workspace.
    EXPECT_EQ(workspaceOf(GraphSpec{}, prefillKernel(96, 96), gfx1201), 0U);
}

// -----------------------------------------------------------------------------
// Score and selection
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersTheTightestBucketPair)
{
    EXPECT_GT(scoreOf(prefillKernel(64, 64)), scoreOf(prefillKernel(96, 96)));
    EXPECT_GT(scoreOf(prefillKernel(128, 128)), scoreOf(prefillKernel(192, 128)));
    EXPECT_GT(scoreOf(prefillKernel(192, 128)), scoreOf(prefillKernel(256, 128)));
    EXPECT_GT(scoreOf(prefillKernel(256, 128)), scoreOf(prefillKernel(256, 256)));
}

TEST(TestGfx1151WmmaAttentionMatchers, ScorePrefersANoMaskObjectOnlyForUnmaskedGraphs)
{
    const auto none = prefillKernel(128, 128, "none", "none");
    const auto causal = prefillKernel(128, 128);
    EXPECT_GT(scoreOf(none, unmaskedGraph()), scoreOf(causal, unmaskedGraph()));
    // The bucket still outranks the mask specialization.
    EXPECT_GT(scoreOf(prefillKernel(96, 96), unmaskedGraph()), scoreOf(none, unmaskedGraph()));
}

TEST(TestGfx1151WmmaAttentionMatchers, ScoreRanksAnEligibleDecodeObjectFirst)
{
    EXPECT_GT(scoreOf(decodeKernel(256), decodeGraph()),
              scoreOf(prefillKernel(64, 64), decodeGraph()));
}

using CatalogEntry = std::pair<std::string, KernelSpec>;

/// The shipped FP16 no-bias objects of one family (configs/gfx1151_wmma_attention.yaml),
/// under their descriptor names without the dtype and bias affixes.
std::vector<CatalogEntry> shippedCatalog(bool gfx12)
{
    std::vector<CatalogEntry> catalog;
    for(const auto& [q, v] : std::vector<std::pair<int64_t, int64_t>>{
            {64, 64}, {96, 96}, {128, 128}, {192, 128}, {256, 128}, {256, 256}})
    {
        auto spec = prefillKernel(q, v);
        if(gfx12 && v <= 128)
        {
            spec.blockM = 128; // gfx12 keeps an unsplit Q up to 256 while V fits 128
        }
        catalog.emplace_back("prefill_d" + std::to_string(q) + "v" + std::to_string(v), spec);
    }
    for(const int64_t bucket : {64, 128, 256})
    {
        int64_t dvSplit = 1;
        if(bucket == 256)
        {
            dvSplit = gfx12 ? 4 : 2;
        }
        catalog.emplace_back("decode_d" + std::to_string(bucket),
                             decodeKernel(bucket, "none", dvSplit));
    }
    return catalog;
}

/// The shipped object the engine selects for @p graphSpec on @p device: the highest
/// score among the matching ones. A tie names every tied object, joined by '|'.
std::string selectedObject(const GraphSpec& graphSpec, const Device& device = {})
{
    std::string best;
    double bestScore = 0.0;
    for(const auto& [name, spec] : shippedCatalog(device.arch.rfind("gfx12", 0) == 0))
    {
        if(!matchesKernel(graphSpec, spec, device))
        {
            continue;
        }
        const double score = scoreOf(spec, graphSpec);
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

TEST(TestGfx1151WmmaAttentionMatchers, SelectsTheShippedObject)
{
    struct Case
    {
        int64_t headDimQ;
        int64_t headDimV;
        const char* prefill;
    };
    // gfx1151.py prefill_head_buckets over the shipped bucket pairs.
    for(const Case& c : {Case{64, 64, "prefill_d64v64"},
                         Case{80, 80, "prefill_d96v96"},
                         Case{64, 128, "prefill_d128v128"},
                         Case{136, 64, "prefill_d192v128"},
                         Case{192, 128, "prefill_d192v128"},
                         Case{200, 128, "prefill_d256v128"},
                         Case{256, 96, "prefill_d256v128"},
                         Case{128, 192, "prefill_d256v256"},
                         Case{256, 256, "prefill_d256v256"}})
    {
        for(const Device& device : {Device{"gfx1151", 20}, Device{"gfx1201", 32}})
        {
            // 128 query rows of 8 heads x 2 batches fill both devices' prefill CTAs, so
            // the dispatcher's rule picks the prefill kernel.
            GraphSpec spec;
            spec.queryLength = 128;
            spec.headSize = c.headDimQ;
            spec.vHeadSize = c.headDimV;
            EXPECT_EQ(selectedObject(spec, device), c.prefill)
                << "Q " << c.headDimQ << " V " << c.headDimV << " " << device.arch;
        }
    }

    EXPECT_EQ(selectedObject(decodeGraph()), "decode_d128");
    GraphSpec d96 = decodeGraph();
    d96.headSize = 96;
    EXPECT_EQ(selectedObject(d96), "decode_d128");
    // No decode bucket serves Q/K 64 beside V/O 128: prefill takes it.
    GraphSpec q64v128 = decodeGraph();
    q64v128.headSize = 64;
    q64v128.vHeadSize = 128;
    EXPECT_EQ(selectedObject(q64v128), "prefill_d128v128");
}

// -----------------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------------

// Unknown catalog metadata fails at plan build, never at launch.
TEST(TestGfx1151WmmaAttentionMatchers, DispatchRefusesIncompleteMetadata)
{
    const Fixture fixture(GraphSpec{});
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    auto kernel = makeKernel(prefillKernel(96, 96));
    kernel.metadata.erase("block_m");
    const auto& handler = dispatchHandler(PREFILL_SYMBOLS);
    EXPECT_THROW(handler.workspaceBytes(fixture.context(), *bound, kernel),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(handler.prepare(fixture.context(), *bound, kernel),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// The decode engine launches two kernels, the prefill engine one: the launch names are
// what the state manager checks every kernel of an engine against.
TEST(TestGfx1151WmmaAttentionMatchers, EnginesDeclareTheirLaunchNames)
{
    EXPECT_TRUE(dispatchHandler(PREFILL_SYMBOLS).launchNames().empty());
    EXPECT_EQ(dispatchHandler(DECODE_SYMBOLS).launchNames(),
              (std::set<std::string>{"decode", "merge"}));
}

// An engine refuses the other engine's kernel: its matcher never admits one, and its
// handler fails at plan build rather than launching it with the wrong shape of launches.
TEST(TestGfx1151WmmaAttentionMatchers, EachEngineRefusesTheOtherEnginesKernel)
{
    const Fixture fixture(decodeGraph());
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    const auto decode = makeKernel(decodeKernel(128));
    const auto prefill = makeKernel(prefillKernel(128, 128));
    EXPECT_TRUE(matchesKernel(DECODE_SYMBOLS, fixture.context(), decode, *bound));
    EXPECT_FALSE(matchesKernel(PREFILL_SYMBOLS, fixture.context(), decode, *bound));
    EXPECT_FALSE(matchesKernel(DECODE_SYMBOLS, fixture.context(), prefill, *bound));
    EXPECT_THROW(dispatchHandler(PREFILL_SYMBOLS).workspaceBytes(fixture.context(), *bound, decode),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(dispatchHandler(DECODE_SYMBOLS).workspaceBytes(fixture.context(), *bound, prefill),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// Both engines bind a dense graph alike; only paged graphs (TestGfx1151WmmaAttentionPaged)
// are the prefill engine's alone, so the dense split between the two is the kernel
// matchers' decode rule.
TEST(TestGfx1151WmmaAttentionMatchers, BothEnginesBindADenseGraphAlike)
{
    GraphSpec prefill;
    prefill.queryLength = 128;
    for(const GraphSpec& spec : {decodeGraph(), prefill})
    {
        const Fixture fixture(spec);
        const auto bound = fixture.bound();
        ASSERT_TRUE(bound.has_value());
        const auto decodeBound = fixture.bound(DECODE_SYMBOLS);
        EXPECT_TRUE(decodeBound == bound);
    }
}

// -----------------------------------------------------------------------------
// Kernel ABIs. A wrong argument count, order or name is not a launch error: the
// kernel reads the kernarg segment as it was compiled, so the failure is misread
// arguments with a success status. prepare() compares the list the native marshals
// against the descriptor's (requireSignatureMatch: kind, size and name), so a
// descriptor recording the ABI rocKE declares must pass that comparison and fail only
// on the absent archive. No device: the archive named here does not exist.
// -----------------------------------------------------------------------------

void appendBias(std::vector<KernelArgument>& abi)
{
    abi.push_back(pointerArgument("attn_bias_ptr"));
    appendScalars(abi, {"bias_stride_b", "bias_stride_h", "bias_stride_q"});
}

/// wmma_fmha_fwd_v2._declare_params for a shipped object (no softcap, sinks, ALiBi).
std::vector<KernelArgument> rockePrefillAbi(bool bias)
{
    auto abi = rockePrefillAbiHead();
    if(bias)
    {
        appendBias(abi);
    }
    return abi;
}

/// wmma_fmha_decode._declare_params for a shipped object.
std::vector<KernelArgument> rockeDecodeAbi(bool bias)
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"Q", "K", "V", "O", "LSE", "WS_O", "WS_LSE", "WS_CNT"})
    {
        abi.push_back(pointerArgument(name));
    }
    abi.push_back(scalarArgument("scale_log2"));
    appendScalars(abi,
                  {"seqlen_q",
                   "seqlen_k",
                   "num_query_heads",
                   "num_kv_heads",
                   "num_v_heads",
                   "head_dim_q",
                   "head_dim_v",
                   "bottom_right",
                   "window_left",
                   "window_right",
                   "write_lse",
                   "num_splits"});
    appendStrides(abi, {"q", "k", "v", "o", "lse"});
    if(bias)
    {
        appendBias(abi);
    }
    return abi;
}

/// wmma_fmha_decode._declare_merge_params (dense).
std::vector<KernelArgument> rockeMergeAbi()
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"O", "LSE", "WS_O", "WS_LSE"})
    {
        abi.push_back(pointerArgument(name));
    }
    appendScalars(abi, {"seqlen_q", "num_query_heads", "head_dim_v", "num_splits", "write_lse"});
    appendStrides(abi, {"o", "lse"});
    return abi;
}

/// A kpack descriptor of @p spec recording @p recorded. A decode object carries its two
/// launches in `sources`, as hkp_pack writes them: "decode" with @p recorded, and "merge"
/// with @p merge, out of the same archive and toc key.
KernelDefinition kpackKernel(const KernelSpec& spec,
                             const std::vector<KernelArgument>& recorded,
                             const std::vector<KernelArgument>& merge = rockeMergeAbi())
{
    auto kernel = kpackDescriptor(spec, recorded);
    if(isDecode(spec))
    {
        KernelSource mergeSource = kernel.source;
        mergeSource.symbol = "wmma_fmha_decode_merge";
        mergeSource.signature = merge;
        kernel.sources = {{"decode", kernel.source}, {"merge", std::move(mergeSource)}};
    }
    return kernel;
}

/// What prepare() throws for @p kernel against @p graphSpec on a gfx1201 (32 WGP) device;
/// empty when it throws nothing, which no case expects (no archive exists).
std::string prepareFailure(const GraphSpec& graphSpec, const KernelDefinition& kernel)
{
    return gfx1151_wmma_attention::prepareFailure(Fixture(graphSpec, Device{"gfx1201", 32}),
                                                  kernel);
}

/// Asserts prepare() got past every signature comparison and failed only in the loader,
/// which names the symbol it was opening ("..., symbol '<symbol>': ...").
void expectPastTheSignatureCheck(const std::string& failure, const char* what)
{
    ASSERT_FALSE(failure.empty()) << what
                                  << ": the archive does not exist, so prepare() "
                                     "cannot have succeeded";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ": the native refused the ABI rocKE declares: " << failure;
    EXPECT_TRUE(failure.find(", symbol 'wmma_fmha_fwd_v2':") != std::string::npos
                || failure.find(", symbol 'wmma_fmha_decode':") != std::string::npos)
        << what << ": prepare() failed before loading the primary: " << failure;
}

void expectSignatureRefusal(const std::string& failure, const char* what)
{
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

GraphSpec biasedGraph(GraphSpec spec)
{
    spec.bias = packedBias({spec.batch, spec.queryHeads, spec.queryLength, spec.kvLength});
    return spec;
}

TEST(TestGfx1151WmmaAttentionSignature, PrefillMarshalsTheRockeAbi)
{
    expectPastTheSignatureCheck(
        prepareFailure(GraphSpec{}, kpackKernel(prefillKernel(96, 96), rockePrefillAbi(false))),
        "prefill");
    expectPastTheSignatureCheck(
        prepareFailure(biasedGraph(GraphSpec{}),
                       kpackKernel(prefillKernel(96, 96, "f32"), rockePrefillAbi(true))),
        "prefill with bias");
}

TEST(TestGfx1151WmmaAttentionSignature, DecodeAndMergeMarshalTheRockeAbis)
{
    // decodeGraph() runs 8 splits on this device, so the merge launch happens too.
    ASSERT_EQ(workspaceOf(decodeGraph(), decodeKernel(128), Device{"gfx1201", 32}), 132096U);
    expectPastTheSignatureCheck(
        prepareFailure(decodeGraph(), kpackKernel(decodeKernel(128), rockeDecodeAbi(false))),
        "decode");
    expectPastTheSignatureCheck(
        prepareFailure(biasedGraph(decodeGraph()),
                       kpackKernel(decodeKernel(128, "f32"), rockeDecodeAbi(true))),
        "decode with bias");
}

// The merge launch's list is the one packaged under `sources["merge"]` of the decode
// object; a merge list in another order or of another length is refused, and so is a
// decode object packaged without the launch.
TEST(TestGfx1151WmmaAttentionSignature, MergeLaunchIsCheckedUnderItsLaunchName)
{
    auto swapped = rockeMergeAbi();
    std::swap(swapped.at(7), swapped.at(8)); // num_splits <-> write_lse
    expectSignatureRefusal(
        prepareFailure(decodeGraph(),
                       kpackKernel(decodeKernel(128), rockeDecodeAbi(false), swapped)),
        "merge with num_splits and write_lse exchanged");

    auto dropped = rockeMergeAbi();
    dropped.pop_back();
    expectSignatureRefusal(
        prepareFailure(decodeGraph(),
                       kpackKernel(decodeKernel(128), rockeDecodeAbi(false), dropped)),
        "merge without its last stride");

    auto noMerge = kpackKernel(decodeKernel(128), rockeDecodeAbi(false));
    noMerge.sources.erase("merge");
    const std::string failure = prepareFailure(decodeGraph(), noMerge);
    EXPECT_NE(failure.find("no launch 'merge'"), std::string::npos) << failure;
}

TEST(TestGfx1151WmmaAttentionSignature, RefusesEachExchangeOfTwoLikeDecodeSlots)
{
    // Kind and size agree position by position, so only the names separate these from
    // the shipped ABI; accepted, the kernel would read num_splits as write_lse, or K as Q.
    const auto abi = rockeDecodeAbi(false);
    for(size_t first = 0; first < abi.size(); ++first)
    {
        for(size_t second = first + 1; second < abi.size(); ++second)
        {
            if(abi.at(first).kind != abi.at(second).kind)
            {
                continue;
            }
            auto exchanged = abi;
            std::swap(exchanged.at(first), exchanged.at(second));
            const std::string what = abi.at(first).name + " <-> " + abi.at(second).name;
            expectSignatureRefusal(
                prepareFailure(decodeGraph(), kpackKernel(decodeKernel(128), exchanged)),
                what.c_str());
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

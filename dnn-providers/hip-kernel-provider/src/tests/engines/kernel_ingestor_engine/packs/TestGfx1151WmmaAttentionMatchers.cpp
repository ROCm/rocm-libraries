// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSource;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

/// The native symbols one engine registers (hipkernel:Gfx1151WmmaAttention is the
/// prefill, hipkernel:Gfx1151WmmaAttentionDecode the decode).
struct EngineSymbols
{
    std::string_view graphMatcher;
    std::string_view kernelMatcher;
    std::string_view score;
    std::string_view dispatch;
};

constexpr EngineSymbols PREFILL_SYMBOLS{"hipkernel.gfx1151_wmma_attention.graph_match",
                                        "hipkernel.gfx1151_wmma_attention.kernel_match",
                                        "hipkernel.gfx1151_wmma_attention.score",
                                        "hipkernel.gfx1151_wmma_attention.dispatch"};
constexpr EngineSymbols DECODE_SYMBOLS{"hipkernel.gfx1151_wmma_attention_decode.graph_match",
                                       "hipkernel.gfx1151_wmma_attention_decode.kernel_match",
                                       "hipkernel.gfx1151_wmma_attention_decode.score",
                                       "hipkernel.gfx1151_wmma_attention_decode.dispatch"};

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
    bool bottomRight = false;
    ScaleForm scale = ScaleForm::VALUE;
    // The VALUE form's attn_scale_value and the CONSTANT_TENSOR form's baked value, when
    // not their defaults (0.125 and 0.25).
    std::optional<float> scaleValue;
    bool overrideShape = false;
    // Elements of padding after each Q row: a row stride that is not a multiple of 8
    // elements leaves rows off the kernels' 16-byte vector alignment.
    int64_t qRowPad = 0;
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
    DeviceProperties properties;
    properties.gcnArchName = device.arch;
    properties.warpSize = 32;
    properties.multiProcessorCount = device.multiprocessors;
    return properties;
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
    graph.add_is_override_shape_enabled(spec.overrideShape);
    builder.Finish(graph.Finish());
    return builder;
}

/// A graph and the device it is matched for.
class Fixture
{
public:
    explicit Fixture(const GraphSpec& spec, const Device& device = {})
        : _builder(buildSdpaGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(testDeviceProperties(device))
    {
        registerNativeIngestorSymbols();
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

    std::optional<BoundTokens> bound() const
    {
        return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(PREFILL_SYMBOLS.graphMatcher))(context());
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    return Fixture(spec).bound();
}

/// The catalog metadata of one object (configs/gfx1151_wmma_attention.yaml).
struct KernelSpec
{
    std::string kind = "prefill";
    std::string dtype = "FP16";
    int64_t headSize = 128;
    int64_t headSizeV = 128;
    std::string maskMode = "causal";
    std::string bias = "none";
    int64_t blockSize = 256;
    int64_t blockM = 128;
    int64_t dvSplit = 1;
};

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
                       {"kind", spec.kind},
                       {"head_size", spec.headSize},
                       {"head_size_v", spec.headSizeV},
                       {"mask_mode", spec.maskMode},
                       {"bias", spec.bias},
                       {"block_size", spec.blockSize},
                       {"block_m", spec.blockM},
                       {"dv_split", spec.dvSplit}};
    return kernel;
}

bool isDecode(const KernelSpec& spec)
{
    return spec.kind == "decode";
}

const EngineSymbols& symbolsFor(const KernelSpec& spec)
{
    return isDecode(spec) ? DECODE_SYMBOLS : PREFILL_SYMBOLS;
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
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(std::string(
        symbolsFor(kernelSpec).kernelMatcher))(fixture.context(), *bound, makeKernel(kernelSpec));
}

double scoreOf(const KernelSpec& kernelSpec, const GraphSpec& graphSpec = {})
{
    const Fixture fixture(graphSpec);
    const auto bound = fixture.bound();
    if(!bound.has_value())
    {
        throw std::logic_error("the scored graph does not match");
    }
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(
        symbolsFor(kernelSpec).score))(fixture.context(), *bound, makeKernel(kernelSpec));
}

/// The registered dispatch handler of one engine; a named failure, not a null
/// dereference, when the symbol the catalog's descriptors name is not registered.
const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>&
    dispatchHandler(const EngineSymbols& symbols)
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(symbols.dispatch));
    if(handler == nullptr)
    {
        throw std::logic_error(std::string(symbols.dispatch) + " is not registered");
    }
    return *handler;
}

/// The handler of the engine @p kernel's own `kind` metadata names.
const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>&
    dispatchHandlerOf(const KernelDefinition& kernel)
{
    const auto kind = kernel.metadata.find("kind");
    const bool decode = kind != kernel.metadata.end()
                        && std::holds_alternative<std::string>(kind->second)
                        && std::get<std::string>(kind->second) == "decode";
    return dispatchHandler(decode ? DECODE_SYMBOLS : PREFILL_SYMBOLS);
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

// The deprecated causal flags override the bounds and alignment, as in the
// hipDNN reference; otherwise a left bound would drop the causal limit.
TEST(TestGfx1151WmmaAttentionMatchers, DeprecatedCausalFlagsOverrideBounds)
{
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
        // A causal request is served by the causal class through runtime bounds;
        // never by a no-mask object.
        EXPECT_TRUE(matchesKernel(spec, prefillKernel(96, 96)));
        EXPECT_FALSE(matchesKernel(spec, prefillKernel(96, 96, "none", "none")));
    }
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
    window.leftBound = 1023;
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
    window.leftBound = 1023;
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
    EXPECT_TRUE(hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(DECODE_SYMBOLS.kernelMatcher))(fixture.context(), *bound, decode));
    EXPECT_FALSE(hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(PREFILL_SYMBOLS.kernelMatcher))(fixture.context(), *bound, decode));
    EXPECT_FALSE(hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(DECODE_SYMBOLS.kernelMatcher))(fixture.context(), *bound, prefill));
    EXPECT_THROW(dispatchHandler(PREFILL_SYMBOLS).workspaceBytes(fixture.context(), *bound, decode),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(dispatchHandler(DECODE_SYMBOLS).workspaceBytes(fixture.context(), *bound, prefill),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// -----------------------------------------------------------------------------
// Kernel ABIs. A wrong argument count, order or name is not a launch error: the
// kernel reads the kernarg segment as it was compiled, so the failure is misread
// arguments with a success status. prepare() compares the list the native marshals
// against the descriptor's (requireSignatureMatch: kind, size and name), so a
// descriptor recording the ABI rocKE declares must pass that comparison and fail only
// on the absent archive. No device: the archive named here does not exist.
// -----------------------------------------------------------------------------

constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

KernelArgument pointerArgument(const char* name)
{
    return KernelArgument{"global_buffer", 8, 0, name};
}

KernelArgument scalarArgument(const std::string& name)
{
    return KernelArgument{"by_value", 4, 0, name};
}

void appendScalars(std::vector<KernelArgument>& abi, std::initializer_list<const char*> names)
{
    for(const char* name : names)
    {
        abi.push_back(scalarArgument(name));
    }
}

void appendStrides(std::vector<KernelArgument>& abi, std::initializer_list<const char*> tensors)
{
    for(const char* tensor : tensors)
    {
        for(const char* axis : {"batch", "token", "head"})
        {
            abi.push_back(scalarArgument(std::string("stride_") + tensor + "_" + axis));
        }
    }
}

void appendBias(std::vector<KernelArgument>& abi)
{
    abi.push_back(pointerArgument("attn_bias_ptr"));
    appendScalars(abi, {"bias_stride_b", "bias_stride_h", "bias_stride_q"});
}

/// wmma_fmha_fwd_v2._declare_params for a shipped object (no softcap, sinks, ALiBi).
std::vector<KernelArgument> rockePrefillAbi(bool bias)
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"Q", "K", "V", "O", "LSE"})
    {
        abi.push_back(pointerArgument(name));
    }
    abi.push_back(scalarArgument("scale_log2"));
    appendScalars(abi,
                  {"batch",
                   "seqlen_q",
                   "seqlen_k",
                   "num_query_heads",
                   "num_kv_heads",
                   "num_v_heads",
                   "head_dim_q",
                   "head_dim_v",
                   "bottom_right",
                   "window_left",
                   "window_right",
                   "write_lse"});
    appendStrides(abi, {"q", "k", "v", "o", "lse"});
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
    auto kernel = makeKernel(spec);
    kernel.name = "wmma_attention.signature_fixture";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = isDecode(spec) ? "wmma_fmha_decode" : "wmma_fmha_fwd_v2";
    kernel.source.signature = recorded;
    // An existing tree, so the containment check passes and the signature comparison
    // runs; the archive itself is absent from it.
    const auto tree = std::filesystem::temp_directory_path().string();
    kernel.originDirectory = tree;
    kernel.treeRoot = tree;
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
    const Fixture fixture(graphSpec, Device{"gfx1201", 32});
    const auto bound = fixture.bound();
    EXPECT_TRUE(bound.has_value()) << "the fixture graph does not match, so prepare() never ran";
    if(!bound.has_value())
    {
        return {};
    }
    try
    {
        dispatchHandlerOf(kernel).prepare(fixture.context(), *bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
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

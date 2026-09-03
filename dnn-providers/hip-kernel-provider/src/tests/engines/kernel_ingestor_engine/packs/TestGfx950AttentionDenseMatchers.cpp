// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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

/**
 * @file TestGfx950AttentionDenseMatchers.cpp
 * @brief Applicability negatives for hipkernel:Gfx950AttentionDense.
 *
 * One case per "must decline" row of the rejection checklist in `mining.md`, in the
 * same severity order: the silent-wrong-answer rows first, then the faults, then the
 * declined features. Each of these is a graph the kernel would accept and compute
 * something plausible-but-wrong for if the matcher did not stop it -- which is exactly
 * why they are C++ tests and not bundles. A bundle for a graph this engine declines is
 * simply served by another engine and proves nothing.
 *
 * These are matcher-only: no device, no compile, no launch.
 *
 * THREE FAMILIES ARE gfx950-SPECIFIC and have no gfx942 counterpart:
 *   - RAGGED. gfx942 declines every non-tile-multiple length; gfx950 serves them
 *     through a separately compiled boundary-padding path, so kernel_match's tile
 *     rule is conditional on the candidate's own `ragged` flag. Both directions are
 *     tested: a ragged graph must not select an aligned binary (whose grid would not
 *     cover the partial final block) and vice versa.
 *   - SINKS and SLIDING WINDOW. The gfx950 kernel implements both, and this
 *     integration declines both -- sinks because no available reference executor can
 *     verify them, windows because every windowed shape in the corpus also carries
 *     sinks. Those declines are load-bearing rather than cosmetic: the kernel's
 *     signature grows extra pointer slots for either feature, and the shipped launch
 *     passes exactly five arguments.
 *   - NO block_m. It is a baked module constant here, not a KMD field.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
/// Named by an attribute under test but never inserted into the tensor map; every
/// optional-feature negative only needs the UID to be PRESENT to be declined.
constexpr int64_t EXTRA_UID = 99;

/// The shape every case starts from: a graph a shipped variant serves.
/// bf16, D128, B=2, Hq=Hkv=4, Sq=Skv=256 -- the quick-tier mha256 cell.
constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 256;
constexpr int64_t HEAD_SIZE = 128;
constexpr float SCALE = 0.08838834764831843F;

/**
 * @brief A fixed, warp-64 gfx942 device, constructed BY VALUE.
 *
 * Never queried from the host: an arch-gated matcher test that calls
 * hipGetDeviceProperties() is vacuous everywhere except whatever arch CI happens to be
 * running on, and would silently stop testing the one thing it exists to test the
 * moment that hardware changes.
 */
DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return properties;
}

/// BSHD strides for (B, H, S, D) LOGICAL dims -- token-major, head varying fastest.
/// This is what the kernel's address arithmetic bakes in.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

/// BHSD strides for the same logical dims -- the layout every SHIPPED SdpaFwd bundle
/// uses, and the one this kernel must decline.
std::vector<int64_t> bhsdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {heads * sequence * headSize, sequence * headSize, headSize, 1};
}

/// Everything a case may vary. Defaults describe the servable graph above; each test
/// perturbs exactly one field, so a decline is attributable to that field alone.
struct GraphSpec
{
    int64_t batch = BATCH;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t headSize = HEAD_SIZE;
    int64_t headSizeV = HEAD_SIZE;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    std::optional<data_objects::DataType> vDataType;
    bool bhsd = false;
    bool omitStrides = false;

    // Mask, in the modern spelling. Defaults to top-left causal.
    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bool causalMaskDeprecated = false;
    bool causalMaskBottomRightDeprecated = false;

    std::optional<float> attnScaleValue = SCALE;

    // Optional features, each declined.
    std::optional<int64_t> attnMaskUid;
    std::optional<int64_t> scaleTensorUid;
    std::optional<int64_t> seqLenQUid;
    std::optional<int64_t> pageTableKUid;
    std::optional<int64_t> sinkTokenUid;
    std::optional<int64_t> blockMaskUid;
    std::optional<int64_t> statsUid;
    std::optional<int64_t> descaleQUid;
    std::optional<float> dropoutProbability;
    std::optional<bool> generateStats;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    /// Two nodes rather than one; no single-graph kernel serves it.
    bool twoNodes = false;
};

flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    const auto strides = [&](int64_t heads, int64_t sequence, int64_t headSize) {
        return spec.bhsd ? bhsdStrides(heads, sequence, headSize)
                         : bshdStrides(heads, sequence, headSize);
    };

    const std::vector<int64_t> qDims{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSize};
    const std::vector<int64_t> kDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const std::vector<int64_t> vDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSizeV};
    const std::vector<int64_t> oDims{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSizeV};

    const auto qStrides = strides(spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto kStrides = strides(spec.numKvHeads, spec.seqLenKv, spec.headSize);
    const auto vStrides = strides(spec.numKvHeads, spec.seqLenKv, spec.headSizeV);
    const auto oStrides = strides(spec.numQueryHeads, spec.seqLenQ, spec.headSizeV);

    // Null, not empty: the field is omitted entirely, so strides() returns nullptr.
    // Applicability runs before anything has validated a caller-supplied graph.
    const std::vector<int64_t>* const qStridesPtr = spec.omitStrides ? nullptr : &qStrides;
    const std::vector<int64_t>* const kStridesPtr = spec.omitStrides ? nullptr : &kStrides;
    const std::vector<int64_t>* const vStridesPtr = spec.omitStrides ? nullptr : &vStrides;
    const std::vector<int64_t>* const oStridesPtr = spec.omitStrides ? nullptr : &oStrides;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Q_UID, nullptr, spec.dataType, qStridesPtr, &qDims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, K_UID, nullptr, spec.dataType, kStridesPtr, &kDims, false));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   V_UID,
                                                   nullptr,
                                                   spec.vDataType.value_or(spec.dataType),
                                                   vStridesPtr,
                                                   &vDims,
                                                   false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, O_UID, nullptr, spec.dataType, oStridesPtr, &oDims, false));

    const auto attributesFor = [&]() {
        data_objects::SdpaAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_q_tensor_uid(Q_UID);
        attributesBuilder.add_k_tensor_uid(K_UID);
        attributesBuilder.add_v_tensor_uid(V_UID);
        attributesBuilder.add_o_tensor_uid(O_UID);

        if(spec.leftBound.has_value())
        {
            attributesBuilder.add_left_bound(*spec.leftBound);
        }
        if(spec.rightBound.has_value())
        {
            attributesBuilder.add_right_bound(*spec.rightBound);
        }
        attributesBuilder.add_diagonal_alignment(spec.alignment);
        attributesBuilder.add_causal_mask(spec.causalMaskDeprecated);
        attributesBuilder.add_causal_mask_bottom_right(spec.causalMaskBottomRightDeprecated);
        if(spec.attnScaleValue.has_value())
        {
            attributesBuilder.add_attn_scale_value(*spec.attnScaleValue);
        }

        if(spec.attnMaskUid.has_value())
        {
            attributesBuilder.add_attn_mask_tensor_uid(*spec.attnMaskUid);
        }
        if(spec.scaleTensorUid.has_value())
        {
            attributesBuilder.add_scale_tensor_uid(*spec.scaleTensorUid);
        }
        if(spec.seqLenQUid.has_value())
        {
            attributesBuilder.add_seq_len_q_tensor_uid(*spec.seqLenQUid);
        }
        if(spec.pageTableKUid.has_value())
        {
            attributesBuilder.add_page_table_k_tensor_uid(*spec.pageTableKUid);
        }
        if(spec.sinkTokenUid.has_value())
        {
            attributesBuilder.add_sink_token_tensor_uid(*spec.sinkTokenUid);
        }
        if(spec.blockMaskUid.has_value())
        {
            attributesBuilder.add_block_mask_tensor_uid(*spec.blockMaskUid);
        }
        if(spec.statsUid.has_value())
        {
            attributesBuilder.add_stats_tensor_uid(*spec.statsUid);
        }
        if(spec.descaleQUid.has_value())
        {
            attributesBuilder.add_descale_q_tensor_uid(*spec.descaleQUid);
        }
        if(spec.dropoutProbability.has_value())
        {
            attributesBuilder.add_dropout_probability(*spec.dropoutProbability);
        }
        if(spec.generateStats.has_value())
        {
            attributesBuilder.add_generate_stats(*spec.generateStats);
        }
        attributesBuilder.add_alibi_mask(spec.alibiMask);
        attributesBuilder.add_padding_mask(spec.paddingMask);
        attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
        attributesBuilder.add_implementation(spec.implementation);
        return attributesBuilder.Finish();
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributesFor().Union()));
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("gfx950_attention_dense_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// Resolving the matcher by the symbol name its DESCRIPTORS carry, rather than calling
/// the C++ function directly, is what makes a descriptor naming a symbol nothing
/// implements fail here rather than at load time on a device.
std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));

    auto builder = buildSdpaGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    return matcher(context);
}

constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_dense.score";

/// A candidate whose baked metadata describes the default GraphSpec above. Each test
/// perturbs ONE field, so a refusal is attributable to that field alone -- the same
/// discipline the graph_match cases use.
///
/// Hand-built rather than read from the shipped bundle on purpose: these tests must be
/// able to express a variant that does NOT exist, which is exactly what proves
/// kernel_match compares rather than waves through. TestGfx950AttentionDensePacks.cpp
/// covers the complementary direction -- that the SHIPPED descriptors say what the
/// config promised.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = HEAD_SIZE;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t batch = BATCH;
    int64_t causal = 1;
    int64_t blockN = 64;
    // No block_m: gfx950 bakes _BLOCK_M=256 as a module constant, so there is
    // no such KMD field to vary. A candidate carrying one would be describing a
    // tile the binary cannot have.
    int64_t ragged = 0;
    int64_t slidingWindow = 0;
    int64_t useSinks = 0;
    int64_t varlen = 0;
    int64_t paged = 0;
    /// Emit the ABI-extension fields at all. A descriptor predating them omits them
    /// entirely, and the matcher must read that silence as "not built with it"
    /// rather than throwing -- getIntMetadata THROWS on a missing key, so this
    /// distinction is the difference between a routine non-match and an exception
    /// escaping the matcher.
    bool emitAbiFields = true;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea3");
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("seqlen_q"), spec.seqLenQ},
        {std::string("seqlen_kv"), spec.seqLenKv},
        {std::string("batch"), spec.batch},
        {std::string("causal"), spec.causal},
        {std::string("block_n"), spec.blockN},
        {std::string("ragged"), spec.ragged},
    };
    if(spec.emitAbiFields)
    {
        kernel.metadata[std::string("sliding_window")] = spec.slidingWindow;
        kernel.metadata[std::string("use_sinks")] = spec.useSinks;
        kernel.metadata[std::string("varlen")] = spec.varlen;
        kernel.metadata[std::string("paged")] = spec.paged;
    }
    return kernel;
}

/// Runs graph_match then kernel_match, as the runtime does: kernel_match reads the
/// tokens graph_match bound, so calling it on an unmatched graph is not a state the
/// engine can reach.
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
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(context, *bound, makeKernel(kernelSpec));
}

double scoreOf(const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));

    auto builder = buildSdpaGraph(GraphSpec{});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto bound = graphMatcher(context);
    EXPECT_TRUE(bound.has_value());
    return scorer(context, bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

// ---------------------------------------------------------------------------
// The positive control. Every negative below is only meaningful because this passes:
// a matcher that declines everything would pass all the negatives.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsDenseBshdCausalGraph)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsNoMaskGraph)
{
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsGroupedQueryAttention)
{
    GraphSpec spec;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 2;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsHeadSize64)
{
    // The gfx950 corpus's gpt-oss traces are D=64; __post_init__ allows 64 and 128.
    GraphSpec spec;
    spec.headSize = 64;
    spec.headSizeV = 64;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Silent-wrong-answer rows: the graph would compute something plausible
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdLayout)
{
    // The kernel bakes BSHD strides and takes no stride kernargs, so a BHSD graph is
    // indexed as if it were BSHD: in-bounds reads of the wrong elements, no fault.
    GraphSpec spec;
    spec.bhsd = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsSingleHeadUnderEitherStrideSpelling)
{
    // A single-head tensor is byte-identically BSHD and BHSD while the two spellings
    // disagree on strides[H]. A strict compare would decline a graph the kernel serves
    // perfectly -- and graph_match returning nullopt empties the WHOLE catalog.
    GraphSpec bshd;
    bshd.numQueryHeads = 1;
    bshd.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(bshd).has_value());

    GraphSpec bhsd = bshd;
    bhsd.bhsd = true;
    EXPECT_TRUE(matchGraph(bhsd).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBottomRightCausalWhenSeqLensDiffer)
{
    // The kernel's causal clamp is TOP-LEFT. Bottom-right differs by a (Skv - Sq)
    // offset, so the two coincide only when Sq == Skv; serving it otherwise is a
    // silent wrong answer.
    GraphSpec spec;
    spec.seqLenKv = SEQ * 2;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsBottomRightCausalWhenSeqLensMatch)
{
    // The complement, and it matters: every shipped quick/SdpaFwd causal bundle sets
    // BOTTOM_RIGHT at Sq == Skv, so declining it outright declines all of them.
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedHeadSize)
{
    // D=256 belongs to the gfx950_d256 sibling candidate. __post_init__ requires
    // head_size in (64, 128): the MFMA tiling needs a multiple of 32 and the async
    // K/V DMA needs 128 % head_size == 0.
    GraphSpec spec;
    spec.headSize = 256;
    spec.headSizeV = 256;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMismatchedHeadSizes)
{
    // hipDNN permits D_qk != D_v; the kernel has ONE head_size.
    GraphSpec spec;
    spec.headSizeV = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesNonDivisibleGqaGrouping)
{
    // The kernel derives its group size by integer division, so a non-divisible pair
    // silently drops heads.
    GraphSpec spec;
    spec.numQueryHeads = 6;
    spec.numKvHeads = 4;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMixedOperandDataTypes)
{
    GraphSpec spec;
    spec.vDataType = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedDataType)
{
    GraphSpec spec;
    spec.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMissingAttentionScale)
{
    // The scale is a REQUIRED kernarg with no default. The mathematically obvious
    // 1/sqrt(D) would silently override whatever the frontend's omission meant.
    GraphSpec spec;
    spec.attnScaleValue = std::nullopt;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Faults and malformed input: graph_match must be TOTAL over an unvalidated graph
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesGraphWithNoStrides)
{
    GraphSpec spec;
    spec.omitStrides = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMultiNodeGraph)
{
    GraphSpec spec;
    spec.twoNodes = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBothDeprecatedCausalBooleans)
{
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesGraphsPastThe32BitExtentLimit)
{
    // Offsets lower to i32 add/mul nsw: signed overflow is UB, so LLVM may poison the
    // whole address chain rather than merely read the wrong place.
    GraphSpec spec;
    spec.batch = 1;
    spec.numQueryHeads = 128;
    spec.numKvHeads = 128;
    spec.seqLenQ = 131072;
    spec.seqLenKv = 131072;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Declined features. Each is a graph the kernel would otherwise accept and then
// silently not perform.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAdditiveAttentionMask)
{
    GraphSpec spec;
    spec.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDeviceResidentScaleTensor)
{
    GraphSpec spec;
    spec.scaleTensorUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesVarlen)
{
    // The kernel HAS a varlen path, but it appends two cu_seqlens pointers to the
    // signature -- arguments the shipped 5-slot launch does not pass.
    GraphSpec spec;
    spec.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPagedKv)
{
    GraphSpec spec;
    spec.pageTableKUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAttentionSinks)
{
    // THE SCOPE DECISION, made enforceable. The gfx950 kernel implements sinks and
    // rocKE serves these shapes; this integration declines them because no reference
    // executor available to the suite can verify one (the GPU ref, the CPU ref and
    // dnn-benchmarking's PyTorch handler all reject sink_token_tensor_uid). Serving
    // one would also launch a binary whose signature expects a sixth pointer.
    GraphSpec spec;
    spec.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBlockSparseMask)
{
    GraphSpec spec;
    spec.blockMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSoftmaxStatsBothSpellings)
{
    GraphSpec byUid;
    byUid.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(byUid).has_value());

    GraphSpec byFlag;
    byFlag.generateStats = true;
    EXPECT_FALSE(matchGraph(byFlag).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsExplicitlyDisabledStats)
{
    // generate_stats is optional<bool>: an explicit false is a request for nothing,
    // not a request the kernel cannot honour.
    GraphSpec spec;
    spec.generateStats = false;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDropout)
{
    GraphSpec spec;
    spec.dropoutProbability = 0.1F;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesFp8Descale)
{
    GraphSpec spec;
    spec.descaleQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAlibiMask)
{
    GraphSpec spec;
    spec.alibiMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddingMask)
{
    GraphSpec spec;
    spec.paddingMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesNonAutoImplementationHint)
{
    GraphSpec spec;
    spec.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsMmaCoreModeFloat)
{
    // An allow-list, not `!= UNSET`: every shipped SdpaFwd bundle sets "float", so the
    // naive form silently declines all of them.
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::FLOAT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedMmaCoreMode)
{
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Sliding window. The kernel implements it; this integration declines it, and the
// ORDERING of that decline is what stops a windowed graph being served as plain
// causal -- a wrong answer rather than a decline.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindow)
{
    GraphSpec spec;
    spec.leftBound = 128;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindowSetAlongsideDeprecatedCausal)
{
    // A REAL BOUND WINS OVER THE DEPRECATED BOOLEAN. Returning on the boolean first
    // serves this graph as plain causal: the window is discarded and the kernel
    // attends the whole triangle. The gfx950 corpus's swa_sink_prefill traces are
    // exactly this shape.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = 128;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, StillServesPlainDeprecatedCausalWithNoBound)
{
    // The control for the two above: the bound-wins rule must not over-fire and
    // decline an ordinary deprecated-causal graph carrying no window.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// kernel_match: does THIS candidate's baked metadata fit the graph?
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseKernelMatch, AcceptsTheCandidateBakedForThisGraph)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherDtype)
{
    KernelSpec kernel;
    kernel.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherBatch)
{
    // Every shape field is compiled in: the K/V buffer resources are sized at build
    // time, so a larger batch reads zero-fill past the bound rather than faulting.
    KernelSpec kernel;
    kernel.batch = BATCH + 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherSeqLen)
{
    KernelSpec kernel;
    kernel.seqLenKv = SEQ * 2;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherHeadCount)
{
    KernelSpec kernel;
    kernel.numQueryHeads = HEADS * 2;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForTheOtherMask)
{
    KernelSpec kernel;
    kernel.causal = 0; // the default GraphSpec is causal
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// Ragged: the arm gfx942 cannot express. Both directions, because each one alone
// would pass with a matcher that ignored the flag.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseKernelMatch, RefusesARaggedCandidateForAnAlignedGraph)
{
    // Different binaries: the ragged path pads boundary tiles on-chip and ceils its
    // grid. Selecting it for an aligned shape runs boundary handling the shape does
    // not need -- and, more to the point, is not the binary the descriptor set built
    // for this shape.
    KernelSpec kernel;
    kernel.ragged = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateForARaggedGraph)
{
    // The dangerous direction. An aligned binary's grid does not cover the partial
    // final query block, so the tail rows are never written and the kernel returns
    // cleanly.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 4000;
    KernelSpec aligned; // ragged = 0
    aligned.seqLenQ = 4000;
    aligned.seqLenKv = 4000;
    EXPECT_FALSE(matchesKernel(graph, aligned));
}

TEST(TestGfx950AttentionDenseKernelMatch, AcceptsARaggedCandidateForARaggedGraph)
{
    // The positive control for the pair above: without it, a matcher that refused
    // every ragged combination would pass both negatives.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 4000;
    KernelSpec ragged;
    ragged.ragged = 1;
    ragged.seqLenQ = 4000;
    ragged.seqLenKv = 4000;
    EXPECT_TRUE(matchesKernel(graph, ragged));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesARaggedCandidateForCrossAttention)
{
    // The ragged path is SELF-ATTENTION ONLY: __post_init__ rejects
    // seqlen_q != seqlen_kv for ragged.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 8000;
    graph.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    KernelSpec ragged;
    ragged.ragged = 1;
    ragged.seqLenQ = 4000;
    ragged.seqLenKv = 8000;
    EXPECT_FALSE(matchesKernel(graph, ragged));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateWhoseTileDoesNotDivideSeqLenKv)
{
    // block_n divides Skv on the aligned path. block_m is not checked against a KMD
    // field because gfx950 has none -- it is the baked constant 256.
    GraphSpec graph;
    graph.seqLenKv = 288; // not a multiple of 64
    graph.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    KernelSpec kernel;
    kernel.seqLenKv = 288;
    kernel.blockN = 64;
    EXPECT_FALSE(matchesKernel(graph, kernel));
}

// ---------------------------------------------------------------------------
// A variant compiled FOR a declined feature must never be selected, even if such a
// descriptor reached the catalog by mistake: the launch passes five arguments and
// those binaries expect more.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseKernelMatch, RefusesASinkVariantForAPlainGraph)
{
    KernelSpec kernel;
    kernel.useSinks = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAWindowedVariantForAPlainGraph)
{
    KernelSpec kernel;
    kernel.slidingWindow = 128;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAVarlenVariantForAPlainGraph)
{
    // varlen appends cu_seqlens_q and cu_seqlens_kv to the kernel signature -- two
    // more pointer arguments than the shipped 5-slot launch passes. A review found
    // this guard missing: sliding_window and use_sinks were compared and varlen and
    // paged were not, so the ONLY thing stopping a varlen-built descriptor from
    // serving a plain graph was graph_match declining varlen GRAPHS, which is a
    // different question and a single point of failure.
    KernelSpec kernel;
    kernel.varlen = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAPagedVariantForAPlainGraph)
{
    // paged appends three (block_tables, kv_lens, block_table_stride).
    KernelSpec kernel;
    kernel.paged = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, TreatsAnAbsentAbiFieldAsNotBuiltWithIt)
{
    // THE CONTROL, and it is the one that caught a real defect in the first version
    // of this guard. `getIntMetadata` THROWS on a missing key rather than defaulting,
    // so a guard written as `intField(VARLEN_FIELD) != 0` turns every descriptor that
    // predates these fields -- which is every descriptor written before the review --
    // into an exception escaping the matcher rather than a routine match. Six tests
    // failed on that, including the positive control, which is how it was found.
    //
    // A descriptor's silence about a feature is the assertion that it was not built
    // with it. Only an explicit non-zero refuses.
    KernelSpec kernel;
    kernel.emitAbiFields = false;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// score
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseScore, RanksOnARealKnobRatherThanReturningAConstant)
{
    // A constant score makes ranking arbitrary and hides mis-specialization, so it is
    // not a legitimate placeholder however honestly it is disclosed.
    KernelSpec wide;
    wide.blockN = 128;
    KernelSpec narrow;
    narrow.blockN = 64;
    EXPECT_NE(scoreOf(wide), scoreOf(narrow));
}

TEST(TestGfx950AttentionDenseScore, IgnoresAxesThatKernelMatchAlreadyPinned)
{
    // dtype/head_size/shape are decided by kernel_match, so score must not also rank
    // on them -- two candidates differing only there would otherwise be ordered by an
    // axis that carries no performance meaning.
    const KernelSpec bf16;
    KernelSpec fp16;
    fp16.dtype = "FP16";
    EXPECT_EQ(scoreOf(bf16), scoreOf(fp16));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

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
 * @file TestGfx942AttentionDenseMatchers.cpp
 * @brief Applicability negatives for hipkernel:Gfx942AttentionDense.
 *
 * One case per "must decline" row of the rejection checklist in `mining.md`, in the
 * same severity order: the silent-wrong-answer rows first, then the faults, then the
 * declined features. Each of these is a graph the kernel would accept and compute
 * something plausible-but-wrong for if the matcher did not stop it -- which is exactly
 * why they are C++ tests and not bundles. A bundle for a graph this engine declines is
 * simply served by another engine and proves nothing.
 *
 * These are matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.graph_match";

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
    properties.gcnArchName = "gfx942";
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

    auto name = builder.CreateString("gfx942_attention_dense_test");
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

constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_dense.score";

/// A candidate whose baked metadata describes the default GraphSpec above. Each test
/// perturbs ONE field, so a refusal is attributable to that field alone -- the same
/// discipline the graph_match cases use.
///
/// Hand-built rather than read from the shipped bundle on purpose: these tests must be
/// able to express a variant that does NOT exist, which is exactly what proves
/// kernel_match compares rather than waves through. TestGfx942AttentionDensePacks.cpp
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
    int64_t blockM = 256;
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
        {std::string("block_m"), spec.blockM},
    };
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

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsDenseBshdCausalGraph)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsNoMaskGraph)
{
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsDeprecatedCausalBoolean)
{
    // The deprecated boolean takes precedence over the trio when set.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsGroupedQueryAttention)
{
    GraphSpec spec;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 2;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsMmaCoreModeFloat)
{
    // Every shipped SdpaFwd bundle sets "float". A `!= UNSET` rejection would decline
    // all of them SILENTLY -- the engine would register, build, and never be selected.
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::FLOAT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Tier 1 -- silent wrong answers. The kernel would compute a plausible result.
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesBhsdLayout)
{
    // THE case this whole file exists for. The kernel bakes BSHD and takes no stride
    // arguments, so a BHSD graph is indexed as if it were BSHD: in-bounds reads of the
    // wrong elements, no fault, wrong numbers. Every shipped quick/SdpaFwd bundle is
    // BHSD, so without this check the engine would happily serve all of them wrong.
    GraphSpec spec;
    spec.bhsd = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsSingleHeadUnderEitherStrideSpelling)
{
    // The degenerate case, and the reason the layout check exempts unit-extent axes.
    // At H == 1 the head index is always 0, so no address depends on strides[H] and
    // the tensor is byte-identically BSHD and BHSD -- while the two spellings disagree
    // on that stride. A strict compare would decline a graph the kernel serves
    // perfectly, and graph_match returning nullopt empties the WHOLE engine catalog.
    // Invisible by inspection; this is the test that pins it.
    GraphSpec bshd;
    bshd.numQueryHeads = 1;
    bshd.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(bshd).has_value());

    GraphSpec bhsd = bshd;
    bhsd.bhsd = true;
    EXPECT_TRUE(matchGraph(bhsd).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesBottomRightCausalWhenSeqLensDiffer)
{
    // The kernel's causal clamp is top-left (no Skv - Sq offset term). Bottom-right
    // differs from top-left by exactly that offset, so serving this at Sq != Skv would
    // mask the wrong triangle -- a silent wrong answer.
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    spec.seqLenKv = 512;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsBottomRightCausalWhenSeqLensMatch)
{
    // ...and accepts it when Sq == Skv, where the two alignments coincide exactly.
    // Load-bearing: every shipped causal SdpaFwd bundle sets BOTTOM_RIGHT at Sq == Skv,
    // so an outright decline would serve none of them.
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesMismatchedHeadSizes)
{
    // hipDNN permits D_qk != D_v; the kernel has ONE head_size and would read V with
    // Q's stride arithmetic.
    GraphSpec spec;
    spec.headSizeV = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesMixedOperandDataTypes)
{
    // One spec.dtype types all four pointers; a bf16 Q with an fp16 V would be read at
    // the wrong element width.
    GraphSpec spec;
    spec.vDataType = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesUnsupportedDataType)
{
    GraphSpec spec;
    spec.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesGraphsPastThe32BitExtentLimit)
{
    // Offsets lower to `add nsw` / `mul nsw` i32: signed overflow is UB, so LLVM may
    // poison the whole address chain rather than merely read the wrong place.
    GraphSpec spec;
    spec.batch = 4096;
    spec.seqLenQ = 4096;
    spec.seqLenKv = 4096;
    spec.numQueryHeads = 32;
    spec.numKvHeads = 32;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesMissingAttentionScale)
{
    // The ABI takes `scale` as a required f32 kernarg. The mathematically obvious
    // 1/sqrt(D) default would silently override whatever the frontend's omission meant,
    // so absence is declined rather than guessed.
    GraphSpec spec;
    spec.attnScaleValue = std::nullopt;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Tier 2 -- faults and malformed graphs. graph_match runs on an UNVALIDATED graph.
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesNonDivisibleGqaGrouping)
{
    // The kernel derives gqa = Hq // Hkv by integer division, silently dropping the
    // remainder heads.
    GraphSpec spec;
    spec.numQueryHeads = 6;
    spec.numKvHeads = 4;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesGraphWithNoStrides)
{
    // strides() returns nullptr. Every layout predicate dereferences it, so this must
    // decline rather than crash.
    GraphSpec spec;
    spec.omitStrides = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesMultiNodeGraph)
{
    GraphSpec spec;
    spec.twoNodes = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesBothDeprecatedCausalBooleans)
{
    // Mutually exclusive. The frontend owns the diagnostic; graph_match must be total
    // over an unvalidated graph, so it declines rather than throws.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Tier 3 -- declined features. An UNCHECKED optional attribute is accepted and then
// silently not performed, which is a wrong answer with no error.
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAdditiveAttentionMask)
{
    GraphSpec spec;
    spec.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesDeviceResidentScaleTensor)
{
    // The ABI's scale is a launch scalar; there is no pointer slot for a scale tensor.
    // Note this is the SAME frontend setter name as attn_scale_value -- set_attn_scale
    // is overloaded on shared_ptr<TensorAttributes> and float -- so a matcher checking
    // only the scalar admits graphs carrying the tensor.
    GraphSpec spec;
    spec.scaleTensorUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesVarlen)
{
    GraphSpec spec;
    spec.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesPagedKv)
{
    // Worth its own case: supports_attention_dense never inspects `spec.paged` at all,
    // so the graph side is the only place a paged request can be caught.
    GraphSpec spec;
    spec.pageTableKUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAttentionSinks)
{
    GraphSpec spec;
    spec.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesBlockSparseMask)
{
    GraphSpec spec;
    spec.blockMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSoftmaxStatsBothSpellings)
{
    // Two spellings of one feature; rejecting only the UID admits graphs carrying the
    // boolean, and vice versa.
    GraphSpec byUid;
    byUid.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(byUid).has_value());

    GraphSpec byFlag;
    byFlag.generateStats = true;
    EXPECT_FALSE(matchGraph(byFlag).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, AcceptsExplicitlyDisabledStats)
{
    // generate_stats is optional<bool>: an explicit `false` is a request for NO stats,
    // which this kernel satisfies. Declining it would be an over-rejection.
    GraphSpec spec;
    spec.generateStats = false;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesDropout)
{
    GraphSpec spec;
    spec.dropoutProbability = 0.1F;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesFp8Descale)
{
    GraphSpec spec;
    spec.descaleQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAlibiMask)
{
    GraphSpec spec;
    spec.alibiMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesPaddingMask)
{
    GraphSpec spec;
    spec.paddingMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSlidingWindow)
{
    // A bounded left edge that is not the causal (-1, 0) pair derives to
    // SLIDING_WINDOW, which supports_attention_dense rejects unconditionally.
    GraphSpec spec;
    spec.leftBound = 64;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSlidingWindowSetAlongsideDeprecatedCausal)
{
    // A REAL graph shape, and the one the sibling test above misses: gpt_oss sets the
    // deprecated causal_mask boolean AND a 128-token left bound. Deriving the mask
    // from the boolean first returns TOP_LEFT_CAUSAL and the pack SERVES the graph,
    // silently attending the whole causal triangle instead of the 128-wide band --
    // a wrong answer where a decline was intended. Five graphs in hipDNN's own
    // servable corpus have exactly this shape.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = 128;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSlidingWindowSetAlongsideBottomRightCausal)
{
    // Same hazard on the bottom-right spelling of the deprecated pair.
    GraphSpec spec;
    spec.causalMaskBottomRightDeprecated = true;
    spec.leftBound = 128;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, StillServesPlainDeprecatedCausalWithNoBound)
{
    // The guard above must not over-decline: a deprecated causal flag with NO bound is
    // ordinary causal and stays servable. 152 graphs in the corpus are this shape.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesNonAutoImplementationHint)
{
    // The one field the completeness check flags as still UNCHECKED in the shipped
    // reference pack. A named execution strategy is a request this pack does not
    // implement, and ignoring it is not the same as honouring it.
    GraphSpec spec;
    spec.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesUnsupportedMmaCoreMode)
{
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// kernel_match -- does THIS candidate's baked metadata fit the graph?
//
// Every shape field is compiled into the emitted binary, so each of these is a
// silent-wrong-answer guard: the K/V buffer-resource extent is sized from `batch` at
// build time, the KV-loop trip count is `seqlen_kv // block_n`, and the grid comes from
// `seqlen_q`/`num_query_heads`/`batch`. A candidate that disagrees on any of them would
// be served zero-fill or a truncated loop rather than faulting.
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseKernelMatch, AcceptsTheCandidateBakedForThisGraph)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherDtype)
{
    KernelSpec kernel;
    kernel.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherHeadSize)
{
    KernelSpec kernel;
    kernel.headSize = 64;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherBatch)
{
    // The defect class this check exists for: `batch` is baked into the K/V
    // buffer-resource extent (attention_dense.py:1264-1265), so a graph with a larger
    // batch than the variant was compiled for reads zero-fill past the bound -- in
    // bounds, no fault, wrong numbers.
    KernelSpec kernel;
    kernel.batch = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherSequenceLength)
{
    KernelSpec shorterQ;
    shorterQ.seqLenQ = 512;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, shorterQ));

    KernelSpec shorterKv;
    shorterKv.seqLenKv = 512;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, shorterKv));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherHeadCount)
{
    KernelSpec queryHeads;
    queryHeads.numQueryHeads = 8;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, queryHeads));

    KernelSpec kvHeads;
    kvHeads.numKvHeads = 2;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kvHeads));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateBakedForTheOtherMask)
{
    // The graph's mask is a DERIVATION (left_bound/right_bound/diagonal_alignment or
    // the deprecated booleans); the candidate bakes a plain 0/1. A mismatch here serves
    // a non-causal kernel for a causal graph: a valid-looking wrong answer.
    KernelSpec nonCausal;
    nonCausal.causal = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, nonCausal));

    GraphSpec noMaskGraph;
    noMaskGraph.leftBound = std::nullopt;
    noMaskGraph.rightBound = std::nullopt;
    EXPECT_FALSE(matchesKernel(noMaskGraph, KernelSpec{}));
    EXPECT_TRUE(matchesKernel(noMaskGraph, nonCausal));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesATileThatDoesNotDivideTheSequence)
{
    // The knob-selection rows. block_n must divide seqlen_kv and block_m must divide
    // seqlen_q -- tested against the CANDIDATE rather than a constant, so a different
    // tile can serve what this one cannot.
    KernelSpec badBlockN;
    badBlockN.blockN = 48; // 256 % 48 != 0
    EXPECT_FALSE(matchesKernel(GraphSpec{}, badBlockN));

    KernelSpec badBlockM;
    badBlockM.blockM = 384; // 256 % 384 != 0
    EXPECT_FALSE(matchesKernel(GraphSpec{}, badBlockM));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesANonPositiveTile)
{
    KernelSpec zeroBlockN;
    zeroBlockN.blockN = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, zeroBlockN));

    KernelSpec zeroBlockM;
    zeroBlockM.blockM = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, zeroBlockM));
}

TEST(TestGfx942AttentionDenseKernelMatch, AcceptsEitherShippedTileForA256KeyGraph)
{
    // Both shipped block_n values divide 256, so both are applicable and `score`
    // decides. This is what makes the ranking test below meaningful rather than
    // vacuous -- and it is why block_n=32 is never SELECTED at these shapes.
    KernelSpec wide;
    wide.blockN = 64;
    KernelSpec narrow;
    narrow.blockN = 32;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, wide));
    EXPECT_TRUE(matchesKernel(GraphSpec{}, narrow));
}

// ---------------------------------------------------------------------------
// score -- ranks the survivors. Higher wins.
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseScore, RanksTheWiderKvTileHigher)
{
    // A larger KV tile amortises the per-tile barrier, LDS publication and loop
    // overhead across more keys. This is the engine's ONLY free axis once
    // kernel_match has pinned every shape field.
    KernelSpec wide;
    wide.blockN = 64;
    KernelSpec narrow;
    narrow.blockN = 32;
    EXPECT_GT(scoreOf(wide), scoreOf(narrow));
}

TEST(TestGfx942AttentionDenseScore, DoesNotReturnAConstant)
{
    // A constant score makes ranking arbitrary and hides mis-specialization, so it is
    // not a legitimate placeholder however honestly it is disclosed. This test is the
    // mechanical guard on that.
    KernelSpec wide;
    wide.blockN = 64;
    KernelSpec narrow;
    narrow.blockN = 32;
    EXPECT_NE(scoreOf(wide), scoreOf(narrow));
}

TEST(TestGfx942AttentionDenseScore, IgnoresAxesThatKernelMatchAlreadyPinned)
{
    // dtype/head_size/shape are decided by kernel_match, so score must not also rank on
    // them -- two candidates differing only there would otherwise be ordered by an axis
    // that carries no performance meaning.
    const KernelSpec bf16;
    KernelSpec fp16;
    fp16.dtype = "FP16";
    EXPECT_EQ(scoreOf(bf16), scoreOf(fp16));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"

/**
 * @file TestAttentionDenseMatchers.cpp
 * @brief The graph and kernel match rules the gfx942 and gfx950 dense packs share, run
 *        on both arches.
 *
 * Both packs bind the same graphMatches/kernelMatches (AttentionDenseNative.cpp), so a
 * rule that holds on one must hold on the other. The per-arch suites
 * (TestGfx94xAttentionDenseMatchers.cpp) keep what differs: masks, stats, tiles, ranking.
 * Each decline here is checked by the cause the matcher logs, so a graph declined for an
 * earlier, unrelated reason cannot pass for the rule under test.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 256;
constexpr int64_t HEAD_SIZE = 128;

/// The kernel's 32-bit offset limit: K/V bytes and Q/O elements must stay below it.
constexpr int64_t OFFSET_LIMIT = int64_t{1} << 31;

using Dims = std::vector<int64_t>;

/// BSHD strides for (B, H, S, D) dims; packed row-major for any other rank.
std::vector<int64_t> bshdStrides(const Dims& dims)
{
    if(dims.size() == 4)
    {
        return {dims[1] * dims[2] * dims[3], dims[3], dims[1] * dims[3], 1};
    }
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t axis = dims.size(); axis > 1; --axis)
    {
        strides[axis - 2] = strides[axis - 1] * dims[axis - 1];
    }
    return strides;
}

/// BHSD strides for (B, H, S, D) dims: head-major, which the kernel cannot read.
std::vector<int64_t> bhsdStrides(const Dims& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

/// One operand: dims, and strides written out or derived as dense BSHD from the dims. A
/// ternary rather than value_or, since an overriding operand's derived strides may be a
/// product that does not fit.
struct Operand
{
    Dims dims;
    std::optional<std::vector<int64_t>> strides;
    std::optional<data_objects::DataType> dataType;

    std::vector<int64_t> stridesOrBshd() const
    {
        return strides.has_value() ? *strides : bshdStrides(dims);
    }
};

/// A top-left causal BF16 SDPA graph both arches serve, with each operand overridable.
struct GraphSpec
{
    int64_t batch = BATCH;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t headSize = HEAD_SIZE;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;

    /// Q, K, V and O by uid - 1; an unset operand takes its default shape.
    std::array<std::optional<Operand>, 4> operands;

    /// @p uid's operand as the graph carries it.
    Operand resolved(int64_t uid) const
    {
        const auto& slot = operands.at(static_cast<size_t>(uid - Q_UID));
        if(slot.has_value())
        {
            return *slot;
        }
        if(uid == K_UID || uid == V_UID)
        {
            return {{batch, numKvHeads, seqLenKv, headSize}, std::nullopt, std::nullopt};
        }
        return {{batch, numQueryHeads, seqLenQ, headSize}, std::nullopt, std::nullopt};
    }

    /// @p uid's operand, materialised so a case can override one field of it.
    Operand& operand(int64_t uid)
    {
        auto& slot = operands.at(static_cast<size_t>(uid - Q_UID));
        if(!slot.has_value())
        {
            slot = resolved(uid);
        }
        return *slot;
    }
};

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        const Operand operand = spec.resolved(uid);
        const auto strides = operand.stridesOrBshd();
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       uid,
                                                       nullptr,
                                                       operand.dataType.value_or(spec.dataType),
                                                       &strides,
                                                       &operand.dims,
                                                       false));
    }

    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(Q_UID);
    attributesBuilder.add_k_tensor_uid(K_UID);
    attributesBuilder.add_v_tensor_uid(V_UID);
    attributesBuilder.add_o_tensor_uid(O_UID);
    attributesBuilder.add_left_bound(-1);
    attributesBuilder.add_right_bound(0);
    attributesBuilder.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
    attributesBuilder.add_causal_mask(false);
    attributesBuilder.add_causal_mask_bottom_right(false);
    attributesBuilder.add_attn_scale_value(0.125F);
    attributesBuilder.add_alibi_mask(false);
    attributesBuilder.add_padding_mask(false);
    attributesBuilder.add_mma_core_mode(data_objects::DataType::UNSET);
    attributesBuilder.add_implementation(data_objects::AttentionImplementation::AUTO);
    const auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   spec.computeType,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributes.Union()));

    auto name = builder.CreateString("attention_dense_shared_rules");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// A built graph and the MatchContext @p arch's engine sees over it.
class Graph
{
public:
    Graph(const AttentionDenseTestArch& arch, const GraphSpec& spec)
        : _builder(buildGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = std::string(arch.gcnArchName);
        _properties.warpSize = 64;
        registerNativeIngestorSymbols();
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

std::optional<BoundTokens> matchGraph(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    const Graph graph(arch, spec);
    return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(arch.graphMatchSymbol))(graph.context());
}

struct LoggedDecline
{
    std::string cause;
    std::string detail;
};

/// The decline @p arch's engine logged at INFO for @p spec, or an empty cause when it
/// logged none or served the graph.
LoggedDecline loggedDecline(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    const std::string marker = std::string(arch.engineName) + " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(matchGraph(arch, spec).has_value());
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        const auto causeEnd
            = at == std::string::npos ? at : log.message.find(']', at + marker.size());
        if(log.severity == HIPDNN_SEV_INFO && causeEnd != std::string::npos)
        {
            return {log.message.substr(at + marker.size(), causeEnd - at - marker.size()),
                    log.message.substr(causeEnd + 1)};
        }
    }
    return {};
}

/// A D128 bm256/bn64 BF16 causal record for @p spec, completed as @p arch's KMD defaults
/// complete it. gfx950 compares the baked head counts, so they are the graph's.
KernelDefinition makeKernel(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000d5a1");
    kernel.name = std::string(arch.kernelName);
    kernel.metadata = completedAttentionDenseMetadata(arch, spec.headSize, 256, 64);
    kernel.metadata[std::string("dtype")] = std::string("BF16");
    kernel.metadata[std::string("num_query_heads")] = spec.numQueryHeads;
    kernel.metadata[std::string("num_kv_heads")] = spec.numKvHeads;
    kernel.metadata[std::string("seqlen_q")] = spec.seqLenQ;
    kernel.metadata[std::string("seqlen_kv")] = spec.seqLenKv;
    kernel.metadata[std::string("batch")] = spec.batch;
    kernel.metadata[std::string("causal")] = int64_t{1};
    kernel.metadata[std::string("sliding_window")] = int64_t{0};
    kernel.metadata[std::string("ragged")] = int64_t{0};
    if(arch.completesGridOrder)
    {
        // gfx942's body knobs, at the bf16 D128 policy.
        kernel.metadata[std::string("lds_row_pad")] = int64_t{8};
        kernel.metadata[std::string("pv_priority")] = int64_t{0};
        kernel.metadata[std::string("emit_lse")] = int64_t{0};
        kernel.metadata[std::string("lds_num_buffers")] = int64_t{1};
        kernel.metadata[std::string("v_reg_prefetch")] = int64_t{0};
        kernel.metadata[std::string("causal_diag_split")] = int64_t{0};
        kernel.metadata[std::string("o_store_width")] = int64_t{4};
        kernel.metadata[std::string("pv_loop_order")] = std::string("d_major");
        kernel.metadata[std::string("use_cfvst")] = int64_t{1};
        kernel.metadata[std::string("use_exp2_fast")] = int64_t{0};
        kernel.metadata[std::string("lds_k_group_pad")] = int64_t{8};
    }
    return kernel;
}

/// graph_match then kernel_match, as a catalog build runs them.
bool matchesKernel(const AttentionDenseTestArch& arch,
                   const GraphSpec& spec,
                   const KernelDefinition& kernel)
{
    const Graph graph(arch, spec);
    const auto bound = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(arch.graphMatchSymbol))(graph.context());
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(arch.kernelMatchSymbol))(graph.context(), *bound, kernel);
}

class TestAttentionDenseMatch : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

/// Every negative below perturbs this graph, so it must be served end to end.
TEST_P(TestAttentionDenseMatch, ServesTheBaseGraphAndRecord)
{
    const GraphSpec spec;
    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernel(arch(), spec)));
}

// =============================================================================
// The node's compute data type. The kernel accumulates in FLOAT, which meets FLOAT, HALF
// and BFLOAT16; UNSET leaves the choice to the provider.
// =============================================================================

TEST_P(TestAttentionDenseMatch, ServesTheComputeTypesItsFloatAccumulationMeets)
{
    for(const auto computeType : {data_objects::DataType::UNSET,
                                  data_objects::DataType::FLOAT,
                                  data_objects::DataType::HALF,
                                  data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        spec.computeType = computeType;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value())
            << data_objects::EnumNameDataType(computeType);
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesEveryOtherComputeDataType)
{
    for(const auto computeType : data_objects::EnumValuesDataType())
    {
        if(computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::HALF
           || computeType == data_objects::DataType::BFLOAT16)
        {
            continue;
        }
        SCOPED_TRACE(data_objects::EnumNameDataType(computeType));
        GraphSpec spec;
        spec.computeType = computeType;
        const auto decline = loggedDecline(arch(), spec);
        EXPECT_EQ(decline.cause, "compute_data_type") << decline.detail;
        EXPECT_NE(decline.detail.find("accumulates in FLOAT"), std::string::npos) << decline.detail;
    }
}

// =============================================================================
// kv_segments. A split-KV build launches kv_segments times the grid plus a combine the
// dispatch does not make, so only a single segment is served. Neither KMD declares the
// field, so a record without it is a single-segment build.
// =============================================================================

TEST_P(TestAttentionDenseMatch, ServesOnlyASingleKvSegment)
{
    const GraphSpec spec;

    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernel(arch(), spec))) << "kv_segments absent";

    auto single = makeKernel(arch(), spec);
    single.metadata[std::string("kv_segments")] = int64_t{1};
    EXPECT_TRUE(matchesKernel(arch(), spec, single)) << "kv_segments 1";

    for(const int64_t segments : {int64_t{2}, int64_t{4}, int64_t{0}, int64_t{-1}})
    {
        auto split = makeKernel(arch(), spec);
        split.metadata[std::string("kv_segments")] = segments;
        EXPECT_FALSE(matchesKernel(arch(), spec, split)) << "kv_segments " << segments;
    }

    auto mistyped = makeKernel(arch(), spec);
    mistyped.metadata[std::string("kv_segments")] = std::string("1");
    EXPECT_FALSE(matchesKernel(arch(), spec, mistyped)) << "kv_segments as a string";
}

// =============================================================================
// 32-bit addressing. K/V is bounded in bytes, Q/O in elements; O must have Q's dims, so
// the Q bound is the O bound. One head, so the sequence length carries the whole count.
// =============================================================================

TEST_P(TestAttentionDenseMatch, QueryElementBoundAdmitsTheLastGraphUnderIt)
{
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("head_size " + std::to_string(headSize));
        GraphSpec spec;
        spec.batch = 1;
        spec.numQueryHeads = 1;
        spec.numKvHeads = 1;
        spec.headSize = headSize;

        spec.seqLenQ = OFFSET_LIMIT / headSize - 1;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "2^31 - head_size elements";

        spec.seqLenQ = OFFSET_LIMIT / headSize;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit") << "exactly 2^31 elements";
    }
}

TEST_P(TestAttentionDenseMatch, KeyValueByteBoundAdmitsTheLastGraphUnderIt)
{
    // Two bytes an element, so the bound falls at half the Q/O sequence length: that pins
    // the bytes factor.
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("head_size " + std::to_string(headSize));
        GraphSpec spec;
        spec.batch = 1;
        spec.numQueryHeads = 1;
        spec.numKvHeads = 1;
        spec.headSize = headSize;

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2) - 1;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "2^31 - 2 * head_size bytes";

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2);
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit") << "exactly 2^31 bytes";
    }
}

TEST_P(TestAttentionDenseMatch, ACandidateServesTheLargestTiledGraphUnderEachBound)
{
    // The bounds, not the bm256/bn64 tile, are what decide at these sizes: the largest
    // tile-aligned Q and K/V under them are served end to end.
    GraphSpec query;
    query.batch = 1;
    query.numQueryHeads = 1;
    query.numKvHeads = 1;
    query.seqLenQ = OFFSET_LIMIT / HEAD_SIZE - 256;
    EXPECT_TRUE(matchesKernel(arch(), query, makeKernel(arch(), query)));

    GraphSpec keyValue = query;
    keyValue.seqLenQ = SEQ;
    keyValue.seqLenKv = OFFSET_LIMIT / (HEAD_SIZE * 2) - 64;
    EXPECT_TRUE(matchesKernel(arch(), keyValue, makeKernel(arch(), keyValue)));
}

// Products past int64_t. An unchecked product would wrap to INT64_MIN, under any bound,
// so each graph is declined only because checkedProduct reports the overflow. Each
// operand's own strides fit, so the layout check passes first.

TEST_P(TestAttentionDenseMatch, DeclinesAKeyValueByteCountPastInt64)
{
    // B 2^24, Skv 2^32, D64: K/V is 2^63 bytes; Q is 2^30 elements.
    GraphSpec spec;
    spec.batch = int64_t{1} << 24;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = int64_t{1} << 32;
    spec.headSize = 64;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit");
}

TEST_P(TestAttentionDenseMatch, DeclinesAQueryElementCountPastInt64)
{
    // B 2^23, Sq 2^34, D64: Q is 2^63 elements; K/V is 2^30 bytes.
    GraphSpec spec;
    spec.batch = int64_t{1} << 23;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.seqLenQ = int64_t{1} << 34;
    spec.seqLenKv = 1;
    spec.headSize = 64;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit");
}

// =============================================================================
// Per-operand rules. The kernel takes one shape and no strides, so each operand must
// agree with the others and be dense BSHD; one bad operand declines the graph.
// =============================================================================

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandThatDisagreesWithTheShape)
{
    struct Case
    {
        const char* name;
        int64_t uid;
        const char* role;
        Dims dims;
    };
    const std::vector<Case> cases{
        {"V batch", V_UID, "V", {3, HEADS, SEQ, HEAD_SIZE}},
        {"V heads", V_UID, "V", {BATCH, 2, SEQ, HEAD_SIZE}},
        {"V sequence", V_UID, "V", {BATCH, HEADS, 128, HEAD_SIZE}},
        {"V head size", V_UID, "V", {BATCH, HEADS, SEQ, 64}},
        {"K batch", K_UID, "K", {3, HEADS, SEQ, HEAD_SIZE}},
        {"K head size", K_UID, "K", {BATCH, HEADS, SEQ, 64}},
        {"O batch", O_UID, "O", {3, HEADS, SEQ, HEAD_SIZE}},
        {"O heads", O_UID, "O", {BATCH, 2, SEQ, HEAD_SIZE}},
        {"O sequence", O_UID, "O", {BATCH, HEADS, 128, HEAD_SIZE}},
        {"O head size", O_UID, "O", {BATCH, HEADS, SEQ, 64}},
    };
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        GraphSpec spec;
        spec.operand(testCase.uid).dims = testCase.dims;
        const auto decline = loggedDecline(arch(), spec);
        EXPECT_EQ(decline.cause, "shape") << decline.detail;
        // The line names the operand that broke the rule first.
        EXPECT_EQ(decline.detail.rfind(std::string(": ") + testCase.role + " (uid ", 0), 0U)
            << decline.detail;
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesKeyHeadsThatDoNotDivideTheQueryHeads)
{
    GraphSpec spec;
    spec.numKvHeads = 3;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "gqa");
}

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandThatIsNotDenseBshd)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        GraphSpec spec;
        auto& operand = spec.operand(uid);
        operand.strides = bhsdStrides(operand.dims);
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "layout");

        // One padded row: dense in every other axis.
        GraphSpec padded;
        auto& paddedOperand = padded.operand(uid);
        auto strides = bshdStrides(paddedOperand.dims);
        strides[0] += 8;
        paddedOperand.strides = strides;
        EXPECT_EQ(loggedDecline(arch(), padded).cause, "layout");
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandOfAnotherRankOrAZeroExtent)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        GraphSpec rank3;
        auto& flattened = rank3.operand(uid);
        flattened.dims
            = {flattened.dims[0], flattened.dims[2], flattened.dims[1] * flattened.dims[3]};
        EXPECT_EQ(loggedDecline(arch(), rank3).cause, "operand");

        for(size_t axis = 0; axis < 4; ++axis)
        {
            GraphSpec empty;
            empty.operand(uid).dims[axis] = 0;
            EXPECT_EQ(loggedDecline(arch(), empty).cause, "operand") << "axis " << axis;
        }
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesMixedOperandDataTypes)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        GraphSpec spec;
        spec.operand(uid).dataType = data_objects::DataType::HALF;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "data_type");
    }

    // One type throughout, but not one the kernel is built for.
    GraphSpec fp32;
    fp32.dataType = data_objects::DataType::FLOAT;
    EXPECT_EQ(loggedDecline(arch(), fp32).cause, "data_type");

    GraphSpec fp16;
    fp16.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(matchGraph(arch(), fp16).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseMatch,
                         ::testing::Values(&GFX942_ATTENTION_DENSE, &GFX950_ATTENTION_DENSE),
                         attentionDenseArchName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

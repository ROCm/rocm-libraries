// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"

/**
 * The shared FlyDSL SDPA backward matcher, one test per rule. A test's name begins with its
 * rule's id with the hyphen dropped (MW-9a is MW9a); FlydslSdpaBwdCommon.hpp documents each
 * rule id beside its check. A decline test asserts the cause and a substring of the detail,
 * and sits beside an accepting control that differs from it in one field. An accept test
 * asserts what the problem records.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using flydsl_sdpa_bwd::Decline;
using flydsl_sdpa_bwd::NO_UID;
using flydsl_sdpa_bwd::NO_WINDOW;
using flydsl_sdpa_bwd::ScaleSource;
using flydsl_sdpa_bwd::SdpaBwdProblem;
using flydsl_sdpa_bwd::SeedForm;
using flydsl_sdpa_bwd::SeedWidth;
using flydsl_sdpa_bwd::StatsLayout;
using flydsl_sdpa_bwd::Strides;

constexpr float TWO_POW_20 = 1048576.0F;

/// The problem the matcher computed, or a failure carrying its decline.
std::optional<SdpaBwdProblem> accepted(const SdpaBwdSpec& spec)
{
    const auto result = matchSdpaBwdSpec(spec);
    if(const auto* decline = std::get_if<Decline>(&result))
    {
        ADD_FAILURE() << "declined [" << decline->cause << "]: " << decline->detail;
        return std::nullopt;
    }
    return std::get<SdpaBwdProblem>(result);
}

void declined(const SdpaBwdSpec& spec, std::string_view cause, std::string_view detail)
{
    const auto result = matchSdpaBwdSpec(spec);
    const auto* decline = std::get_if<Decline>(&result);
    if(decline == nullptr)
    {
        ADD_FAILURE() << "accepted; expected a decline [" << cause << "] naming '" << detail << "'";
        return;
    }
    EXPECT_EQ(decline->cause, cause) << decline->detail;
    EXPECT_NE(decline->detail.find(detail), std::string::npos)
        << "detail '" << decline->detail << "' does not name '" << detail << "'";
}

void expectStrides(const Strides& actual, int64_t batch, int64_t head, int64_t seq)
{
    EXPECT_EQ(actual.batch, batch);
    EXPECT_EQ(actual.head, head);
    EXPECT_EQ(actual.seq, seq);
}

SdpaBwdSpec baseline()
{
    return SdpaBwdSpec{};
}

struct NamedOperand
{
    const char* tensor;
    const char* label;
};

/// The eight data tensors, then the statistics.
constexpr std::array<NamedOperand, 9> OPERANDS{{{"q", "Q"},
                                                {"k", "K"},
                                                {"v", "V"},
                                                {"o", "O"},
                                                {"do", "dO"},
                                                {"dq", "dQ"},
                                                {"dk", "dK"},
                                                {"dv", "dV"},
                                                {"stats", "stats"}}};

// ---------------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, Baseline_TestSdkGraphIsAccepted)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph();
    const auto result = matchSdpaBwdBuffer(builder);
    const auto* problem = std::get_if<SdpaBwdProblem>(&result);
    ASSERT_NE(problem, nullptr) << std::get<Decline>(result).cause << ": "
                                << std::get<Decline>(result).detail;
    EXPECT_EQ(problem->dtype, data_objects::DataType::HALF);
    EXPECT_EQ(problem->batch, 2);
    EXPECT_EQ(problem->numHeads, 8);
    EXPECT_EQ(problem->numKvHeads, 8);
    EXPECT_EQ(problem->seqLenQ, 16);
    EXPECT_EQ(problem->seqLenKv, 16);
    EXPECT_EQ(problem->headDimQk, 64);
    EXPECT_EQ(problem->headDimV, 64);
    expectStrides(problem->q, 8192, 1024, 64);
    expectStrides(problem->dV, 8192, 1024, 64);
    EXPECT_EQ(problem->scaleSource, ScaleSource::VALUE);
    EXPECT_EQ(problem->scaleValue, 1.0F);
    EXPECT_FALSE(problem->masked);
    EXPECT_FALSE(problem->paddingMask);
    EXPECT_EQ(problem->biasUid, NO_UID);
    EXPECT_EQ(problem->dropoutP, 0.0F);
    EXPECT_EQ(problem->statsLayout, StatsLayout::HT);
}

TEST(TestFlydslSdpaBwdMatcher, Baseline_DefaultSpecMatchesTheTestSdkGraph)
{
    const auto problem = accepted(baseline());
    ASSERT_TRUE(problem.has_value());
    const auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph();
    const auto sdkResult = matchSdpaBwdBuffer(builder);
    const auto* sdkProblem = std::get_if<SdpaBwdProblem>(&sdkResult);
    ASSERT_NE(sdkProblem, nullptr);
    EXPECT_EQ(problem->dtype, sdkProblem->dtype);
    EXPECT_EQ(problem->batch, sdkProblem->batch);
    EXPECT_EQ(problem->seqLenQ, sdkProblem->seqLenQ);
    EXPECT_EQ(problem->q.seq, sdkProblem->q.seq);
    EXPECT_EQ(problem->statsLayout, sdkProblem->statsLayout);
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, ST1_NodeCountOrKindOtherThanOneBackwardNodeDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.nodeCount = 2; }),
             "node",
             "2 nodes, expected 1");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.nodeKind = SdpaBwdNodeKind::FORWARD; }),
             "node",
             "not an SDPA backward node");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.nodeCount = 1; })));
}

TEST(TestFlydslSdpaBwdMatcher, ST2_OverrideShapeGraphDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.overrideShape = true; }),
             "node",
             "override shapes");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.overrideShape = false; })));
}

TEST(TestFlydslSdpaBwdMatcher, ST3_ComputeTypeOutsideUnsetFloatHalfBf16Declines)
{
    for(const auto type : {data_objects::DataType::UNSET,
                           data_objects::DataType::FLOAT,
                           data_objects::DataType::HALF,
                           data_objects::DataType::BFLOAT16})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(type));
        EXPECT_TRUE(
            accepted(baseline().withEdit([type](SdpaBwdGraph& g) { g.computeType = type; })));
    }
    for(const auto type : {data_objects::DataType::DOUBLE,
                           data_objects::DataType::INT32,
                           data_objects::DataType::FP8_E4M3})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(type));
        declined(baseline().withEdit([type](SdpaBwdGraph& g) { g.computeType = type; }),
                 "node",
                 "compute type is " + std::string(data_objects::EnumNameDataType(type)));
    }
}

TEST(TestFlydslSdpaBwdMatcher, ST4_MissingVirtualOrPassByValueOperandDeclines)
{
    for(const auto& operand : OPERANDS)
    {
        SCOPED_TRACE(operand.tensor);
        const std::string tensor = operand.tensor;
        const std::string label = std::string(operand.label) + " (uid ";
        declined(
            baseline().withEdit([tensor](SdpaBwdGraph& g) { g.tensor(tensor).isVirtual = true; }),
            "operand",
            label);
        declined(baseline().withEdit(
                     [tensor](SdpaBwdGraph& g) { g.tensor(tensor).runtimePassByValue = true; }),
                 "operand",
                 "is pass-by-value");
        declined(baseline().withEdit([tensor](SdpaBwdGraph& g) { g.tensor(tensor).value = 1.0F; }),
                 "operand",
                 "is pass-by-value");
    }
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.q = 999; }),
             "operand",
             "Q (uid 999) is not in the graph");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.stats = 999; }),
             "operand",
             "stats (uid 999) is not in the graph");
    EXPECT_TRUE(accepted(baseline()));
}

TEST(TestFlydslSdpaBwdMatcher, ST5_OutputUidSharedWithAnotherRoleDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.dQ = sdpa_bwd_uid::Q; }),
             "operand",
             "dQ (uid 1) shares its uid with Q");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.dV = sdpa_bwd_uid::DK; }),
             "operand",
             "dK (uid 8) shares its uid with dV");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
        g.attributes.dbiasUid = sdpa_bwd_uid::BIAS;
    }),
             "operand",
             "dBias (uid 11) shares its uid with bias");
    // Inputs may share a uid among themselves.
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.k = sdpa_bwd_uid::Q;
        g.attributes.v = sdpa_bwd_uid::Q;
    })));
}

// ---------------------------------------------------------------------------
// Dtype and shape
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, DT1_F16OrBf16OnEveryTensorAcceptsAndMixedDeclines)
{
    for(const auto type : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(type));
        SdpaBwdSpec spec;
        spec.dtype = type;
        const auto problem = accepted(spec);
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->dtype, type);
    }
    SdpaBwdSpec bf16;
    bf16.dtype = data_objects::DataType::BFLOAT16;
    declined(
        bf16.withEdit([](SdpaBwdGraph& g) { g.tensor("v").dtype = data_objects::DataType::HALF; }),
        "data_type",
        "V (uid 3) is HALF, but Q is BFLOAT16");
    declined(baseline().withEdit(
                 [](SdpaBwdGraph& g) { g.tensor("dk").dtype = data_objects::DataType::BFLOAT16; }),
             "data_type",
             "dK (uid 8) is BFLOAT16, but Q is HALF");
}

TEST(TestFlydslSdpaBwdMatcher, DT2_Fp32Declines)
{
    SdpaBwdSpec fp32;
    fp32.dtype = data_objects::DataType::FLOAT;
    declined(fp32, "data_type", "Q (uid 1) is FLOAT, not f16/bf16");
    EXPECT_TRUE(accepted(baseline()));
}

TEST(TestFlydslSdpaBwdMatcher, DT3_Fp8Declines)
{
    SdpaBwdSpec fp8;
    fp8.dtype = data_objects::DataType::FP8_E4M3;
    declined(fp8, "data_type", "FP8_E4M3");
    EXPECT_TRUE(accepted(baseline()));
}

TEST(TestFlydslSdpaBwdMatcher, DT4_HeadDimAbove512Declines)
{
    SdpaBwdSpec wideQk;
    wideQk.headDimQk = 520;
    declined(wideQk, "head_size", "exceed 512");
    SdpaBwdSpec wideV;
    wideV.headDimV = 520;
    declined(wideV, "head_size", "exceed 512");
    SdpaBwdSpec limit;
    limit.headDimQk = 512;
    limit.headDimV = 512;
    EXPECT_TRUE(accepted(limit));
}

TEST(TestFlydslSdpaBwdMatcher, DT5_UnequalQkAndVHeadDimsAccept)
{
    SdpaBwdSpec spec;
    spec.headDimQk = 128;
    spec.headDimV = 64;
    auto problem = accepted(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->headDimQk, 128);
    EXPECT_EQ(problem->headDimV, 64);
    expectStrides(problem->q, 16384, 2048, 128);
    expectStrides(problem->v, 8192, 1024, 64);

    spec.headDimQk = 64;
    spec.headDimV = 128;
    problem = accepted(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->headDimQk, 64);
    EXPECT_EQ(problem->headDimV, 128);
}

TEST(TestFlydslSdpaBwdMatcher, DT6_GqaAndMqaAcceptAndNonMultipleHeadCountsDecline)
{
    SdpaBwdSpec ragged;
    ragged.heads = 8;
    ragged.kvHeads = 3;
    declined(ragged, "gqa", "key/value heads 3");
    SdpaBwdSpec moreKv;
    moreKv.heads = 4;
    moreKv.kvHeads = 8;
    declined(moreKv, "gqa", "key/value heads 8");

    const std::vector<std::pair<int64_t, int64_t>> served{{8, 2}, {8, 1}, {9, 1}, {16, 1}, {18, 2}};
    for(const auto& [heads, kvHeads] : served)
    {
        SCOPED_TRACE(std::to_string(heads) + " query heads, " + std::to_string(kvHeads) + " kv");
        SdpaBwdSpec spec;
        spec.heads = heads;
        spec.kvHeads = kvHeads;
        const auto problem = accepted(spec);
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->numHeads, heads);
        EXPECT_EQ(problem->numKvHeads, kvHeads);
    }
}

TEST(TestFlydslSdpaBwdMatcher, DT7_SequenceAbove8192AndTensorAbove4GiBAccept)
{
    SdpaBwdSpec longSequence;
    longSequence.seqQ = 16384;
    longSequence.seqKv = 16384;
    auto problem = accepted(longSequence);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->seqLenQ, 16384);
    EXPECT_EQ(problem->seqLenKv, 16384);

    // 512 * 8 * 16384 * 64 elements of 2 bytes is 8 GiB per tensor; each slab is small.
    SdpaBwdSpec huge = longSequence;
    huge.batch = 512;
    problem = accepted(huge);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->batch, 512);
}

TEST(TestFlydslSdpaBwdMatcher, DT8_HeadDimNotAMultipleOf8NeedsRowStrideOfThePaddedWidth)
{
    for(const int64_t headDim : {7, 12, 23, 37})
    {
        SCOPED_TRACE(headDim);
        SdpaBwdSpec padded;
        padded.headDimQk = headDim;
        padded.headDimV = headDim;
        const auto problem = accepted(padded);
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->headDimQk, headDim);
        EXPECT_EQ(problem->q.seq, (headDim + 7) / 8 * 8);
    }
    SdpaBwdSpec tight;
    tight.headDimQk = 12;
    tight.headDimV = 12;
    declined(tight.withEdit([](SdpaBwdGraph& g) { g.tensor("q").strides = {1024, 128, 8, 1}; }),
             "layout",
             "head dim 12 is not a multiple of 8");
    declined(tight.withEdit([](SdpaBwdGraph& g) { g.tensor("dv").strides = {1024, 128, 8, 1}; }),
             "layout",
             "dV (uid 9) head dim 12");

    // One row of one head has no row stride to carry the padding.
    SdpaBwdSpec singleRow;
    singleRow.batch = 1;
    singleRow.heads = 1;
    singleRow.kvHeads = 1;
    singleRow.seqQ = 1;
    singleRow.seqKv = 1;
    singleRow.headDimQk = 12;
    singleRow.headDimV = 12;
    declined(singleRow, "layout", "smallest row stride does not exist");
    singleRow.seqQ = 2;
    singleRow.seqKv = 2;
    EXPECT_TRUE(accepted(singleRow));
}

TEST(TestFlydslSdpaBwdMatcher, DT9_SlabAbove32BitByteOffsetsDeclines)
{
    const auto withQStride = [](int64_t seqStride) {
        return baseline().withEdit(
            [seqStride](SdpaBwdGraph& g) { g.tensor("q").strides[2] = seqStride; });
    };
    declined(withQStride(134217728), "layout", "32-bit byte offset limit");
    EXPECT_TRUE(accepted(withQStride(1048576)));

    const auto withBiasStride = [](int64_t seqStride, bool dbias) {
        return baseline().withEdit([seqStride, dbias](SdpaBwdGraph& g) {
            addBias(g);
            if(!dbias)
            {
                g.tensor("bias").strides[2] = seqStride;
            }
            addDbias(g);
            if(dbias)
            {
                g.tensor("dbias").strides[2] = seqStride;
            }
        });
    };
    declined(withBiasStride(134217728, false), "layout", "bias (uid 11) per-head slab");
    declined(withBiasStride(134217728, true), "layout", "dBias (uid 12) per-head slab");
    EXPECT_TRUE(accepted(withBiasStride(1048576, false)));
    EXPECT_TRUE(accepted(withBiasStride(1048576, true)));
}

TEST(TestFlydslSdpaBwdMatcher, SH1_ShapeDisagreementDeclines)
{
    struct Case
    {
        const char* tensor;
        std::vector<int64_t> dims;
        const char* detail;
    };
    const std::vector<Case> cases{
        {"dq", {2, 4, 16, 64}, "dQ (uid 7) dims [2,4,16,64] != [2,8,16,64]"},
        {"dk", {2, 8, 8, 64}, "dK (uid 8)"},
        {"dv", {2, 8, 16, 32}, "dV (uid 9)"},
        {"o", {2, 8, 16, 32}, "O (uid 4)"},
        {"do", {2, 8, 8, 64}, "dO (uid 5)"},
        {"v", {2, 4, 16, 64}, "V (uid 3)"},
        {"k", {2, 8, 16, 32}, "K (uid 2)"},
        {"stats", {2, 8, 16, 2}, "stats (uid 6)"},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.tensor);
        const std::string tensor = c.tensor;
        const auto dims = c.dims;
        declined(
            baseline().withEdit([tensor, dims](SdpaBwdGraph& g) { g.tensor(tensor).dims = dims; }),
            "shape",
            c.detail);
    }
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("q").dims[2] = 0; }),
             "shape",
             "non-positive extent");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.tensor("q").dims = {8, 16, 64};
        g.tensor("q").strides = {1024, 64, 1};
    }),
             "shape",
             "not rank 4");
    EXPECT_TRUE(accepted(baseline()));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, LY1_AlignmentBelow16BytesDeclinesOnEveryDataTensorBiasAndDbias)
{
    for(const auto& operand : OPERANDS)
    {
        if(std::string_view(operand.tensor) == "stats")
        {
            continue;
        }
        SCOPED_TRACE(operand.tensor);
        const std::string tensor = operand.tensor;
        declined(baseline().withEdit([tensor](SdpaBwdGraph& g) { g.tensor(tensor).alignment = 8; }),
                 "operand",
                 std::string(operand.label) + " (uid " + "");
        EXPECT_TRUE(accepted(
            baseline().withEdit([tensor](SdpaBwdGraph& g) { g.tensor(tensor).alignment = 32; })));
    }
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("q").alignment = 24; }),
             "operand",
             "declares alignment 24");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g).alignment = 8; }),
             "operand",
             "bias (uid 11) declares alignment 8");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g).alignment = 8;
    }),
             "operand",
             "dBias (uid 12) declares alignment 8");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g).alignment = 16;
        addDbias(g);
    })));
}

TEST(TestFlydslSdpaBwdMatcher, LY2_LastStrideOtherThanOneDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("k").strides[3] = 2; }),
             "layout",
             "K (uid 2) last stride is 2, not 1");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g).strides[3] = 2; }),
             "layout",
             "bias (uid 11) last stride is 2");
    EXPECT_TRUE(
        accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("k").strides[3] = 1; })));
}

TEST(TestFlydslSdpaBwdMatcher, LY3_StrideOfANonUnitAxisNotAMultipleOf8Declines)
{
    const auto withQSeqStride = [](int64_t stride) {
        return baseline().withEdit(
            [stride](SdpaBwdGraph& g) { g.tensor("q").strides[2] = stride; });
    };
    declined(withQSeqStride(68), "layout", "multiple of 8 elements");
    declined(withQSeqStride(-64), "layout", "multiple of 8 elements");
    const auto problem = accepted(withQSeqStride(72));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->q.seq, 72);

    // A unit-extent axis has no stride the kernels read.
    SdpaBwdSpec unitBatch;
    unitBatch.batch = 1;
    const auto unit
        = accepted(unitBatch.withEdit([](SdpaBwdGraph& g) { g.tensor("q").strides[0] = 3; }));
    ASSERT_TRUE(unit.has_value());
    EXPECT_EQ(unit->q.batch, 0);
}

TEST(TestFlydslSdpaBwdMatcher, LY4_PackedQkvViewsAndOneBufferGradientsAccept)
{
    // Q, K and V (and dQ, dK, dV) as strided views into one [B, S, 3, H, D] buffer.
    const std::vector<int64_t> packed{16 * 3 * 8 * 64, 64, 3 * 8 * 64, 1};
    const auto problem = accepted(baseline().withEdit([packed](SdpaBwdGraph& g) {
        for(const char* name : {"q", "k", "v", "dq", "dk", "dv"})
        {
            g.tensor(name).strides = packed;
        }
    }));
    ASSERT_TRUE(problem.has_value());
    expectStrides(problem->q, 24576, 64, 1536);
    expectStrides(problem->dK, 24576, 64, 1536);
    expectStrides(problem->o, 8192, 1024, 64);
}

TEST(TestFlydslSdpaBwdMatcher, LY5_OverlappingOutputElementsDecline)
{
    for(const auto& operand :
        {NamedOperand{"dq", "dQ"}, NamedOperand{"dk", "dK"}, NamedOperand{"dv", "dV"}})
    {
        SCOPED_TRACE(operand.tensor);
        const std::string tensor = operand.tensor;
        declined(baseline().withEdit(
                     [tensor](SdpaBwdGraph& g) { g.tensor(tensor).strides = {1024, 128, 8, 1}; }),
                 "layout",
                 std::string(operand.label) + " (uid ");
    }
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g).strides = {2048, 256, 8, 1};
    }),
             "layout",
             "dBias (uid 12) elements overlap");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
    })));
}

TEST(TestFlydslSdpaBwdMatcher, LY6_ZeroSequenceStrideOnAMultiRowAxisDeclines)
{
    for(const auto& operand : {NamedOperand{"q", "Q"},
                               NamedOperand{"k", "K"},
                               NamedOperand{"v", "V"},
                               NamedOperand{"o", "O"},
                               NamedOperand{"do", "dO"}})
    {
        SCOPED_TRACE(operand.tensor);
        const std::string tensor = operand.tensor;
        declined(
            baseline().withEdit([tensor](SdpaBwdGraph& g) { g.tensor(tensor).strides[2] = 0; }),
            "layout",
            std::string(operand.label) + " (uid ");
    }
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("do").strides[2] = 0; }),
             "layout",
             "sequence axis (extent 16) has stride 0");

    // The contiguous stride of the same tensor, and zero batch and head strides, stay legal.
    EXPECT_TRUE(accepted(baseline()));
    const auto broadcast = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.tensor("v").strides[0] = 0;
        g.tensor("v").strides[1] = 0;
    }));
    ASSERT_TRUE(broadcast.has_value());
    expectStrides(broadcast->v, 0, 0, 64);

    // A single-row axis has no row to alias.
    SdpaBwdSpec singleRow;
    singleRow.seqKv = 1;
    const auto single
        = accepted(singleRow.withEdit([](SdpaBwdGraph& g) { g.tensor("v").strides[2] = 0; }));
    ASSERT_TRUE(single.has_value());
    EXPECT_EQ(single->v.seq, 0);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, SS1_PackedStatsInEitherOrderAcceptAndOtherStridesDecline)
{
    auto problem = accepted(baseline());
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->statsLayout, StatsLayout::HT);

    problem = accepted(baseline().withEdit(
        [](SdpaBwdGraph& g) { g.tensor("stats").strides = {16 * 8, 1, 8, 1}; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->statsLayout, StatsLayout::TH);

    declined(
        baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("stats").strides = {129, 16, 1, 1}; }),
        "layout",
        "stats (uid 6) strides [129,16,1,1]");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.tensor("stats").dtype = data_objects::DataType::BFLOAT16;
    }),
             "layout",
             "stats (uid 6) is BFLOAT16, not FLOAT");
}

// ---------------------------------------------------------------------------
// Mask and window
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, MW1_BothDeprecatedCausalFlagsDecline)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.causalMaskBottomRight = true;
    }),
             "mask",
             "both causal_mask and causal_mask_bottom_right set");
    EXPECT_TRUE(
        accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; })));
    EXPECT_TRUE(accepted(
        baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; })));
}

TEST(TestFlydslSdpaBwdMatcher, MW2_CausalFlagImpliesRightBoundZeroUnlessRightBoundIsSet)
{
    auto problem
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->masked);
    EXPECT_EQ(problem->windowRight, 0);
    EXPECT_EQ(problem->windowLeft, NO_WINDOW);
    EXPECT_EQ(problem->diagonalOffset, 0);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.rightBound = 3;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->windowRight, 3);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.leftBound = 5;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->windowLeft, 5);
    EXPECT_EQ(problem->windowRight, 0);
}

TEST(TestFlydslSdpaBwdMatcher, MW3_BoundBelowMinusOneDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.rightBound = -2; }),
             "mask",
             "right_bound -2 is below -1");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = -5; }),
             "mask",
             "left_bound -5 is below -1");
    const auto unset = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = -1;
        g.attributes.rightBound = -1;
    }));
    ASSERT_TRUE(unset.has_value());
    EXPECT_FALSE(unset->masked);
}

TEST(TestFlydslSdpaBwdMatcher,
     MW4_AlignmentOtherThanTopLeftOrBottomRightDeclinesAndFlagsKeepOrForceIt)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.diagonalAlignment = static_cast<data_objects::DiagonalAlignment>(2);
    }),
             "mask",
             "diagonal_alignment 2");
    SdpaBwdSpec rect;
    rect.seqQ = 8;
    rect.seqKv = 16;
    auto problem = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_FALSE(problem->alignBottomRight);
    EXPECT_EQ(problem->diagonalOffset, 0);

    problem = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->alignBottomRight);
    EXPECT_EQ(problem->diagonalOffset, 8);

    problem = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMaskBottomRight = true;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->alignBottomRight);
    EXPECT_EQ(problem->diagonalOffset, 8);

    // Square dims give offset 0 either way; the alignment bit still tells them apart.
    problem = accepted(
        baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->alignBottomRight);
    EXPECT_EQ(problem->diagonalOffset, 0);
}

TEST(TestFlydslSdpaBwdMatcher, MW5_LeftBoundWithoutRightAccepts)
{
    auto problem
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 128; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->masked);
    EXPECT_EQ(problem->windowLeft, 128);
    EXPECT_EQ(problem->windowRight, NO_WINDOW);
    EXPECT_EQ(problem->diagonalOffset, 0);

    SdpaBwdSpec rect;
    rect.seqQ = 300;
    rect.seqKv = 588;
    problem = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 300;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->windowLeft, 300);
    EXPECT_EQ(problem->windowRight, NO_WINDOW);
    EXPECT_EQ(problem->diagonalOffset, 288);
}

TEST(TestFlydslSdpaBwdMatcher, MW6_NonCausalBandWithRightBoundAboveZeroAccepts)
{
    auto problem
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.rightBound = 4; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->masked);
    EXPECT_EQ(problem->windowRight, 4);
    EXPECT_EQ(problem->windowLeft, NO_WINDOW);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 2;
        g.attributes.rightBound = 4;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->windowLeft, 2);
    EXPECT_EQ(problem->windowRight, 4);
}

TEST(TestFlydslSdpaBwdMatcher, MW7_DiagonalOffsetPastSkvMinusSqAccepts)
{
    SdpaBwdSpec rect;
    rect.seqQ = 300;
    rect.seqKv = 588;
    const auto problem = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.rightBound = 4096;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->windowRight, 4096);
    EXPECT_EQ(problem->diagonalOffset, 288);
}

TEST(TestFlydslSdpaBwdMatcher, MW8_LeftBoundIsRecordedInTheCudnnWindowConvention)
{
    // left_bound L keeps the L keys up to the diagonal; the engine derives the kernel's left
    // as L - 1 - diagonalOffset, so the problem carries L itself and the offset.
    SdpaBwdSpec rect;
    rect.seqQ = 8;
    rect.seqKv = 16;
    const auto topLeft
        = accepted(rect.withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 4; }));
    ASSERT_TRUE(topLeft.has_value());
    EXPECT_EQ(topLeft->windowLeft, 4);
    EXPECT_EQ(topLeft->diagonalOffset, 0);

    const auto bottomRight = accepted(rect.withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 4;
        g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }));
    ASSERT_TRUE(bottomRight.has_value());
    EXPECT_EQ(bottomRight->windowLeft, 4);
    EXPECT_EQ(bottomRight->diagonalOffset, 8);
}

TEST(TestFlydslSdpaBwdMatcher, MW9a_LeftBoundZeroWithRightBoundAtLeastOneAccepts)
{
    const auto problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 0;
        g.attributes.rightBound = 1;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->masked);
    EXPECT_EQ(problem->windowLeft, 0);
    EXPECT_EQ(problem->windowRight, 1);
}

TEST(TestFlydslSdpaBwdMatcher, MW9b_LeftBoundZeroWithRightBoundZeroDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 0;
        g.attributes.rightBound = 0;
    }),
             "mask",
             "empty band");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 0;
        g.attributes.causalMask = true;
    }),
             "mask",
             "empty band");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 0;
        g.attributes.rightBound = 1;
    })));
}

TEST(TestFlydslSdpaBwdMatcher, MW9c_LeftBoundZeroWithUnboundedRightDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 0; }),
             "mask",
             "left_bound 0 with an unbounded right bound");
    const auto control
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 1; }));
    ASSERT_TRUE(control.has_value());
    EXPECT_EQ(control->windowLeft, 1);
    EXPECT_EQ(control->windowRight, NO_WINDOW);
}

TEST(TestFlydslSdpaBwdMatcher, MW10_AnyMaskWithBiasDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        g.attributes.causalMask = true;
    }),
             "mask",
             "bias (uid 11)");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        g.attributes.leftBound = 4;
    }),
             "mask",
             "bias (uid 11)");
    const auto biasOnly = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g); }));
    ASSERT_TRUE(biasOnly.has_value());
    EXPECT_EQ(biasOnly->biasUid, sdpa_bwd_uid::BIAS);
    EXPECT_FALSE(biasOnly->masked);
}

TEST(TestFlydslSdpaBwdMatcher, MW11_BottomRightCausalWithDropoutAccepts)
{
    const auto problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMaskBottomRight = true;
        addDropout(g, 0.25F);
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->masked);
    EXPECT_EQ(problem->windowRight, 0);
    EXPECT_EQ(problem->dropoutP, 0.25F);
}

TEST(TestFlydslSdpaBwdMatcher, MW12_AlibiMaskDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.alibiMask = true; }),
             "alibi_mask",
             "alibi_mask");
    EXPECT_TRUE(
        accepted(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.alibiMask = false; })));
}

TEST(TestFlydslSdpaBwdMatcher, MW13_WindowWithBottomRightAlignmentUnderPaddingMaskDeclines)
{
    const auto bottomRightCausal = [](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.attributes.causalMaskBottomRight = true;
    };
    declined(baseline().withEdit(bottomRightCausal).withEdit([](SdpaBwdGraph& g) {
        g.attributes.leftBound = 4;
    }),
             "mask",
             "BOTTOM_RIGHT alignment under padding_mask");
    declined(baseline().withEdit(bottomRightCausal).withEdit([](SdpaBwdGraph& g) {
        g.attributes.rightBound = 2;
    }),
             "mask",
             "BOTTOM_RIGHT alignment under padding_mask");
    // Plain bottom-right causal stays servable through the sentinel.
    const auto plain = accepted(baseline().withEdit(bottomRightCausal));
    ASSERT_TRUE(plain.has_value());
    EXPECT_TRUE(plain->alignBottomRight);
    EXPECT_TRUE(plain->paddingMask);
    // Top-left needs no per-sequence anchoring, so the same window is servable there.
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.attributes.leftBound = 4;
    })));
}

// ---------------------------------------------------------------------------
// Padding and varlen
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, PV1_PaddingMaskWithBothLengthTensorsAccepts)
{
    const auto problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addSeqLens(g); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->paddingMask);
    EXPECT_EQ(problem->seqLenQUid, sdpa_bwd_uid::SEQ_LEN_Q);
    EXPECT_EQ(problem->seqLenKvUid, sdpa_bwd_uid::SEQ_LEN_KV);
}

TEST(TestFlydslSdpaBwdMatcher, PV2_PaddingFlagAndLengthTensorsMustComeTogether)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.paddingMask = true; }),
             "varlen",
             "padding_mask is set without seq_len tensors");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.attributes.paddingMask = false;
    }),
             "varlen",
             "seq_len tensors are set without padding_mask");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.attributes.seqLenKvUid.reset();
    }),
             "varlen",
             "only one of seq_len_q and seq_len_kv");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addSeqLens(g); })));
}

TEST(TestFlydslSdpaBwdMatcher, PV3_RaggedOffsetOnADataTensorDeclines)
{
    for(const auto& operand : OPERANDS)
    {
        if(std::string_view(operand.tensor) == "stats")
        {
            continue;
        }
        SCOPED_TRACE(operand.tensor);
        const std::string tensor = operand.tensor;
        declined(baseline().withEdit(
                     [tensor](SdpaBwdGraph& g) { g.tensor(tensor).raggedOffsetUid = 99; }),
                 "varlen",
                 std::string(operand.label) + " (uid ");
    }
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addSeqLens(g); })));
}

TEST(TestFlydslSdpaBwdMatcher, PV4_SeqLenOtherThanInt32Declines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_q").dtype = data_objects::DataType::INT64;
    }),
             "varlen",
             "seq_len_q (uid 13) is INT64, not INT32");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_kv").dims = {1, 1, 1, 1};
    }),
             "varlen",
             "seq_len_kv (uid 14) dims [1,1,1,1] != [2,1,1,1]");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_q").strides = {2, 1, 1, 1};
    }),
             "varlen",
             "seq_len_q (uid 13) batch stride 2 is not 1");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_q").dtype = data_objects::DataType::INT32;
    })));
}

TEST(TestFlydslSdpaBwdMatcher, PV7_SeqLenMustBeADeviceTensorWithoutRaggedOffsets)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_q").isVirtual = true;
    }),
             "varlen",
             "seq_len_q (uid 13) is virtual");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_kv").runtimePassByValue = true;
    }),
             "varlen",
             "seq_len_kv (uid 14) is pass-by-value");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_q").value = int32_t{3};
    }),
             "varlen",
             "seq_len_q (uid 13) is pass-by-value");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        g.tensor("seq_len_kv").raggedOffsetUid = 99;
    }),
             "varlen",
             "seq_len_kv (uid 14) has a ragged offset tensor");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addSeqLens(g); })));
}

TEST(TestFlydslSdpaBwdMatcher, PV8_RaggedOffsetOnBiasDbiasSeedOffsetOrStatsDeclines)
{
    const auto withBias = [](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
    };
    declined(baseline().withEdit(withBias).withEdit(
                 [](SdpaBwdGraph& g) { g.tensor("bias").raggedOffsetUid = 99; }),
             "varlen",
             "bias (uid 11) has a ragged offset tensor");
    declined(baseline().withEdit(withBias).withEdit(
                 [](SdpaBwdGraph& g) { g.tensor("dbias").raggedOffsetUid = 99; }),
             "varlen",
             "dBias (uid 12) has a ragged offset tensor");
    const auto withDropout = [](SdpaBwdGraph& g) { addDropout(g); };
    declined(baseline().withEdit(withDropout).withEdit([](SdpaBwdGraph& g) {
        g.tensor("seed").raggedOffsetUid = 99;
    }),
             "varlen",
             "seed (uid 15) has a ragged offset tensor");
    declined(baseline().withEdit(withDropout).withEdit([](SdpaBwdGraph& g) {
        g.tensor("offset").raggedOffsetUid = 99;
    }),
             "varlen",
             "offset (uid 16) has a ragged offset tensor");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.tensor("stats").raggedOffsetUid = 99; }),
             "varlen",
             "stats (uid 6) has a ragged offset tensor");
    EXPECT_TRUE(accepted(baseline().withEdit(withBias).withEdit(withDropout)));
}

TEST(TestFlydslSdpaBwdMatcher, PV10_BiasOrDbiasWithPaddingMaskDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        addBias(g);
    }),
             "varlen",
             "bias (uid 11) with padding_mask");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addSeqLens(g);
        addBias(g);
        addDbias(g);
    }),
             "varlen",
             "bias (uid 11) with padding_mask");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
    })));
}

// ---------------------------------------------------------------------------
// Bias and dBias
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, BI1_BiasRankUpToFourRightAlignedWithBroadcastBatchHeadAndRows)
{
    struct Case
    {
        std::vector<int64_t> dims;
        bool broadcastBatch;
        bool broadcastHead;
        int64_t seqStride;
    };
    const std::vector<Case> accepts{
        {{2, 8, 16, 16}, false, false, 16},
        {{1, 8, 16, 16}, true, false, 16},
        {{2, 1, 16, 16}, false, true, 16},
        {{2, 8, 1, 16}, false, false, 0},
        {{8, 16, 16}, true, false, 16},
        {{16, 16}, true, true, 16},
        {{16}, true, true, 0},
    };
    for(const auto& c : accepts)
    {
        SCOPED_TRACE(flydsl_sdpa_bwd::detail::listOf(c.dims));
        const auto dims = c.dims;
        const auto problem
            = accepted(baseline().withEdit([dims](SdpaBwdGraph& g) { addBias(g, dims); }));
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->biasUid, sdpa_bwd_uid::BIAS);
        EXPECT_EQ(problem->biasBroadcastBatch, c.broadcastBatch);
        EXPECT_EQ(problem->biasBroadcastHead, c.broadcastHead);
        EXPECT_EQ(problem->biasStrides.seq, c.seqStride);
        EXPECT_EQ(problem->biasAlignmentBytes, 16);
    }
    // The key axis does not broadcast, and the rank stops at four.
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g, {2, 8, 16, 1}); }),
             "attn_mask",
             "do not broadcast to [2,8,16,16]");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g, {2, 8, 16, 8}); }),
             "attn_mask",
             "do not broadcast");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g, {1, 2, 8, 16, 16}); }),
             "attn_mask",
             "rank 5");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g).strides[2] = -16; }),
             "layout",
             "bias (uid 11) has a negative stride");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g).strides[2] = 16; })));
}

TEST(TestFlydslSdpaBwdMatcher, BI2_BiasDtypeOtherThanQsDeclines)
{
    declined(baseline().withEdit(
                 [](SdpaBwdGraph& g) { addBias(g).dtype = data_objects::DataType::BFLOAT16; }),
             "attn_mask",
             "bias (uid 11) is BFLOAT16, but Q is HALF");
    declined(baseline().withEdit(
                 [](SdpaBwdGraph& g) { addBias(g).dtype = data_objects::DataType::BOOLEAN; }),
             "attn_mask",
             "BOOLEAN");
    EXPECT_TRUE(accepted(baseline().withEdit(
        [](SdpaBwdGraph& g) { addBias(g).dtype = data_objects::DataType::HALF; })));
}

TEST(TestFlydslSdpaBwdMatcher, BI3_BiasBroadcastOverBatchOrHeadWithDbiasAccepts)
{
    auto problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g, {1, 8, 16, 16});
        addDbias(g);
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->biasBroadcastBatch);
    EXPECT_FALSE(problem->biasBroadcastHead);
    EXPECT_EQ(problem->dbiasUid, sdpa_bwd_uid::DBIAS);
    expectStrides(problem->dbiasStrides, 0, 256, 16);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g, {2, 1, 16, 16});
        addDbias(g);
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_FALSE(problem->biasBroadcastBatch);
    EXPECT_TRUE(problem->biasBroadcastHead);
    EXPECT_EQ(problem->dbiasUid, sdpa_bwd_uid::DBIAS);
}

TEST(TestFlydslSdpaBwdMatcher, BI4_RowBroadcastBiasWithDbiasDeclinesAndBiasAloneAccepts)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g, {2, 8, 1, 16});
        addDbias(g);
    }),
             "dbias",
             "row-broadcast bias");
    const auto biasOnly
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addBias(g, {2, 8, 1, 16}); }));
    ASSERT_TRUE(biasOnly.has_value());
    EXPECT_EQ(biasOnly->biasUid, sdpa_bwd_uid::BIAS);
    EXPECT_EQ(biasOnly->dbiasUid, NO_UID);
}

TEST(TestFlydslSdpaBwdMatcher, BI5_DbiasShapeOtherThanBiasShapeDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g).dims = {1, 8, 16, 16};
    }),
             "dbias",
             "dBias dims [1,8,16,16] != bias dims [2,8,16,16]");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
    })));
}

TEST(TestFlydslSdpaBwdMatcher, BI6_DbiasWithoutBiasDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
        g.attributes.biasUid.reset();
    }),
             "dbias",
             "without a bias");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g);
    })));
}

TEST(TestFlydslSdpaBwdMatcher, BI7_DbiasDtypeOtherThanBiasDtypeDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g).dtype = data_objects::DataType::BFLOAT16;
    }),
             "dbias",
             "dBias (uid 12) is BFLOAT16, but bias is HALF");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addDbias(g).dtype = data_objects::DataType::HALF;
    })));
}

// ---------------------------------------------------------------------------
// Dropout and seed
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, DR1_NullOrZeroProbabilityIsOffAndOpenUnitIntervalIsOn)
{
    auto problem = accepted(baseline());
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dropoutP, 0.0F);
    EXPECT_EQ(problem->seedWidth, SeedWidth::NONE);
    EXPECT_EQ(problem->seedForm, SeedForm::NONE);

    problem = accepted(
        baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.dropoutProbability = 0.0F; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dropoutP, 0.0F);
    EXPECT_EQ(problem->seedUid, NO_UID);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g, 0.5F); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dropoutP, 0.5F);
    EXPECT_EQ(problem->seedUid, sdpa_bwd_uid::SEED);
    EXPECT_EQ(problem->offsetUid, sdpa_bwd_uid::OFFSET);
}

TEST(TestFlydslSdpaBwdMatcher, DR2_ProbabilityOutsideZeroToOneDeclines)
{
    for(const float p : {1.0F, 1.5F, -0.1F, std::numeric_limits<float>::quiet_NaN()})
    {
        SCOPED_TRACE(p);
        declined(baseline().withEdit([p](SdpaBwdGraph& g) { addDropout(g, p); }),
                 "dropout",
                 "is outside [0, 1)");
    }
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g, 0.999F); })));
}

TEST(TestFlydslSdpaBwdMatcher, DR3_ProbabilityNeedsBothSeedAndOffsetAndTheyNeedProbability)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.attributes.offsetUid.reset();
    }),
             "dropout",
             "needs both seed and offset");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.attributes.seedUid.reset();
    }),
             "dropout",
             "needs both seed and offset");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.dropoutProbability = 0.3F; }),
             "dropout",
             "needs both seed and offset");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.attributes.dropoutProbability.reset();
    }),
             "dropout",
             "without dropout_probability > 0");
    declined(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g, 0.0F); }),
             "dropout",
             "without dropout_probability > 0");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g); })));
}

TEST(TestFlydslSdpaBwdMatcher, DR4_Int32AndInt64SeedAndOffsetAcceptAndRecordTheWidth)
{
    auto problem = accepted(baseline().withEdit(
        [](SdpaBwdGraph& g) { addDropout(g, 0.2F, data_objects::DataType::INT32); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->seedWidth, SeedWidth::INT32);

    problem = accepted(baseline().withEdit(
        [](SdpaBwdGraph& g) { addDropout(g, 0.2F, data_objects::DataType::INT64); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->seedWidth, SeedWidth::INT64);

    declined(baseline().withEdit(
                 [](SdpaBwdGraph& g) { addDropout(g, 0.2F, data_objects::DataType::FLOAT); }),
             "dropout",
             "is FLOAT, not INT32 or INT64");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g, 0.2F, data_objects::DataType::INT64);
        g.tensor("offset").dtype = data_objects::DataType::INT32;
    }),
             "dropout",
             "seed (uid 15) is INT64 but offset (uid 16) is INT32");
}

TEST(TestFlydslSdpaBwdMatcher, DR6_SeedAndOffsetAsDeviceOrPassByValueScalarsAccept)
{
    auto problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->seedForm, SeedForm::DEVICE);

    for(const auto type : {data_objects::DataType::INT32, data_objects::DataType::INT64})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(type));
        problem = accepted(
            baseline().withEdit([type](SdpaBwdGraph& g) { addDropout(g, 0.2F, type, true); }));
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->seedForm, SeedForm::PASS_BY_VALUE);
        EXPECT_EQ(problem->seedWidth,
                  type == data_objects::DataType::INT32 ? SeedWidth::INT32 : SeedWidth::INT64);
    }
    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.tensor("seed").runtimePassByValue = true;
        g.tensor("offset").runtimePassByValue = true;
    }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->seedForm, SeedForm::PASS_BY_VALUE);

    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.tensor("seed").isVirtual = true;
    }),
             "dropout",
             "seed (uid 15) is virtual");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.tensor("offset").dims = {1};
        g.tensor("offset").strides = {1};
    }),
             "dropout",
             "offset (uid 16) dims [1] != [1,1,1,1]");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.tensor("seed").dims = {1, 1, 1, 2};
    }),
             "dropout",
             "seed (uid 15) dims [1,1,1,2] != [1,1,1,1]");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addDropout(g);
        g.tensor("seed").value = int64_t{7};
    }),
             "dropout",
             "differ");
}

TEST(TestFlydslSdpaBwdMatcher, DR7_DropoutMaskScaleAndScaleInvTensorsDecline)
{
    declined(baseline().withEdit(
                 [](SdpaBwdGraph& g) { g.attributes.dropoutMaskUid = sdpa_bwd_uid::DROPOUT_MASK; }),
             "dropout",
             "dropout_mask_tensor_uid");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.dropoutScaleUid = sdpa_bwd_uid::DROPOUT_SCALE;
    }),
             "dropout",
             "dropout_scale_tensor_uid");
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        g.attributes.dropoutScaleInvUid = sdpa_bwd_uid::DROPOUT_SCALE_INV;
    }),
             "dropout",
             "dropout_scale_inv_tensor_uid");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addDropout(g); })));
}

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMatcher, SC1_ScaleValueOrPassByValueTensorAcceptsAndAbsentIsOne)
{
    auto problem = accepted(baseline());
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->scaleSource, ScaleSource::VALUE);
    EXPECT_EQ(problem->scaleValue, 1.0F);

    problem = accepted(
        baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.attnScaleValue = 0.125F; }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->scaleSource, ScaleSource::VALUE);
    EXPECT_EQ(problem->scaleValue, 0.125F);
    EXPECT_EQ(problem->scaleUid, NO_UID);

    problem = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, 0.25F); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->scaleSource, ScaleSource::TENSOR);
    EXPECT_EQ(problem->scaleUid, sdpa_bwd_uid::SCALE);
    EXPECT_EQ(problem->scaleValue, 0.25F);

    problem
        = accepted(baseline().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, std::nullopt); }));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->scaleSource, ScaleSource::TENSOR);
    EXPECT_EQ(problem->scaleUid, sdpa_bwd_uid::SCALE);
}

TEST(TestFlydslSdpaBwdMatcher, SC2_DeviceScaleTensorDeclines)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        auto& scale = addScaleTensor(g, std::nullopt);
        scale.runtimePassByValue = false;
    }),
             "scale",
             "scale (uid 10) is a device tensor");
    EXPECT_TRUE(accepted(baseline().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, 0.25F); })));
}

TEST(TestFlydslSdpaBwdMatcher, SC3_ScaleValueAndScaleTensorTogetherDecline)
{
    declined(baseline().withEdit([](SdpaBwdGraph& g) {
        addScaleTensor(g, 0.25F);
        g.attributes.attnScaleValue = 0.125F;
    }),
             "scale",
             "both attn_scale_value and scale_tensor_uid set");
    EXPECT_TRUE(accepted(
        baseline().withEdit([](SdpaBwdGraph& g) { g.attributes.attnScaleValue = 0.125F; })));
}

TEST(TestFlydslSdpaBwdMatcher, SC4_ZeroAndNegativeScalesAccept)
{
    for(const float scale : {0.0F, -1.2F, 1.2F})
    {
        SCOPED_TRACE(scale);
        auto problem = accepted(
            baseline().withEdit([scale](SdpaBwdGraph& g) { g.attributes.attnScaleValue = scale; }));
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->scaleSource, ScaleSource::VALUE);
        EXPECT_EQ(problem->scaleValue, scale);

        problem
            = accepted(baseline().withEdit([scale](SdpaBwdGraph& g) { addScaleTensor(g, scale); }));
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->scaleSource, ScaleSource::TENSOR);
        EXPECT_EQ(problem->scaleValue, scale);
    }
}

TEST(TestFlydslSdpaBwdMatcher, SC5_ScaleBeyond2Pow20InMagnitudeOrNonFiniteDeclines)
{
    for(const float scale : {TWO_POW_20 + 1.0F,
                             -2.0F * TWO_POW_20,
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()})
    {
        SCOPED_TRACE(scale);
        declined(
            baseline().withEdit([scale](SdpaBwdGraph& g) { g.attributes.attnScaleValue = scale; }),
            "scale",
            "attn_scale_value");
        declined(baseline().withEdit([scale](SdpaBwdGraph& g) { addScaleTensor(g, scale); }),
                 "scale",
                 "scale (uid 10) value");
    }
    for(const float scale : {TWO_POW_20, -TWO_POW_20})
    {
        SCOPED_TRACE(scale);
        EXPECT_TRUE(accepted(baseline().withEdit(
            [scale](SdpaBwdGraph& g) { g.attributes.attnScaleValue = scale; })));
        EXPECT_TRUE(
            accepted(baseline().withEdit([scale](SdpaBwdGraph& g) { addScaleTensor(g, scale); })));
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

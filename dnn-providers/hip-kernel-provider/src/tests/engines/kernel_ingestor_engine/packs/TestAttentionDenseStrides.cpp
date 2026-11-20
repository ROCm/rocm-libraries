// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDenseStrides.hpp"

/**
 * @file TestAttentionDenseStrides.cpp
 * @brief The C++ twin of rocKE's `dense_strides_admit`, held to the vectors file the Python
 *        predicate is held to.
 *
 * `rocke/platform/tests/golden/attention_dense_stride_vectors.json` is written by
 * `rocke/library/tests/test_attention_dense_strided_admission.py --regenerate` from its
 * hand-written case table, and that test requires the Python predicate to give every row's
 * answer. This suite requires the same of denseStridesAdmit: every row admits or declines as
 * recorded, with the recorded reason, and an admitted row launches with the recorded canonical
 * strides. The file is embedded into this binary at configure time (CMakeLists.txt), so the
 * installed test needs no data path.
 *
 * The rest of the file covers what the vectors cannot say: the pieces of the header the
 * engine calls on their own (packed launch values, the slab rule), and the arithmetic past
 * int64_t that the Python, whose integers do not overflow, never meets.
 *
 * Pure host code: no device, no graph, no engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace attention_dense;
using nlohmann::json;

// The embedded file: a byte array whose size is the include's, which no std::array spells.
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
const char VECTORS_JSON[] = {
#include "attention_dense_stride_vectors.inc"
    0};

/// The rows of the vectors file. Update with the file: it pins that none was dropped on the way
/// in, which no per-row check can see.
constexpr std::size_t VECTOR_CASE_COUNT = 110;

/// One row of the vectors file.
struct VectorCase
{
    std::string name;
    DenseStrideProblem problem;
    std::array<DenseOperand, 4> operands;
    /// "admit", or the reason token of the decline.
    std::string expect;
    /// The canonical strides of an admit row.
    std::optional<AttentionDenseStrides> strides;
};

std::vector<int64_t> int64Array(const json& values)
{
    std::vector<int64_t> out;
    for(const auto& value : values)
    {
        out.push_back(value.get<int64_t>());
    }
    return out;
}

DenseOperand operandOf(const json& row)
{
    DenseOperand operand;
    operand.dims = int64Array(row.at("dims"));
    operand.strides = int64Array(row.at("strides"));
    operand.align = row.at("align").get<int64_t>();
    operand.isVirtual = row.at("virtual").get<bool>();
    operand.isRagged = row.at("ragged").get<bool>();
    return operand;
}

/// The parsed file, or null when it does not parse.
const json* vectors()
{
    static const json s_parsed = json::parse(VECTORS_JSON, nullptr, /*allow_exceptions=*/false);
    return s_parsed.is_object() ? &s_parsed : nullptr;
}

std::vector<VectorCase> loadCases()
{
    std::vector<VectorCase> cases;
    const json* file = vectors();
    if(file == nullptr || !file->contains("cases"))
    {
        return cases;
    }
    for(const auto& row : file->at("cases"))
    {
        VectorCase testCase;
        testCase.name = row.at("name").get<std::string>();
        const auto& problem = row.at("problem");
        testCase.problem.batch = problem.at("batch").get<int64_t>();
        testCase.problem.seqLenQ = problem.at("seqlen_q").get<int64_t>();
        testCase.problem.seqLenKv = problem.at("seqlen_kv").get<int64_t>();
        testCase.problem.numQueryHeads = problem.at("num_query_heads").get<int64_t>();
        testCase.problem.numKvHeads = problem.at("num_kv_heads").get<int64_t>();
        testCase.problem.headSize = problem.at("head_size").get<int64_t>();
        testCase.problem.blockN = problem.at("block_n").get<int64_t>();
        testCase.operands = {operandOf(row.at("q")),
                             operandOf(row.at("k")),
                             operandOf(row.at("v")),
                             operandOf(row.at("o"))};
        testCase.expect = row.at("expect").get<std::string>();
        if(row.contains("strides"))
        {
            const auto values = int64Array(row.at("strides"));
            AttentionDenseStrides strides{};
            if(values.size() == strides.size())
            {
                std::copy(values.begin(), values.end(), strides.begin());
                testCase.strides = strides;
            }
        }
        cases.push_back(std::move(testCase));
    }
    return cases;
}

class TestAttentionDenseStrideVectors : public ::testing::TestWithParam<VectorCase>
{
};

/// One packed BSHD operand of the vectors' base problem (B2, Hq 8, Hkv 2, S 64, D128, bn 64).
DenseStrideProblem baseProblem()
{
    DenseStrideProblem problem;
    problem.batch = 2;
    problem.seqLenQ = 64;
    problem.seqLenKv = 64;
    problem.numQueryHeads = 8;
    problem.numKvHeads = 2;
    problem.headSize = 128;
    problem.blockN = 64;
    return problem;
}

/// The four packed operands of @p problem, as the graph declares them.
std::array<DenseOperand, 4> packedOperands(const DenseStrideProblem& problem)
{
    std::array<DenseOperand, 4> operands;
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto dims = denseOperandDims(problem, tensor);
        operands.at(tensor).dims.assign(dims.begin(), dims.end());
        // The raw packed strides: batch S*H*D, head D, token H*D, whatever the extent.
        operands.at(tensor).strides
            = {dims.at(2) * dims.at(1) * dims.at(3), dims.at(3), dims.at(1) * dims.at(3), 1};
    }
    return operands;
}

} // namespace

TEST(TestAttentionDenseStrides, ReadsEveryRowOfTheVectorsFile)
{
    ASSERT_NE(vectors(), nullptr) << "the embedded vectors file is not valid JSON";
    EXPECT_EQ(loadCases().size(), VECTOR_CASE_COUNT);
}

TEST(TestAttentionDenseStrides, TheHeaderConstantsAreTheVectorsFilesOwn)
{
    const json* file = vectors();
    ASSERT_NE(file, nullptr);
    EXPECT_EQ(file->at("elem_bytes").get<int64_t>(), DENSE_ELEM_BYTES);
    EXPECT_EQ(file->at("stride_multiple").get<int64_t>(), DENSE_STRIDE_MULTIPLE);
    EXPECT_EQ(file->at("pointer_align").get<int64_t>(), DENSE_POINTER_ALIGN);
    EXPECT_EQ(file->at("slab_slack_tiles").get<int64_t>(), DENSE_SLAB_SLACK_TILES);

    const auto& reasons = file->at("reasons");
    ASSERT_EQ(reasons.size(), DENSE_STRIDE_REASONS.size());
    for(std::size_t index = 0; index < reasons.size(); ++index)
    {
        EXPECT_EQ(reasons.at(index).get<std::string>(), DENSE_STRIDE_REASONS.at(index));
        EXPECT_EQ(denseStrideReasonToken(static_cast<DenseStrideReason>(index)),
                  DENSE_STRIDE_REASONS.at(index));
    }
}

TEST(TestAttentionDenseStrides, EveryDeclineReasonHasARowInTheVectorsFile)
{
    std::array<int, DENSE_STRIDE_REASONS.size()> declines{};
    int admits = 0;
    for(const auto& testCase : loadCases())
    {
        if(testCase.expect == "admit")
        {
            ++admits;
            continue;
        }
        for(std::size_t index = 0; index < DENSE_STRIDE_REASONS.size(); ++index)
        {
            if(testCase.expect == DENSE_STRIDE_REASONS.at(index))
            {
                ++declines.at(index);
            }
        }
    }
    EXPECT_GT(admits, 0);
    for(std::size_t index = 0; index < declines.size(); ++index)
    {
        EXPECT_GT(declines.at(index), 0) << DENSE_STRIDE_REASONS.at(index);
    }
}

TEST_P(TestAttentionDenseStrideVectors, AdmitsOrDeclinesAsTheRowRecords)
{
    const auto& testCase = GetParam();
    const auto verdict = denseStridesAdmit(testCase.problem, testCase.operands);
    if(testCase.expect == "admit")
    {
        EXPECT_TRUE(verdict.ok) << "declined (" << denseStrideReasonToken(verdict.reason)
                                << "): " << verdict.detail;
        ASSERT_TRUE(testCase.strides.has_value()) << "an admit row without canonical strides";
        EXPECT_EQ(verdict.strides, *testCase.strides);
    }
    else
    {
        ASSERT_FALSE(verdict.ok) << "admitted, expected " << testCase.expect;
        EXPECT_EQ(denseStrideReasonToken(verdict.reason), testCase.expect) << verdict.detail;
        EXPECT_FALSE(verdict.detail.empty());
    }
}

INSTANTIATE_TEST_SUITE_P(Vectors,
                         TestAttentionDenseStrideVectors,
                         ::testing::ValuesIn(loadCases()),
                         [](const ::testing::TestParamInfo<VectorCase>& info) {
                             return info.param.name;
                         });

// =============================================================================
// The pieces the engine calls on their own.
// =============================================================================

TEST(TestAttentionDenseStrides, PackedLaunchStridesAreTheCanonicalPackedValues)
{
    auto problem = baseProblem();
    const auto packed = packedLaunchStrides(problem);
    ASSERT_TRUE(packed.has_value());
    // The vectors' first row: packed BSHD, B2, Hq 8, Hkv 2, S 64, D128.
    const AttentionDenseStrides expected{
        65536, 128, 1024, 16384, 128, 256, 16384, 128, 256, 65536, 128, 1024};
    EXPECT_EQ(*packed, expected);

    // Extent-1 axes canonicalize: batch 1, one KV head and one query row. Q and O: batch 0,
    // head D, token D (an extent-1 token stride is D); K and V: batch 0, head 0 (one KV
    // head), token H * D.
    problem.batch = 1;
    problem.numKvHeads = 1;
    problem.seqLenQ = 1;
    const auto canonical = packedLaunchStrides(problem);
    ASSERT_TRUE(canonical.has_value());
    const AttentionDenseStrides expectedCanonical{0, 128, 128, 0, 0, 128, 0, 0, 128, 0, 128, 128};
    EXPECT_EQ(*canonical, expectedCanonical);
}

TEST(TestAttentionDenseStrides, PackedLaunchStridesRefuseAProductPastInt64)
{
    auto problem = baseProblem();
    problem.seqLenQ = int64_t{1} << 40;
    problem.numQueryHeads = int64_t{1} << 20;
    problem.headSize = int64_t{1} << 10;
    EXPECT_FALSE(packedLaunchStrides(problem).has_value());
}

TEST(TestAttentionDenseStrides,
     CanonicalizationKeepsEveryStrideOfAnAxisThatStepsAndOverwritesOnesThatDoNot)
{
    auto problem = baseProblem();
    problem.batch = 1;
    problem.numKvHeads = 1;
    AttentionDenseStrides raw{};
    for(std::size_t index = 0; index < raw.size(); ++index)
    {
        raw.at(index) = static_cast<int64_t>(1000 + index);
    }
    const auto canonical = canonicalDenseStrides(problem, raw);
    // Batch (extent 1) of every tensor and the head of K and V (one KV head) are rewritten.
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        EXPECT_EQ(canonical.at(denseStrideIndex(tensor, 0)), 0) << tensor;
        EXPECT_EQ(canonical.at(denseStrideIndex(tensor, 2)), raw.at(denseStrideIndex(tensor, 2)))
            << tensor;
    }
    EXPECT_EQ(canonical.at(denseStrideIndex(0, 1)), raw.at(denseStrideIndex(0, 1)));
    EXPECT_EQ(canonical.at(denseStrideIndex(1, 1)), 0);
    EXPECT_EQ(canonical.at(denseStrideIndex(2, 1)), 0);
    EXPECT_EQ(canonical.at(denseStrideIndex(3, 1)), raw.at(denseStrideIndex(3, 1)));
}

TEST(TestAttentionDenseStrides, TheSlabBoundIsTheLastElementOfTheLastRowInItsUnit)
{
    // 64 rows of stride 256 and D128: the last row starts at 63 * 256, one row is 128 long.
    EXPECT_EQ(denseSlabBytes(64, 256, 128, 2), 2 * (63 * 256 + 128));
    EXPECT_EQ(denseSlabBytes(64, 256, 128, 1), 63 * 256 + 128);
    EXPECT_EQ(denseSlabBytes(1, 4096, 128, 2), 2 * 128);
    // No rows, no bytes: the empty KV segment of a workgroup.
    EXPECT_EQ(denseSlabBytes(0, 256, 128, 2), 0);
    EXPECT_EQ(denseSlabBytes(-5, 256, 128, 2), 0);
    // A product past int64_t is reported, not wrapped.
    EXPECT_FALSE(denseSlabBytes(std::numeric_limits<int64_t>::max(), 4, 128, 2).has_value());
    EXPECT_FALSE(denseSlabBytes(int64_t{1} << 40, int64_t{1} << 30, 128, 2).has_value());
    EXPECT_FALSE(
        denseSlabBytes(1 << 20, 1 << 20, std::numeric_limits<int64_t>::max(), 2).has_value());
}

TEST(TestAttentionDenseStrides, TheKeyValueReachPadsToTheTileThenAddsTwoTiles)
{
    EXPECT_EQ(denseKvReachRows(64, 64), 64 + 2 * 64);
    EXPECT_EQ(denseKvReachRows(65, 64), 128 + 2 * 64);
    EXPECT_EQ(denseKvReachRows(1, 32), 32 + 2 * 32);
    EXPECT_EQ(denseKvReachRows(256, 128), 256 + 2 * 128);
    EXPECT_FALSE(denseKvReachRows(64, 0).has_value());
    EXPECT_FALSE(denseKvReachRows(64, -64).has_value());
    EXPECT_FALSE(denseKvReachRows(std::numeric_limits<int64_t>::max(), 64).has_value());
}

TEST(TestAttentionDenseStrides, TheKeyValueSlabFitsUpToTheLastRowTheSlackReaches)
{
    // seqlen_kv 64, block_n 64, D128: reach rows 192; the rule is 2 * (191 * sS + 128) - 2 <
    // 2^31, i.e. 191 * sS + 127 < 2^30. 5621680 is the last admitted token stride.
    EXPECT_TRUE(denseKvSlabFits(64, 5621680, 128, 64));
    EXPECT_FALSE(denseKvSlabFits(64, 5621688, 128, 64));
    // Without the slack (rows = 64) the stride 8000000 would fit; with it, it does not.
    EXPECT_FALSE(denseKvSlabFits(64, 8000000, 128, 64));
    // A larger tile reaches further: the same stride that fits at block_n 64 does not at 128.
    EXPECT_TRUE(denseKvSlabFits(64, 4210744, 128, 64));
    EXPECT_FALSE(denseKvSlabFits(64, 4210744, 128, 128));
    // A reach of exactly 2^31 bytes is admitted: the last element starts at 2^31 - 2.
    EXPECT_TRUE(denseKvSlabFits(8388480, 128, 128, 64));
    EXPECT_FALSE(denseKvSlabFits(8388481, 128, 128, 64));
    // Past int64_t is past the limit.
    EXPECT_FALSE(denseKvSlabFits(int64_t{1} << 40, int64_t{1} << 40, 128, 64));
}

TEST(TestAttentionDenseStrides, TheQuerySlabFitsBelow2p31Elements)
{
    // (256 - 1) * 8421496 + 128 = 2147481608 < 2^31; one more 8 elements of stride is not.
    EXPECT_TRUE(denseQoSlabFits(256, 8421496, 128));
    EXPECT_FALSE(denseQoSlabFits(256, 8421504, 128));
    EXPECT_FALSE(denseQoSlabFits(int64_t{1} << 40, int64_t{1} << 40, 128));
}

// =============================================================================
// Arithmetic past int64_t. Python integers do not overflow; every product here is checked,
// and one past int64_t is past the limit it is compared with.
// =============================================================================

TEST(TestAttentionDenseStrides, DeclinesAnExtentWhoseSlabBaseOverflowsInt64)
{
    // B 2^62 with batch stride 2^30 (below 2^31): 2 * (B - 1) * sB is past 2^63.
    DenseStrideProblem problem;
    problem.batch = int64_t{1} << 62;
    problem.seqLenQ = 8;
    problem.seqLenKv = 8;
    problem.numQueryHeads = 1;
    problem.numKvHeads = 1;
    problem.headSize = 64;
    problem.blockN = 32;
    std::array<DenseOperand, 4> operands;
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        operands.at(tensor).dims = {problem.batch, 1, 8, 64};
        operands.at(tensor).strides = {int64_t{1} << 30, 64, 64, 1};
    }
    const auto verdict = denseStridesAdmit(problem, operands);
    ASSERT_FALSE(verdict.ok);
    EXPECT_EQ(verdict.reason, DenseStrideReason::SIZE_LIMIT) << verdict.detail;
}

TEST(TestAttentionDenseStrides, DeclinesALastSlabByteAtOrPast2p63ThoughTheBaseFits)
{
    // One head, 8 tokens of 64 (slab 1024 bytes), batch stride 256 * 6366403 elements: (B - 1)
    // is (2^54 - 1) / 6366403, so the base 2 * (B - 1) * sB is 2^63 - 512 and fits int64_t,
    // while base + slab is 2^63 + 512 and does not. 6366403 divides 2^54 - 1.
    constexpr int64_t DIVISOR = 6366403;
    const int64_t batchStride = 256 * DIVISOR;
    const int64_t batch = ((int64_t{1} << 54) - 1) / DIVISOR + 1;
    ASSERT_EQ((batch - 1) * DIVISOR, (int64_t{1} << 54) - 1);
    ASSERT_LT(batchStride, int64_t{1} << 31);

    DenseStrideProblem problem;
    problem.batch = batch;
    problem.seqLenQ = 8;
    problem.seqLenKv = 8;
    problem.numQueryHeads = 1;
    problem.numKvHeads = 1;
    problem.headSize = 64;
    problem.blockN = 32;
    std::array<DenseOperand, 4> operands;
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        operands.at(tensor).dims = {batch, 1, 8, 64};
        operands.at(tensor).strides = {batchStride, 64, 64, 1};
    }
    auto verdict = denseStridesAdmit(problem, operands);
    ASSERT_FALSE(verdict.ok);
    EXPECT_EQ(verdict.reason, DenseStrideReason::SIZE_LIMIT) << verdict.detail;
    EXPECT_NE(verdict.detail.find("64 bits"), std::string::npos) << verdict.detail;

    // One batch fewer puts the last byte at 2^63 - 512 - 2 * sB + 1024, below the limit.
    problem.batch = batch - 1;
    for(auto& operand : operands)
    {
        operand.dims.at(0) = batch - 1;
    }
    verdict = denseStridesAdmit(problem, operands);
    EXPECT_TRUE(verdict.ok) << verdict.detail;
}

TEST(TestAttentionDenseStrides, AdmitsOnlyAMultipleOfSixteenBytesAsABaseAlignment)
{
    const auto problem = baseProblem();
    for(const int64_t align : {int64_t{16}, int64_t{32}, int64_t{48}, int64_t{256}})
    {
        auto operands = packedOperands(problem);
        operands.at(2).align = align;
        EXPECT_TRUE(denseStridesAdmit(problem, operands).ok) << align;
    }
    for(const int64_t align : {int64_t{1}, int64_t{8}, int64_t{17}, int64_t{24}, int64_t{40}})
    {
        auto operands = packedOperands(problem);
        operands.at(2).align = align;
        const auto verdict = denseStridesAdmit(problem, operands);
        ASSERT_FALSE(verdict.ok) << align;
        EXPECT_EQ(verdict.reason, DenseStrideReason::POINTER_ALIGNMENT) << align;
    }
}

TEST(TestAttentionDenseStrides, DeclinesAStrideTooBigForTheOverlapSpanWithoutWrapping)
{
    // Two axes of extent 2^40 with strides 2^60 and 2^61: the first span, 1 + (2^40 - 1) *
    // 2^60, passes int64_t. The second axis then overlaps, in the arithmetic the Python
    // does with unbounded integers. Both are multiples of 8 and positive, so only the span
    // can decline it.
    DenseStrideProblem problem;
    problem.batch = int64_t{1} << 40;
    problem.seqLenQ = 8;
    problem.seqLenKv = 8;
    problem.numQueryHeads = int64_t{1} << 40;
    problem.numKvHeads = int64_t{1} << 40;
    problem.headSize = 64;
    problem.blockN = 32;
    std::array<DenseOperand, 4> operands;
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto dims = denseOperandDims(problem, tensor);
        operands.at(tensor).dims.assign(dims.begin(), dims.end());
        operands.at(tensor).strides = {int64_t{1} << 61, int64_t{1} << 60, 64, 1};
    }
    const auto verdict = denseStridesAdmit(problem, operands);
    ASSERT_FALSE(verdict.ok);
    EXPECT_EQ(verdict.reason, DenseStrideReason::OVERLAP) << verdict.detail;
}

TEST(TestAttentionDenseStrides, AdmitsPackedOperandsWithTheCanonicalLaunchStrides)
{
    // The ordinary served graph: packed BSHD at the base problem.
    const auto problem = baseProblem();
    const auto verdict = denseStridesAdmit(problem, packedOperands(problem));
    ASSERT_TRUE(verdict.ok) << verdict.detail;
    EXPECT_EQ(verdict.strides, *packedLaunchStrides(problem));
}

TEST(TestAttentionDenseStrides, TheDeclineOrderIsTheReasonOrder)
{
    // A graph that breaks two rules is declined for the earlier one: here the head-dim
    // stride (2) and a misaligned base.
    const auto problem = baseProblem();
    auto operands = packedOperands(problem);
    operands.at(1).strides.at(3) = 2;
    operands.at(2).align = 4;
    auto verdict = denseStridesAdmit(problem, operands);
    ASSERT_FALSE(verdict.ok);
    EXPECT_EQ(verdict.reason, DenseStrideReason::HEAD_DIM_STRIDE);

    // Among operands of one check the first one wins: q before o.
    operands = packedOperands(problem);
    operands.at(0).align = 4;
    operands.at(3).strides.at(0) += 1;
    verdict = denseStridesAdmit(problem, operands);
    ASSERT_FALSE(verdict.ok);
    EXPECT_EQ(verdict.reason, DenseStrideReason::STRIDE_ALIGNMENT);
    EXPECT_NE(verdict.detail.find("o batch"), std::string::npos) << verdict.detail;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

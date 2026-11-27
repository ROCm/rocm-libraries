// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
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

// =============================================================================
// The launch-time output check: the span of each operand, and whether O's meets an input's
// =============================================================================

namespace
{

/// The base problem with as many KV heads as query heads, so Q, K, V and O can share a
/// [B, S, slots, H, D] buffer: B2, H 8, S 64, D128.
DenseStrideProblem fusableProblem()
{
    auto problem = baseProblem();
    problem.numKvHeads = problem.numQueryHeads;
    return problem;
}

/// The strides of every operand cut out of one [B, S, @p slots, H, D] buffer of @p problem.
AttentionDenseStrides slotStrides(const DenseStrideProblem& problem, int64_t slots)
{
    const int64_t row = slots * problem.numQueryHeads * problem.headSize;
    AttentionDenseStrides strides{};
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        strides.at(denseStrideIndex(tensor, 0)) = problem.seqLenQ * row;
        strides.at(denseStrideIndex(tensor, 1)) = problem.headSize;
        strides.at(denseStrideIndex(tensor, 2)) = row;
    }
    return strides;
}

} // namespace

TEST(TestAttentionDenseStrides, AnOperandSpansTheBytesUpToItsLastElement)
{
    // Packed: exactly the tensor, B * S * H * D * 2 bytes. Q and O over 8 heads, K and V over 2.
    const auto problem = baseProblem();
    const auto packed = *packedLaunchStrides(problem);
    EXPECT_EQ(denseOperandSpanBytes(problem, packed, 0), 2 * 64 * 8 * 128 * 2);
    EXPECT_EQ(denseOperandSpanBytes(problem, packed, 1), 2 * 64 * 2 * 128 * 2);
    EXPECT_EQ(denseOperandSpanBytes(problem, packed, 3), 2 * 64 * 8 * 128 * 2);

    // One slot of a three-slot buffer: up to the last row's last element, short of the two
    // slots after it, 2 * ((B - 1) * 3SHD + (H - 1) * D + (S - 1) * 3HD + D) bytes.
    const auto fused = fusableProblem();
    const auto strides = slotStrides(fused, 3);
    const int64_t row = int64_t{3} * 8 * 128;
    EXPECT_EQ(denseOperandSpanBytes(fused, strides, 1),
              2 * (64 * row + int64_t{7} * 128 + 63 * row + 128));

    // Extent-1 axes reach nothing: one batch, one query row, one head is one row of D.
    auto single = baseProblem();
    single.batch = 1;
    single.seqLenQ = 1;
    single.numQueryHeads = 1;
    EXPECT_EQ(denseOperandSpanBytes(single, *packedLaunchStrides(single), 0), 2 * 128);

    // Past int64_t is reported, not wrapped.
    auto huge = baseProblem();
    huge.batch = int64_t{1} << 40;
    AttentionDenseStrides far = packed;
    far.at(denseStrideIndex(0, 0)) = int64_t{1} << 30;
    EXPECT_FALSE(denseOperandSpanBytes(huge, far, 0).has_value());
}

TEST(TestAttentionDenseStrides, ByteRangesIntersectOnlyWhereTheyShareAByte)
{
    // Adjacent, either way round, share none; one byte more does.
    EXPECT_FALSE(denseByteRangesIntersect(0x1000, 16, 0x1010, 16));
    EXPECT_FALSE(denseByteRangesIntersect(0x1010, 16, 0x1000, 16));
    EXPECT_TRUE(denseByteRangesIntersect(0x1000, 17, 0x1010, 16));
    EXPECT_TRUE(denseByteRangesIntersect(0x1010, 16, 0x1000, 17));
    // Equal, and one inside the other.
    EXPECT_TRUE(denseByteRangesIntersect(0x1000, 16, 0x1000, 16));
    EXPECT_TRUE(denseByteRangesIntersect(0x1000, 4096, 0x1800, 16));
    // An empty range shares nothing, even at the other's base.
    EXPECT_FALSE(denseByteRangesIntersect(0x1000, 0, 0x1000, 16));
    // A range that ends past the address space does not wrap around to low addresses.
    constexpr auto TOP = std::numeric_limits<std::uintptr_t>::max() - 7;
    EXPECT_FALSE(denseByteRangesIntersect(TOP, 64, 0, 64));
    EXPECT_TRUE(denseByteRangesIntersect(TOP, 64, TOP + 4, 1));
}

TEST(TestAttentionDenseStrides, ReportsTheFirstInputWhoseBytesTheOutputShares)
{
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    const auto span = static_cast<std::uintptr_t>(*denseOperandSpanBytes(problem, packed, 0));
    constexpr std::uintptr_t BASE = 0x7f0000000000;

    // Packed tensors in buffers of their own, side by side: no byte shared.
    EXPECT_FALSE(denseOutputOverlapsInput(
                     problem, packed, {BASE, BASE + span, BASE + 2 * span, BASE + 3 * span})
                     .has_value());
    // The output passed as one of the inputs.
    for(std::size_t input = 0; input < 3; ++input)
    {
        std::array<std::uintptr_t, 4> bases{BASE, BASE + span, BASE + 2 * span, 0};
        bases.at(3) = bases.at(input);
        EXPECT_EQ(denseOutputOverlapsInput(problem, packed, bases), input) << input;
    }
    // The output starting one element inside the value's last bytes.
    EXPECT_EQ(denseOutputOverlapsInput(
                  problem, packed, {BASE, BASE + span, BASE + 2 * span, BASE + 3 * span - 2}),
              std::size_t{2});
    // The output ending on the byte before the query, and one element inside it.
    EXPECT_FALSE(denseOutputOverlapsInput(
                     problem, packed, {BASE + span, BASE + 2 * span, BASE + 3 * span, BASE})
                     .has_value());
    EXPECT_EQ(denseOutputOverlapsInput(
                  problem, packed, {BASE + span - 2, BASE + 4 * span, BASE + 5 * span, BASE}),
              std::size_t{0});
}

TEST(TestAttentionDenseStrides, ComparesTheSpansOfViewsIntoOneFusedBuffer)
{
    const auto problem = fusableProblem();
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    const std::uintptr_t slot = std::uintptr_t{8} * 128 * 2;

    // Q, K and V cut out of a [B, S, 3, H, D] buffer share its storage: the inputs may, since
    // the kernels only read them. An output of its own, after the buffer, shares nothing.
    const auto threeSlots = slotStrides(problem, 3);
    const auto buffer = static_cast<std::uintptr_t>(2 * 64 * 3 * 8 * 128 * 2);
    auto strides = threeSlots;
    const auto packed = *packedLaunchStrides(problem);
    for(std::size_t axis = 0; axis < 3; ++axis)
    {
        strides.at(denseStrideIndex(3, axis)) = packed.at(denseStrideIndex(3, axis));
    }
    EXPECT_FALSE(denseOutputOverlapsInput(
                     problem, strides, {BASE, BASE + slot, BASE + 2 * slot, BASE + buffer})
                     .has_value());

    // The output as a fourth slot of the same buffer: its rows interleave with the inputs'
    // without sharing an element, and the spans meet, so it is reported (the conservative
    // answer), against the query first.
    const auto fourSlots = slotStrides(problem, 4);
    EXPECT_EQ(denseOutputOverlapsInput(
                  problem, fourSlots, {BASE, BASE + slot, BASE + 2 * slot, BASE + 3 * slot}),
              std::size_t{0});

    // The output written over the key's slot of the three-slot buffer: inside the query's span
    // too, which is asked first.
    EXPECT_EQ(denseOutputOverlapsInput(
                  problem, threeSlots, {BASE, BASE + slot, BASE + 2 * slot, BASE + slot}),
              std::size_t{0});
}

TEST(TestAttentionDenseStrides, SkipsAnOperandWithoutABuffer)
{
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    // No output buffer: nothing to compare. No query buffer: the key it sits on is reported.
    EXPECT_FALSE(denseOutputOverlapsInput(problem, packed, {BASE, BASE, BASE, 0}).has_value());
    EXPECT_EQ(denseOutputOverlapsInput(problem, packed, {0, BASE, BASE + (1U << 20), BASE}),
              std::size_t{1});
}

namespace
{

/// The bias strides (batch, query head, query row) of a packed FP32 [B, Hq, Sq, Skv] bias.
std::array<int64_t, 3> packedBiasStrides(const DenseStrideProblem& problem)
{
    return {problem.numQueryHeads * problem.seqLenQ * problem.seqLenKv,
            problem.seqLenQ * problem.seqLenKv,
            problem.seqLenKv};
}

/// Where denseStatsOverlapsOperand reports the additive bias.
constexpr std::size_t BIAS_OPERAND = 4;

} // namespace

TEST(TestAttentionDenseStrides, TheStatsOutputSpansItsWholePackedTensor)
{
    // FP32 [B, Hq, Sq, 1]: 4 * 2 * 8 * 64 bytes, whatever the K/V heads.
    EXPECT_EQ(denseStatsSpanBytes(baseProblem()), 4 * 2 * 8 * 64);
    auto single = baseProblem();
    single.batch = 1;
    single.numQueryHeads = 1;
    single.seqLenQ = 1;
    EXPECT_EQ(denseStatsSpanBytes(single), 4);

    // Past int64_t is reported, not wrapped: in the element count (2^60 * 8 rows), and in the
    // byte count of 2^61 elements. 2^60 elements are 2^62 bytes, which fit.
    auto huge = baseProblem();
    huge.batch = int64_t{1} << 60;
    EXPECT_FALSE(denseStatsSpanBytes(huge).has_value());
    huge.batch = int64_t{1} << 52;
    EXPECT_FALSE(denseStatsSpanBytes(huge).has_value());
    huge.batch = int64_t{1} << 51;
    EXPECT_EQ(denseStatsSpanBytes(huge), int64_t{1} << 62);
}

TEST(TestAttentionDenseStrides, TheBiasSpansUpToTheLastElementTheKernelReads)
{
    const auto problem = baseProblem();
    // Packed [B, Hq, Sq, Skv]: exactly the tensor, 4 * B * Hq * Sq * Skv bytes.
    EXPECT_EQ(denseBiasSpanBytes(problem, packedBiasStrides(problem)), 4 * 2 * 8 * 64 * 64);
    // The head axis broadcast through stride 0: one head's worth per batch.
    EXPECT_EQ(denseBiasSpanBytes(problem, {int64_t{64} * 64, 0, 64}), 4 * 2 * 64 * 64);
    // Every leading axis broadcast: one row of Skv.
    EXPECT_EQ(denseBiasSpanBytes(problem, {0, 0, 0}), 4 * 64);
    // Pitched rows end at the last row's last key, short of its padding:
    // 4 * ((B - 1) * sB + (Hq - 1) * sH + (Sq - 1) * sS + Skv).
    EXPECT_EQ(denseBiasSpanBytes(problem, {100000, 8000, 96}),
              4 * (100000 + 7 * 8000 + 63 * 96 + 64));

    // Past int64_t is reported, not wrapped: in a product, and in the byte count.
    auto huge = baseProblem();
    huge.batch = int64_t{1} << 40;
    EXPECT_FALSE(denseBiasSpanBytes(huge, {int64_t{1} << 30, 0, 0}).has_value());
    EXPECT_FALSE(denseBiasSpanBytes(problem, {int64_t{1} << 62, 0, 0}).has_value());
}

TEST(TestAttentionDenseStrides, ReportsTheFirstBufferWhoseBytesTheStatsOutputShares)
{
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    const auto biasStrides = packedBiasStrides(problem);
    const auto span = static_cast<std::uintptr_t>(*denseOperandSpanBytes(problem, packed, 0));
    const auto lseSpan = static_cast<std::uintptr_t>(*denseStatsSpanBytes(problem));
    const auto biasSpan = static_cast<std::uintptr_t>(*denseBiasSpanBytes(problem, biasStrides));
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    const std::array<std::uintptr_t, 4> bases{BASE, BASE + span, BASE + 2 * span, BASE + 3 * span};
    const std::uintptr_t bias = BASE + 4 * span;

    // Every buffer of its own, side by side: the LSE on the byte after the bias's last one, and on
    // the byte after the output's last one with the bias elsewhere.
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, bases, bias + biasSpan, bias, biasStrides)
            .has_value());
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, bases, BASE + 4 * span, bias + span, biasStrides)
            .has_value());
    // The LSE ending on the byte before the query.
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, bases, BASE - lseSpan, bias, biasStrides)
            .has_value());

    // The LSE passed as each operand, then as the bias.
    for(std::size_t operand = 0; operand < 4; ++operand)
    {
        EXPECT_EQ(
            denseStatsOverlapsOperand(problem, packed, bases, bases.at(operand), bias, biasStrides),
            operand)
            << operand;
    }
    EXPECT_EQ(denseStatsOverlapsOperand(problem, packed, bases, bias, bias, biasStrides),
              BIAS_OPERAND);
    // Inside the query, one float into the bias's end, and one float into the query's start.
    EXPECT_EQ(denseStatsOverlapsOperand(problem, packed, bases, BASE + 1024, bias, biasStrides),
              std::size_t{0});
    EXPECT_EQ(
        denseStatsOverlapsOperand(problem, packed, bases, bias + biasSpan - 4, bias, biasStrides),
        BIAS_OPERAND);
    EXPECT_EQ(
        denseStatsOverlapsOperand(problem, packed, bases, BASE - lseSpan + 4, bias, biasStrides),
        std::size_t{0});
}

TEST(TestAttentionDenseStrides, ComparesTheStatsOutputWithSpansNotElements)
{
    // A query cut out of a [B, S, 3, H, D] buffer leaves two slots between its rows. An LSE of
    // exactly that size placed there shares no element with the query, but lies inside its span,
    // so it is reported (the conservative answer).
    const auto problem = fusableProblem();
    const std::uintptr_t slot = std::uintptr_t{8} * 128 * 2;
    ASSERT_EQ(denseStatsSpanBytes(problem), static_cast<int64_t>(2 * slot));
    auto strides = *packedLaunchStrides(problem);
    const auto threeSlots = slotStrides(problem, 3);
    for(std::size_t axis = 0; axis < 3; ++axis)
    {
        strides.at(denseStrideIndex(0, axis)) = threeSlots.at(denseStrideIndex(0, axis));
    }
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    constexpr std::uintptr_t FAR = std::uintptr_t{1} << 30;
    const std::array<std::uintptr_t, 4> bases{BASE, BASE + FAR, BASE + 2 * FAR, BASE + 3 * FAR};
    EXPECT_EQ(denseStatsOverlapsOperand(problem, strides, bases, BASE + slot, 0, std::nullopt),
              std::size_t{0});
}

TEST(TestAttentionDenseStrides, MeasuresABroadcastBiasByTheElementsTheKernelReads)
{
    // A [1, 1, Sq, Skv] bias reads Sq rows of Skv: an LSE on the byte after them shares nothing,
    // one float earlier it does, and with the batch axis stepped the same LSE lies inside.
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    constexpr std::uintptr_t BIAS = 0x7f0000000000;
    const std::array<std::uintptr_t, 4> far{BIAS + (std::uintptr_t{1} << 30),
                                            BIAS + (std::uintptr_t{2} << 30),
                                            BIAS + (std::uintptr_t{3} << 30),
                                            BIAS + (std::uintptr_t{4} << 30)};
    const std::array<int64_t, 3> broadcast{0, 0, 64};
    const auto end = static_cast<std::uintptr_t>(4 * 64 * 64);
    ASSERT_EQ(denseBiasSpanBytes(problem, broadcast), 4 * 64 * 64);
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, far, BIAS + end, BIAS, broadcast).has_value());
    EXPECT_EQ(denseStatsOverlapsOperand(problem, packed, far, BIAS + end - 4, BIAS, broadcast),
              BIAS_OPERAND);
    EXPECT_EQ(denseStatsOverlapsOperand(problem,
                                        packed,
                                        far,
                                        BIAS + end,
                                        BIAS,
                                        std::array<int64_t, 3>{int64_t{64} * 64, 0, 64}),
              BIAS_OPERAND);
}

TEST(TestAttentionDenseStrides, SkipsABufferTheStatsOutputHasNoAddressFor)
{
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    const auto biasStrides = packedBiasStrides(problem);
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    // No LSE buffer: nothing to compare, though every other buffer sits at one address.
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, {BASE, BASE, BASE, BASE}, 0, BASE, biasStrides)
            .has_value());
    // No query buffer: the key the LSE sits on is reported.
    EXPECT_EQ(denseStatsOverlapsOperand(problem,
                                        packed,
                                        {0, BASE, BASE + (1U << 20), BASE + (2U << 20)},
                                        BASE,
                                        0,
                                        std::nullopt),
              std::size_t{1});
    // No bias buffer, or no bias strides (a kernel that reads no bias): the bias is not compared.
    const std::array<std::uintptr_t, 4> elsewhere{0, 0, 0, 0};
    EXPECT_FALSE(
        denseStatsOverlapsOperand(problem, packed, elsewhere, BASE, 0, biasStrides).has_value());
    EXPECT_FALSE(denseStatsOverlapsOperand(problem, packed, elsewhere, BASE, BASE, std::nullopt)
                     .has_value());
}

TEST(TestAttentionDenseStrides, CountsAStatsOrBiasSpanPastInt64AsShared)
{
    constexpr std::uintptr_t BASE = 0x7f0000000000;
    constexpr std::uintptr_t FAR = std::uintptr_t{1} << 40;
    const auto problem = fusableProblem();
    const auto packed = *packedLaunchStrides(problem);
    // A bias whose span passes int64_t is reported wherever it lies.
    EXPECT_EQ(denseStatsOverlapsOperand(problem,
                                        packed,
                                        {0, 0, 0, 0},
                                        BASE,
                                        BASE + FAR,
                                        std::array<int64_t, 3>{int64_t{1} << 62, 0, 0}),
              BIAS_OPERAND);
    // An LSE whose span passes int64_t is reported against the first buffer there is, though
    // that buffer's own span (zero strides: one row of D) lies far from the LSE's base.
    auto huge = problem;
    huge.batch = int64_t{1} << 60;
    ASSERT_EQ(denseOperandSpanBytes(huge, AttentionDenseStrides{}, 2), 2 * 128);
    EXPECT_EQ(denseStatsOverlapsOperand(
                  huge, AttentionDenseStrides{}, {0, 0, BASE + FAR, 0}, BASE, 0, std::nullopt),
              std::size_t{2});
    EXPECT_EQ(denseStatsOverlapsOperand(
                  huge, packed, {0, 0, 0, 0}, BASE, BASE + FAR, std::array<int64_t, 3>{0, 0, 0}),
              BIAS_OPERAND);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

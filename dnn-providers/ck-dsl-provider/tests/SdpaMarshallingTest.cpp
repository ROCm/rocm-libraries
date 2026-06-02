// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "adapters/sdpa/SdpaMarshalling.hpp"

namespace {

using ck_dsl_provider::blockTableStride;
using ck_dsl_provider::marshalDenseDegenerate;
using ck_dsl_provider::marshalRealPaged;
using ck_dsl_provider::marshalVarlen;
using ck_dsl_provider::SdpaMarshalInputs;
using ck_dsl_provider::SdpaMarshalledArrays;

// ---------------------------------------------------------------------------
// blockTableStride: ceil(max_seqlen_k / block_size).
// ---------------------------------------------------------------------------

TEST(SdpaMarshallingStride, ExactDivisionGivesQuotient) {
    // 128 / 32 == 4 exactly.
    EXPECT_EQ(blockTableStride(/*max_seqlen_k=*/128, /*block_size=*/32), 4);
    // 64 / 16 == 4.
    EXPECT_EQ(blockTableStride(64, 16), 4);
}

TEST(SdpaMarshallingStride, NonDivisibleRoundsUp) {
    // 100 / 32 -> ceil == 4 (3.125 -> 4).
    EXPECT_EQ(blockTableStride(100, 32), 4);
    // 17 / 16 -> ceil == 2.
    EXPECT_EQ(blockTableStride(17, 16), 2);
    // 1 / 64 -> ceil == 1.
    EXPECT_EQ(blockTableStride(1, 64), 1);
}

TEST(SdpaMarshallingStride, ZeroSeqlenIsZeroStride) {
    EXPECT_EQ(blockTableStride(0, 32), 0);
}

TEST(SdpaMarshallingStride, NonPositiveBlockSizeThrows) {
    EXPECT_THROW(blockTableStride(128, 0), std::invalid_argument);
    EXPECT_THROW(blockTableStride(128, -16), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Dense-degenerate marshalling.
// ---------------------------------------------------------------------------

TEST(SdpaMarshallingDense, BlockTableConsecutiveAndStrideExact) {
    // B=2, block_size=32, Skv=64 -> stride = 64/32 = 2; 2 seqs * 2 blocks.
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 32;
    in.max_seqlen_k = 64;
    in.seqlen_q = 16;
    in.seqlen_k = 64;

    const SdpaMarshalledArrays out = marshalDenseDegenerate(in);

    EXPECT_EQ(out.block_table_stride, 2);
    // Consecutive physical blocks, row-major: seq0 -> {0,1}, seq1 -> {2,3}.
    const std::vector<std::int32_t> expectedTable = {0, 1, 2, 3};
    EXPECT_EQ(out.block_tables, expectedTable);

    // cu_seqlens_q = [0, Sq, 2Sq] (length B+1).
    const std::vector<std::int32_t> expectedCu = {0, 16, 32};
    EXPECT_EQ(out.cu_seqlens_q, expectedCu);

    // seqused_k = [Skv, Skv].
    const std::vector<std::int32_t> expectedSeqused = {64, 64};
    EXPECT_EQ(out.seqused_k, expectedSeqused);
}

TEST(SdpaMarshallingDense, NonDivisibleSkvCeilsStrideAndSizesTable) {
    // B=3, block_size=32, Skv=100 -> stride = ceil(100/32) = 4.
    SdpaMarshalInputs in;
    in.num_seqs = 3;
    in.block_size = 32;
    in.max_seqlen_k = 100;
    in.seqlen_q = 10;
    in.seqlen_k = 100;

    const SdpaMarshalledArrays out = marshalDenseDegenerate(in);

    EXPECT_EQ(out.block_table_stride, 4);
    // 3 seqs * 4 blocks = 12 consecutive entries 0..11.
    ASSERT_EQ(out.block_tables.size(), 12u);
    for (std::int32_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
    // cu_seqlens_q = [0, 10, 20, 30].
    const std::vector<std::int32_t> expectedCu = {0, 10, 20, 30};
    EXPECT_EQ(out.cu_seqlens_q, expectedCu);
    // seqused_k = [100, 100, 100].
    EXPECT_EQ(out.seqused_k, (std::vector<std::int32_t>{100, 100, 100}));
}

TEST(SdpaMarshallingDense, RejectsBadInputs) {
    SdpaMarshalInputs in;
    in.num_seqs = 0;  // invalid
    in.block_size = 32;
    in.max_seqlen_k = 64;
    EXPECT_THROW(marshalDenseDegenerate(in), std::invalid_argument);

    in.num_seqs = 2;
    in.block_size = 0;  // invalid
    EXPECT_THROW(marshalDenseDegenerate(in), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Varlen marshalling.
// ---------------------------------------------------------------------------

TEST(SdpaMarshallingVarlen, PrefixSumAndSequsedFromLengths) {
    // 3 packed sequences with distinct q/k lengths.
    SdpaMarshalInputs in;
    in.num_seqs = 3;
    in.block_size = 16;
    in.max_seqlen_k = 48;  // stride = 48/16 = 3.

    const std::vector<std::int32_t> qLens = {4, 7, 2};
    const std::vector<std::int32_t> kLens = {16, 48, 32};

    const SdpaMarshalledArrays out = marshalVarlen(in, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 3);
    // cu_seqlens_q = prefix sum: [0, 4, 11, 13].
    const std::vector<std::int32_t> expectedCu = {0, 4, 11, 13};
    EXPECT_EQ(out.cu_seqlens_q, expectedCu);
    // seqused_k mirrors kLens.
    EXPECT_EQ(out.seqused_k, kLens);
    // Block table sized to num_seqs * stride = 9, consecutive 0..8.
    ASSERT_EQ(out.block_tables.size(), 9u);
    for (std::int32_t i = 0; i < 9; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
}

TEST(SdpaMarshallingVarlen, RejectsWrongLengthVectors) {
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 16;
    in.max_seqlen_k = 32;
    // q_lens too short.
    EXPECT_THROW(marshalVarlen(in, {4}, {16, 16}), std::invalid_argument);
    // k_lens too long.
    EXPECT_THROW(marshalVarlen(in, {4, 5}, {16, 16, 16}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Real-paged marshalling.
// ---------------------------------------------------------------------------

TEST(SdpaMarshallingRealPaged, ValidatesStrideAndPassesTableThrough) {
    // 2 seqs, block_size 32, max_seqlen_k 96 -> stride = ceil(96/32) = 3.
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 32;
    in.max_seqlen_k = 96;

    // Caller-owned physical table (non-consecutive page ids), size 2*3 = 6.
    const std::vector<std::int32_t> table = {7, 3, 11, 0, 9, 5};
    const std::vector<std::int32_t> qLens = {12, 20};
    const std::vector<std::int32_t> kLens = {96, 64};

    const SdpaMarshalledArrays out = marshalRealPaged(in, table, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 3);
    EXPECT_EQ(out.block_tables, table);  // passthrough.
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 12, 32}));
    EXPECT_EQ(out.seqused_k, kLens);
}

TEST(SdpaMarshallingRealPaged, RejectsTableSizeMismatch) {
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 32;
    in.max_seqlen_k = 96;  // stride 3 -> expect 6 entries.

    const std::vector<std::int32_t> badTable = {0, 1, 2, 3};  // size 4, not 6.
    EXPECT_THROW(marshalRealPaged(in, badTable, {1, 1}, {1, 1}), std::invalid_argument);
}

}  // namespace

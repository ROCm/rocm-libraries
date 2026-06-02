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

// ---------------------------------------------------------------------------
// Runtime layout matrix (Phase 3). These axes are excluded from the codegen
// generation matrix because they change only the host-prepared launch arrays
// (block_tables / cu_seqlens_q / seqused_k) and the block_table_stride scalar
// -- the kernel BINARY and 18-arg ABI are identical. They are covered here,
// against the pure marshalling functions, on the CPU.
// ---------------------------------------------------------------------------

// --- Ceil stride when block_size does not divide Skv, for {16,32,64}. ------

TEST(SdpaMarshallingDenseCeilStride, BlockSize16NonDivisible) {
    // B=2, block_size=16, Skv=50 -> stride = ceil(50/16) = 4 (50/16=3.125).
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 16;
    in.max_seqlen_k = 50;
    in.seqlen_q = 8;
    in.seqlen_k = 50;

    const SdpaMarshalledArrays out = marshalDenseDegenerate(in);

    EXPECT_EQ(out.block_table_stride, 4);
    // 2 seqs * 4 blocks = 8 consecutive entries 0..7.
    ASSERT_EQ(out.block_tables.size(), 8u);
    for (std::int32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 8, 16}));
    EXPECT_EQ(out.seqused_k, (std::vector<std::int32_t>{50, 50}));
}

TEST(SdpaMarshallingDenseCeilStride, BlockSize32NonDivisible) {
    // B=2, block_size=32, Skv=70 -> stride = ceil(70/32) = 3 (70/32=2.1875).
    SdpaMarshalInputs in;
    in.num_seqs = 2;
    in.block_size = 32;
    in.max_seqlen_k = 70;
    in.seqlen_q = 4;
    in.seqlen_k = 70;

    const SdpaMarshalledArrays out = marshalDenseDegenerate(in);

    EXPECT_EQ(out.block_table_stride, 3);
    ASSERT_EQ(out.block_tables.size(), 6u);
    for (std::int32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
}

TEST(SdpaMarshallingDenseCeilStride, BlockSize64NonDivisible) {
    // B=3, block_size=64, Skv=200 -> stride = ceil(200/64) = 4 (200/64=3.125).
    SdpaMarshalInputs in;
    in.num_seqs = 3;
    in.block_size = 64;
    in.max_seqlen_k = 200;
    in.seqlen_q = 5;
    in.seqlen_k = 200;

    const SdpaMarshalledArrays out = marshalDenseDegenerate(in);

    EXPECT_EQ(out.block_table_stride, 4);
    // 3 seqs * 4 blocks = 12 consecutive entries.
    ASSERT_EQ(out.block_tables.size(), 12u);
    for (std::int32_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 5, 10, 15}));
}

// --- Varlen with ragged and zero-length sequences. -------------------------

TEST(SdpaMarshallingVarlenRagged, ZeroLengthQAndKSequences) {
    // 4 packed sequences; seq 1 has zero query tokens, seq 3 has zero KV.
    SdpaMarshalInputs in;
    in.num_seqs = 4;
    in.block_size = 32;
    in.max_seqlen_k = 96;  // stride = ceil(96/32) = 3.

    const std::vector<std::int32_t> qLens = {5, 0, 12, 3};
    const std::vector<std::int32_t> kLens = {64, 32, 96, 0};

    const SdpaMarshalledArrays out = marshalVarlen(in, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 3);
    // Prefix sum tolerates the zero-length query: [0, 5, 5, 17, 20].
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 5, 5, 17, 20}));
    // seqused_k mirrors kLens, including the zero-KV sequence.
    EXPECT_EQ(out.seqused_k, kLens);
    // Block table sized to num_seqs * stride = 12, consecutive.
    ASSERT_EQ(out.block_tables.size(), 12u);
    for (std::int32_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out.block_tables[static_cast<std::size_t>(i)], i);
    }
}

TEST(SdpaMarshallingVarlenRagged, LargeNumSeqsRaggedLengths) {
    // 8 sequences with monotonically varying lengths; larger num_seqs.
    SdpaMarshalInputs in;
    in.num_seqs = 8;
    in.block_size = 16;
    in.max_seqlen_k = 64;  // stride = 64/16 = 4.

    const std::vector<std::int32_t> qLens = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::int32_t> kLens = {16, 16, 32, 32, 48, 48, 64, 64};

    const SdpaMarshalledArrays out = marshalVarlen(in, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 4);
    // Prefix sum of 1..8 = [0,1,3,6,10,15,21,28,36].
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 1, 3, 6, 10, 15, 21, 28, 36}));
    EXPECT_EQ(out.seqused_k, kLens);
    // 8 seqs * 4 blocks = 32 entries.
    ASSERT_EQ(out.block_tables.size(), 32u);
}

// --- Real-paged with a multi-block, non-consecutive physical layout. -------

TEST(SdpaMarshallingRealPagedMultiBlock, MultiBlockNonConsecutiveStride) {
    // 3 seqs, block_size 64, max_seqlen_k 256 -> stride = 256/64 = 4. Each
    // sequence's KV spans up to 4 physical cache blocks, scattered.
    SdpaMarshalInputs in;
    in.num_seqs = 3;
    in.block_size = 64;
    in.max_seqlen_k = 256;

    // Caller-owned physical table, size 3*4 = 12, deliberately scattered.
    const std::vector<std::int32_t> table = {
        12, 3, 88, 1,   // seq 0
        9,  9, 0,  41,  // seq 1 (a repeated page id is legal)
        7,  6, 5,  4,   // seq 2
    };
    const std::vector<std::int32_t> qLens = {64, 128, 32};
    const std::vector<std::int32_t> kLens = {256, 192, 100};

    const SdpaMarshalledArrays out = marshalRealPaged(in, table, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 4);
    EXPECT_EQ(out.block_tables, table);  // verbatim passthrough.
    EXPECT_EQ(out.cu_seqlens_q, (std::vector<std::int32_t>{0, 64, 192, 224}));
    EXPECT_EQ(out.seqused_k, kLens);
}

TEST(SdpaMarshallingRealPagedMultiBlock, LargeNumSeqsTablePassthrough) {
    // 16 seqs, block_size 32, max_seqlen_k 128 -> stride = 4; table 16*4 = 64.
    SdpaMarshalInputs in;
    in.num_seqs = 16;
    in.block_size = 32;
    in.max_seqlen_k = 128;

    std::vector<std::int32_t> table;
    table.reserve(64);
    // Reverse-ordered physical ids to prove no implicit consecutive
    // assumption is baked in.
    for (std::int32_t i = 63; i >= 0; --i) {
        table.push_back(i);
    }
    std::vector<std::int32_t> qLens(16, 1);  // pure decode (one q token each).
    std::vector<std::int32_t> kLens(16, 128);

    const SdpaMarshalledArrays out = marshalRealPaged(in, table, qLens, kLens);

    EXPECT_EQ(out.block_table_stride, 4);
    EXPECT_EQ(out.block_tables, table);
    // cu_seqlens_q for 16 unit-length queries = [0,1,2,...,16].
    ASSERT_EQ(out.cu_seqlens_q.size(), 17u);
    for (std::int32_t i = 0; i <= 16; ++i) {
        EXPECT_EQ(out.cu_seqlens_q[static_cast<std::size_t>(i)], i);
    }
    EXPECT_EQ(out.seqused_k, kLens);
}

}  // namespace

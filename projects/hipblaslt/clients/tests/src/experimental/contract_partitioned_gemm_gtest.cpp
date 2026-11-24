// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CONTRACT TESTS for experimental feature PARTITIONED_GEMM.
//
// These tests define the promised behavior of the feature, as documented in
// hipblaslt-experimental-partitioned-gemm.h. Adding a test is free. Changing or deleting one is a
// breaking change: tools/scripts/check_experimental_api.py fails the PR unless it also bumps
// HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM, the feature's revision. The same goes for the helpers
// in contract_partitioned_gemm_helpers.hpp.
//
// The feature is supported on gfx1250 only, so every CI lane checks part of the contract: on
// gfx1250 the tests of its behavior run, and on any other architecture NotSupportedOffGfx1250
// checks that setProblem refuses.

#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#include "contract_partitioned_gemm_helpers.hpp"

using namespace partitioned_gemm_test;

TEST(pre_checkin_partitioned_gemm_contract, PartitionCountAndSliceRule)
{
    SKIP_UNLESS_GFX1250();
    auto parts = onDeviceZero(3);
    auto gemm  = makeGemm(parts);
    EXPECT_EQ(gemm.getPartitionCount(), 3u);

    // The first N-1 partitions take size/N, the last takes the rest.
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), /*m=*/10, kN),
              HIPBLAS_STATUS_SUCCESS);
    const int64_t wantOffset[3] = {0, 3, 6}, wantSize[3] = {3, 3, 4};
    for(size_t i = 0; i < 3; i++)
    {
        int64_t offset = -1, size = -1;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(offset, wantOffset[i]);
        EXPECT_EQ(size, wantSize[i]);
    }
    int64_t offset, size;
    EXPECT_EQ(gemm.getSlice(3, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, MSlicesRoundTo256BytesWithinOnePercent)
{
    SKIP_UNLESS_GFX1250();
    // fp16, lda = 128 (op T), ldc = ldd = 256: the M offset moves 2 bytes per row in C and D, so
    // the slice is rounded up to 128 rows. It is used only when the last slice is at most 1%
    // smaller than the others; otherwise the plain size/N split is kept.
    struct Case
    {
        size_t  parts;
        int64_t m;
        int64_t wantOffset[3], wantSize[3];
    };
    const Case cases[] = {{2, 1000, {0, 500, 0}, {500, 500, 0}}, // 512/488 is 4.7% off: plain
                          {3, 1000, {0, 333, 666}, {333, 333, 334}},
                          {2, 255, {0, 128, 0}, {128, 127, 0}}}; // 0.78% off: rounded
    for(auto const& c : cases)
    {
        auto              parts = onDeviceZero(c.parts);
        auto              gemm  = makeGemm(parts);
        Operands          op;
        ReplicatedBuffers buffers(parts, op);
        ASSERT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), c.m, kN),
                  HIPBLAS_STATUS_SUCCESS);
        for(size_t i = 0; i < c.parts; i++)
        {
            int64_t offset = -1, size = -1;
            ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(offset, c.wantOffset[i]) << c.parts << " partitions, slice " << i;
            EXPECT_EQ(size, c.wantSize[i]) << c.parts << " partitions, slice " << i;
        }
    }
}

TEST(pre_checkin_partitioned_gemm_contract, GetSliceNeedsAProblem)
{
    SKIP_UNLESS_GFX1250();
    auto    parts = onDeviceZero(2);
    auto    gemm  = makeGemm(parts);
    int64_t offset, size;
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, StorePeerDNeedsTwoDifferentDs)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // Two partitions with their own D: accepted. Whether a kernel exists is up to the heuristic.
    auto in       = buffers.inputs();
    in.storePeerD = true;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);

    // The D list in the other order: the peer of each partition is found by device.
    std::swap(in.d[0], in.d[1]);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);

    // Overlapping Ds: one partition would overwrite the other's slice.
    in            = buffers.inputs();
    in.storePeerD = true;
    in.d[1]       = static_cast<char*>(in.d[0]) + 64;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    in            = buffers.inputs();
    in.storePeerD = true;
    // The same D twice: there is no peer to write to.
    in.d[1] = in.d[0];
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, RejectsWrongPointerCounts)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // A Replicated matrix needs N pointers.
    auto in = buffers.inputs();
    in.a.pop_back();
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    // A single pointer is too few: every matrix needs one per partition.
    in = buffers.inputs();
    in.a.resize(1);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    // A null pointer is rejected.
    in      = buffers.inputs();
    in.d[1] = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, RejectsBadSizesAndScalars)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    EXPECT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs(), kM, 0),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), -1, kN),
              HIPBLAS_STATUS_INVALID_VALUE);
    // The split dimension must hold at least one index per partition.
    EXPECT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), 1, kN),
              HIPBLAS_STATUS_INVALID_VALUE);

    auto in  = buffers.inputs();
    in.alpha = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
    in      = buffers.inputs();
    in.beta = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, UserArgsAreNotSupportedYet)
{
    SKIP_UNLESS_GFX1250();
    auto parts = onDeviceZero(2);
    auto gemm  = makeGemm(parts);

    // Any non-null device user-args pointer is refused. Null pointers fall through to a
    // standard run, which then fails because no problem is set.
    int                dummy = 0;
    std::vector<void*> userArgs{&dummy, &dummy};
    EXPECT_EQ(gemm.run(userArgs, parts.streams), HIPBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(gemm.getDefaultValueForDeviceUserArguments(userArgs), HIPBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(gemm.run(std::vector<void*>(2, nullptr), parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_contract, TwoPartitionsOnOneDeviceMatchGemm)
{
    SKIP_UNLESS_GFX1250();
    // SPX mode: both partitions are on the same device, so their streams share it. Speed does
    // not matter there, the result must still be right.
    for(SplitDim dim : {SplitDim::N, SplitDim::M})
    {
        auto              parts = onDeviceZero(2);
        auto              gemm  = makeGemm(parts);
        Operands          op;
        ReplicatedBuffers buffers(parts, op);
        ASSERT_EQ(setReplicated(gemm, dim, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

        GemmPreference pref;
        pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
        gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
        ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
        if(heuristic.empty())
            GTEST_SKIP() << "no algorithm for the test problem on this device";

        ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, parts.streams),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
        parts.sync();

        auto ref = referenceGemm(op, parts.rawHandles()[0], 0);
        ASSERT_FALSE(ref.empty());
        expectSlicesMatch(gemm, parts, buffers, ref, dim);
    }
}

TEST(pre_checkin_partitioned_gemm_contract, RejectsWrongStreamAndWorkspaceCounts)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm for the test problem on this device";

    EXPECT_EQ(gemm.run({parts.streams[0]}), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.initialize(heuristic[0].algo, {parts.workspaces()[0]}, true, parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// One partition is a plain GEMM: it must match Gemm exactly.
TEST(pre_checkin_partitioned_gemm_contract, OnePartitionMatchesGemm)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(1);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm for the test problem on this device";

    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    auto ref = referenceGemm(op, parts.rawHandles()[0], 0);
    ASSERT_FALSE(ref.empty());
    expectSlicesMatch(gemm, parts, buffers, ref, SplitDim::N);
}

// PartitionedGemm is supported on gfx1250 only. On any other architecture setProblem must return
// NOT_SUPPORTED, with and without storePeerD, and leave the object without a problem. This runs
// on one device of any other architecture, so that CI lanes that are not gfx1250 cover it.
TEST(pre_checkin_partitioned_gemm_contract, NotSupportedOffGfx1250)
{
    if(isGfx1250())
        GTEST_SKIP() << "PartitionedGemm is supported on gfx1250";
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    EXPECT_EQ(gemm.getPartitionCount(), 2u);

    for(SplitDim dim : {SplitDim::N, SplitDim::M})
        EXPECT_EQ(setReplicated(gemm, dim, buffers.inputs()), HIPBLAS_STATUS_NOT_SUPPORTED);
    auto in       = buffers.inputs();
    in.storePeerD = true;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_NOT_SUPPORTED);

    // No problem was set: nothing can be queried, initialized or launched.
    int64_t offset = 0, size = 0;
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    EXPECT_NE(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(heuristic.empty());
    EXPECT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_INVALID_VALUE);
}

// Batch split: whole matrices, the first partitions take batch_count / N batches, the last the rest.
TEST(pre_checkin_partitioned_gemm_contract, BatchSplitSlices)
{
    SKIP_UNLESS_GFX1250();
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    auto              in = buffers.inputs();

    const int64_t sa = kLda * kM, sb = kLdb * kN, sc = kLdc * kN, sd = kLdd * kN;
    ASSERT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              5,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              sa,
                              sb,
                              sc,
                              sd,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);
    int64_t offset = -1, size = -1;
    ASSERT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(offset, 0);
    EXPECT_EQ(size, 2);
    ASSERT_EQ(gemm.getSlice(1, offset, size), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(offset, 2);
    EXPECT_EQ(size, 3);

    // One batch cannot be split between two partitions.
    EXPECT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              1,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              sa,
                              sb,
                              sc,
                              sd,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

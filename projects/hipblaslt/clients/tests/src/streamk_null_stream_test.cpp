// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the legacy null stream losing its Stream-K flag block.
//
// Keyed on the raw hipStream_t the null stream's key is nullptr, which the claim
// loop in streamKFlagsForStream() also uses to mean "block free": the CAS stores
// nullptr into a block that already holds nullptr, the block stays free, and the
// null stream aliases whichever stream claims it next.
//
// Asserted here as accounting rather than by running that alias into the
// deadlock it causes. rocblaslt_matmul_impl claims a block for every matmul,
// before it knows the solution, and refuses the call once the handle's blocks
// are all taken -- so the number of distinct streams a handle serves before its
// first refusal counts the blocks that handle has spent. Two handles are
// counted: one handed nothing but fresh explicit streams, one handed a
// null-stream matmul first. That matmul has to cost exactly one block, so the
// second count has to come out one lower. Before the fix it cost nothing and the
// two counts were equal.
//
// The claim is unconditional, so no Stream-K solution is needed and the shape
// can be small. Nothing runs concurrently, so nothing here can hang whether the
// library keys the null stream apart or not.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "streamk_test_util.hpp"

#include <cstdint>

namespace
{
    using namespace streamk_test;

    // As small as a real GEMM gets: only the claim in front of it is measured,
    // and the two counts together launch a couple of hundred of them.
    constexpr int64_t kSize = 256;

    // _rocblaslt_handle::c_syncSkStreamSlots. Reported and used to size the loop
    // bound, never asserted on: raising it in the library must not fail this
    // test, which is why the assertion is on the difference between the two
    // counts rather than on either count.
    constexpr int kDocumentedBlocks = 64;

    // Past any plausible c_syncSkStreamSlots. A count that reaches it never saw
    // a refusal, which means blocks are not being claimed per stream at all and
    // the difference this test measures would mean nothing.
    constexpr int kStreamLimit = 4 * kDocumentedBlocks;

    struct Count
    {
        int             served  = 0;
        hipblasStatus_t refusal = HIPBLAS_STATUS_SUCCESS;
    };

    // One handle and the buffers its matmuls share. Every stream it is handed
    // stays in r.streams and is destroyed only with it: hipStreamDestroy lets
    // the next hipStreamCreate hand back the same hipStream_t, and two equal
    // keys would claim one block between them and undercount.
    struct Handle
    {
        Resources                        r;
        hipblasLtMatmulHeuristicResult_t heuristic{};
        float                            alpha = 1.0f, beta = 0.0f;

        hipblasStatus_t matmul(hipStream_t stream)
        {
            return hipblasLtMatmul(r.handle,
                                   r.desc,
                                   &alpha,
                                   r.dA,
                                   r.layA,
                                   r.dB,
                                   r.layB,
                                   &beta,
                                   r.dD[0],
                                   r.layD,
                                   r.dD[0],
                                   r.layD,
                                   &heuristic.algo,
                                   r.dWs[0],
                                   heuristic.workspaceSize,
                                   stream);
        }
    };

    // Asserts, so callers must check HasFatalFailure()/IsSkipped() before using
    // `h`. The solution is whatever the heuristic offers first: every one of
    // them claims a block, so there is nothing to select for here.
    void setUp(Handle& h)
    {
        createProblem(h.r, kSize, kSize, kSize);
        if(::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure())
            return;

        int returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(h.r.handle,
                                                  h.r.desc,
                                                  h.r.layA,
                                                  h.r.layB,
                                                  h.r.layD,
                                                  h.r.layD,
                                                  h.r.pref,
                                                  1,
                                                  &h.heuristic,
                                                  &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "No solution for " << kSize << "x" << kSize << "x" << kSize
                         << " on this device";

        const size_t bytes = static_cast<size_t>(kSize * kSize) * sizeof(uint16_t);
        ASSERT_EQ(hipMalloc(&h.r.dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&h.r.dB, bytes), hipSuccess);
        static_cast<void>(hipMemset(h.r.dA, 0, bytes));
        static_cast<void>(hipMemset(h.r.dB, 0, bytes));

        // One D and one workspace: the matmuls are serialised, so they may share
        // them.
        h.r.dD.assign(1, nullptr);
        ASSERT_EQ(hipMalloc(&h.r.dD[0], bytes), hipSuccess);
        h.r.dWs.assign(1, nullptr);
        if(h.heuristic.workspaceSize > 0)
            ASSERT_EQ(hipMalloc(&h.r.dWs[0], h.heuristic.workspaceSize), hipSuccess);
    }

    // Runs one matmul per fresh stream until `h` refuses, and reports how many
    // it served. Waits on each launch before making the next: the claim is
    // host-side bookkeeping, so overlapping them buys nothing, and serialising
    // keeps a library that does hand two streams one flag region from wedging
    // this test instead of failing it.
    Count countStreamsServed(Handle& h)
    {
        Count count;
        for(; count.served < kStreamLimit; ++count.served)
        {
            hipStream_t stream = nullptr;
            EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
            if(stream == nullptr)
                break;
            h.r.streams.push_back(stream);

            count.refusal = h.matmul(stream);
            if(count.refusal != HIPBLAS_STATUS_SUCCESS)
                break;
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        }
        return count;
    }

    TEST(StreamKNullStream, NullStreamSpendsOneFlagBlock)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        // Blocks are per-handle, so the two counts need a handle each. Both live
        // to the end of the test so that neither one's streams can be recycled
        // into the other's.
        Handle reference, afterNullStream;
        setUp(reference);
        if(::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure())
            return;
        setUp(afterNullStream);
        if(::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure())
            return;

        // The block this spends is the one the count below must come up short
        // by. The reference handle never sees the null stream at all.
        ASSERT_EQ(afterNullStream.matmul(nullptr), HIPBLAS_STATUS_SUCCESS)
            << "the null stream was refused by a handle with every flag block still free";
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

        const Count withoutNullStream = countStreamsServed(reference);
        const Count withNullStream    = countStreamsServed(afterNullStream);

        ASSERT_LT(withoutNullStream.served, kStreamLimit)
            << kStreamLimit << " distinct streams were served without a refusal, so this handle "
            << "is not spending a flag block per stream and the counts below measure nothing";
        ASSERT_LT(withNullStream.served, kStreamLimit)
            << kStreamLimit << " distinct streams were served without a refusal, so this handle "
            << "is not spending a flag block per stream and the counts below measure nothing";
        EXPECT_EQ(withoutNullStream.refusal, HIPBLAS_STATUS_INTERNAL_ERROR)
            << "the count stopped for some reason other than the handle running out of blocks";
        EXPECT_EQ(withNullStream.refusal, HIPBLAS_STATUS_INTERNAL_ERROR)
            << "the count stopped for some reason other than the handle running out of blocks";

        EXPECT_EQ(withNullStream.served, withoutNullStream.served - 1)
            << "a matmul on the legacy null stream left the handle's flag blocks untouched: it "
               "keys on nullptr, which the claim loop reads as an unclaimed block, so the block "
               "it was handed stays free for the next stream to take and the two end up sharing "
               "one flag region";

        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    }
} // namespace

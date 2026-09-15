// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for TensileLite::appendFusedSegment (no GPU, no file I/O).

#include <gtest/gtest.h>

#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/KernelArguments.hpp>

using namespace TensileLite;

namespace
{
    std::vector<FusedA2APeerFields> makePeers(size_t n)
    {
        std::vector<FusedA2APeerFields> peers(n);
        for(size_t j = 0; j < n; j++)
            for(size_t k = 0; k < FUSED_A2A_SLOT_COUNT; k++)
                peers[j][k] = reinterpret_cast<void*>(0x1000 + j * 0x100 + k * 8);
        return peers;
    }

    void appendWorld(KernelArgumentsCounter& args, uint32_t world, size_t peerCount)
    {
        void* counter = reinterpret_cast<void*>(0xC0FFEE00);
        appendFusedSegment(args, makePeers(peerCount), counter, 0, world, 0, 0, 304);
    }
}

TEST(FusedA2AKernArg, AppendsFixedSegmentAtSupportedWorldSizes)
{
    for(uint32_t w : {1u, 2u, 4u, 8u})
    {
        KernelArgumentsCounter args;
        const size_t           before = args.size();
        ASSERT_NO_THROW(appendWorld(args, w, w)) << "world size " << w;
        EXPECT_EQ(args.size() - before, FUSED_A2A_SEGMENT_BYTES) << "world size " << w;
    }
}

TEST(FusedA2AKernArg, RejectsWorldSizeAboveMaxRanks)
{
    for(uint32_t w : {(uint32_t)FUSED_A2A_MAX_RANKS + 1, 16u})
    {
        KernelArgumentsCounter args;
        EXPECT_THROW(appendWorld(args, w, w), std::runtime_error) << "world size " << w;
    }
}

TEST(FusedA2AKernArg, RejectsZeroWorldSize)
{
    KernelArgumentsCounter args;
    EXPECT_THROW(appendWorld(args, 0, 0), std::runtime_error);
}

TEST(FusedA2AKernArg, RejectsOversizedWorldWithClampedPeerList)
{
    KernelArgumentsCounter args;
    EXPECT_THROW(appendWorld(args, 16, FUSED_A2A_MAX_RANKS), std::runtime_error);
}

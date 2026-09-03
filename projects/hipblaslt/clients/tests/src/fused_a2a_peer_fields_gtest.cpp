// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the fused-A2A peer kernarg groups: which slot each source lands in
// and what happens when a source is absent.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_fused_a2a_peers.hpp"

namespace
{
    constexpr size_t kFlagSlot       = 0;
    constexpr size_t kRecvSlot       = 1;
    constexpr size_t kFirstQueueSlot = 2;
    constexpr size_t kQueueBufSlot   = 2;
    constexpr size_t kRptrSlot       = 3;
    constexpr size_t kWptrSlot       = 4;
    constexpr size_t kDoorbellSlot   = 5;

    void* fakePointer(uintptr_t value)
    {
        return reinterpret_cast<void*>(value);
    }

    // The slot literals above are spelled out rather than aliased to the enum, so a
    // reordered or widened FusedA2APeerSlot fails here and not silently downstream.
    TEST(FusedA2APeerFields, pinsTheKernargSlotLayout)
    {
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_COUNT), 6u);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_FLAG_PTR), kFlagSlot);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_RECV_PTR), kRecvSlot);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_QUEUE_BUF), kQueueBufSlot);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_RPTR), kRptrSlot);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_WPTR), kWptrSlot);
        EXPECT_EQ(size_t(TensileLite::FUSED_A2A_SLOT_DOORBELL), kDoorbellSlot);
    }

    TEST(FusedA2APeerFields, placesFlagAndRecvPerRank)
    {
        void* flags[4] = {fakePointer(0x1000), fakePointer(0x2000), fakePointer(0x3000),
                          fakePointer(0x4000)};
        void* recvs[4] = {fakePointer(0x5000), fakePointer(0x6000), fakePointer(0x7000),
                          fakePointer(0x8000)};

        const auto peers = rocblaslt::buildFusedA2APeerFields(flags, recvs, 4, 0, nullptr);

        ASSERT_EQ(peers.size(), 4u);
        for(uint32_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ(peers[j][kFlagSlot], flags[j]) << "rank " << j;
            EXPECT_EQ(peers[j][kRecvSlot], recvs[j]) << "rank " << j;
        }
    }

    TEST(FusedA2APeerFields, leavesQueueSlotsNull)
    {
        void* flags[2] = {fakePointer(0x1000), fakePointer(0x2000)};
        void* recvs[2] = {fakePointer(0x3000), fakePointer(0x4000)};

        const auto peers = rocblaslt::buildFusedA2APeerFields(flags, recvs, 2, 0, nullptr);

        ASSERT_EQ(peers.size(), 2u);
        for(const auto& peer : peers)
            for(size_t slot = kFirstQueueSlot; slot < peer.size(); ++slot)
                EXPECT_EQ(peer[slot], nullptr) << "slot " << slot;
    }

    TEST(FusedA2APeerFields, nullRecvLeavesRecvSlotNull)
    {
        void* flags[2] = {fakePointer(0x1000), fakePointer(0x2000)};

        const auto peers = rocblaslt::buildFusedA2APeerFields(flags, nullptr, 2, 0, nullptr);

        ASSERT_EQ(peers.size(), 2u);
        EXPECT_EQ(peers[0][kFlagSlot], flags[0]);
        EXPECT_EQ(peers[1][kFlagSlot], flags[1]);
        EXPECT_EQ(peers[0][kRecvSlot], nullptr);
        EXPECT_EQ(peers[1][kRecvSlot], nullptr);
    }

    TEST(FusedA2APeerFields, nullFlagLeavesFlagSlotNull)
    {
        void* recvs[2] = {fakePointer(0x3000), fakePointer(0x4000)};

        const auto peers = rocblaslt::buildFusedA2APeerFields(nullptr, recvs, 2, 0, nullptr);

        ASSERT_EQ(peers.size(), 2u);
        EXPECT_EQ(peers[0][kFlagSlot], nullptr);
        EXPECT_EQ(peers[1][kFlagSlot], nullptr);
        EXPECT_EQ(peers[0][kRecvSlot], recvs[0]);
        EXPECT_EQ(peers[1][kRecvSlot], recvs[1]);
    }

    TEST(FusedA2APeerFields, placesQueueFieldsPerRank)
    {
        hipblasLtSdmaQueue_t queues[2]
            = {{fakePointer(0x100), fakePointer(0x110), fakePointer(0x120), fakePointer(0x130)},
               {fakePointer(0x200), fakePointer(0x210), fakePointer(0x220), fakePointer(0x230)}};

        const auto peers = rocblaslt::buildFusedA2APeerFields(nullptr, nullptr, 2, 0, queues);

        ASSERT_EQ(peers.size(), 2u);
        for(uint32_t j = 0; j < 2; ++j)
        {
            EXPECT_EQ(peers[j][kQueueBufSlot], queues[j].queueBuf) << "rank " << j;
            EXPECT_EQ(peers[j][kRptrSlot], queues[j].rptr) << "rank " << j;
            EXPECT_EQ(peers[j][kWptrSlot], queues[j].wptr) << "rank " << j;
            EXPECT_EQ(peers[j][kDoorbellSlot], queues[j].doorbell) << "rank " << j;
        }
    }

    TEST(FusedA2APeerFields, allThreeSourcesLandInDisjointSlots)
    {
        void*                flags[1]  = {fakePointer(0x1000)};
        void*                recvs[1]  = {fakePointer(0x2000)};
        hipblasLtSdmaQueue_t queues[1] = {
            {fakePointer(0x100), fakePointer(0x110), fakePointer(0x120), fakePointer(0x130)}};

        const auto peers = rocblaslt::buildFusedA2APeerFields(flags, recvs, 1, 0, queues);

        ASSERT_EQ(peers.size(), 1u);
        EXPECT_EQ(peers[0][kFlagSlot], flags[0]);
        EXPECT_EQ(peers[0][kRecvSlot], recvs[0]);
        EXPECT_EQ(peers[0][kQueueBufSlot], queues[0].queueBuf);
        EXPECT_EQ(peers[0][kDoorbellSlot], queues[0].doorbell);
    }

    TEST(FusedA2APeerFields, channelBiasesTheFlagSlotOnly)
    {
        void*                flags[2] = {fakePointer(0x10000), fakePointer(0x20000)};
        void*                recvs[2] = {fakePointer(0x30000), fakePointer(0x40000)};
        hipblasLtSdmaQueue_t queues[2]
            = {{fakePointer(0x100), fakePointer(0x110), fakePointer(0x120), fakePointer(0x130)},
               {fakePointer(0x200), fakePointer(0x210), fakePointer(0x220), fakePointer(0x230)}};

        const auto ch0 = rocblaslt::buildFusedA2APeerFields(flags, recvs, 2, 0, queues);
        const auto ch1 = rocblaslt::buildFusedA2APeerFields(flags, recvs, 2, 1, queues);

        ASSERT_EQ(ch0.size(), 2u);
        ASSERT_EQ(ch1.size(), 2u);
        for(uint32_t j = 0; j < 2; ++j)
        {
            EXPECT_EQ(ch0[j][kFlagSlot], flags[j]) << "rank " << j;
            EXPECT_EQ(static_cast<char*>(ch1[j][kFlagSlot]) - static_cast<char*>(ch0[j][kFlagSlot]),
                      ptrdiff_t(TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES))
                << "rank " << j;
            EXPECT_EQ(ch1[j][kRecvSlot], ch0[j][kRecvSlot]) << "rank " << j;
            for(size_t slot = kFirstQueueSlot; slot < ch1[j].size(); ++slot)
                EXPECT_EQ(ch1[j][slot], ch0[j][slot]) << "rank " << j << " slot " << slot;
        }
    }

    TEST(FusedA2APeerFields, channelLeavesAbsentFlagsNull)
    {
        void* flags[2] = {fakePointer(0x1000), nullptr};

        const auto peers = rocblaslt::buildFusedA2APeerFields(flags, nullptr, 2, 3, nullptr);

        ASSERT_EQ(peers.size(), 2u);
        EXPECT_EQ(static_cast<char*>(peers[0][kFlagSlot]) - static_cast<char*>(flags[0]),
                  ptrdiff_t(3 * TensileLite::FUSED_A2A_FLAG_BLOCK_BYTES));
        EXPECT_EQ(peers[1][kFlagSlot], nullptr);
    }

    TEST(FusedA2ADrain, inKernelCompletionAsksForRecvDrain)
    {
        EXPECT_EQ(rocblaslt::fusedA2ADrainFor(HIPBLASLT_A2A_COMPLETION_IN_KERNEL),
                  TensileLite::FUSED_A2A_DRAIN_RECV);
    }

    TEST(FusedA2ADrain, neverAsksForSendDrain)
    {
        EXPECT_EQ(rocblaslt::fusedA2ADrainFor(HIPBLASLT_A2A_COMPLETION_IN_KERNEL)
                      & TensileLite::FUSED_A2A_DRAIN_SEND,
                  0u);
    }

    TEST(FusedA2APeerFields, rejectsWorldOutsideTheSegment)
    {
        void* flags[1] = {fakePointer(0x1000)};

        EXPECT_TRUE(rocblaslt::buildFusedA2APeerFields(flags, nullptr, 0, 0, nullptr).empty());
        EXPECT_TRUE(rocblaslt::buildFusedA2APeerFields(
                        flags, nullptr, uint32_t(TensileLite::FUSED_A2A_MAX_RANKS) + 1, 0, nullptr)
                        .empty());
    }

    TEST(FusedA2APeerFields, acceptsTheFullSegmentWidth)
    {
        void* flags[TensileLite::FUSED_A2A_MAX_RANKS] = {};
        for(int j = 0; j < TensileLite::FUSED_A2A_MAX_RANKS; ++j)
            flags[j] = fakePointer(0x1000 + uintptr_t(j));

        const auto peers = rocblaslt::buildFusedA2APeerFields(
            flags, nullptr, uint32_t(TensileLite::FUSED_A2A_MAX_RANKS), 0, nullptr);

        ASSERT_EQ(peers.size(), size_t(TensileLite::FUSED_A2A_MAX_RANKS));
        for(int j = 0; j < TensileLite::FUSED_A2A_MAX_RANKS; ++j)
            EXPECT_EQ(peers[size_t(j)][kFlagSlot], flags[j]) << "rank " << j;
    }
}

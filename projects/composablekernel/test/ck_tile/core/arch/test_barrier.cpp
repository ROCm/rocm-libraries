// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <utility>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/named_barrier.hpp"

using ck_tile::index_t;
using ck_tile::kMaxNamedBarrierId;
using ck_tile::named_barrier_pipeline;
using ck_tile::ring_spec;

namespace {

// The device entry points cannot be instantiated off gfx1250, so what is testable here is
// the compile-time contract they rest on: the id allocation the backend must match.

// Flattens one ring's id space: DATA slots first, then FREE per producer.
template <typename Ring, index_t I>
constexpr index_t nth_id()
{
    constexpr index_t kSlots = Ring::kNumSlots;
    if constexpr(I < kSlots)
    {
        return Ring::template data_id<I>();
    }
    else
    {
        return Ring::template free_id<(I - kSlots) / kSlots, (I - kSlots) % kSlots>();
    }
}

template <typename Ring, std::size_t... Is>
constexpr auto collect_ids(std::index_sequence<Is...>)
{
    return std::array<index_t, sizeof...(Is)>{nth_id<Ring, static_cast<index_t>(Is)>()...};
}

template <typename Ring>
constexpr auto collect_ids()
{
    return collect_ids<Ring>(
        std::make_index_sequence<static_cast<std::size_t>(Ring::kNumBarriers)>{});
}

// A ring's ids must be a bijection onto the range its base claims, and every id must be one
// the hardware can address. An alias merges two handshakes into one, which surfaces as a
// hang rather than a wrong result.
template <typename Ring, index_t BaseId>
constexpr bool ids_are_bijective_and_addressable()
{
    const auto ids = collect_ids<Ring>();

    for(std::size_t i = 0; i < ids.size(); ++i)
    {
        // Addressable by S_BARRIER_SIGNAL/WAIT, and not the workgroup-wide id 0.
        if(ids[i] < 1 || ids[i] > kMaxNamedBarrierId)
        {
            return false;
        }
        // Inside this ring's own slice of the arena.
        if(ids[i] < BaseId || ids[i] >= BaseId + Ring::kNumBarriers)
        {
            return false;
        }
        for(std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if(ids[i] == ids[j])
            {
                return false;
            }
        }
    }
    return true;
}

// Two rings of different depth sharing one arena: the case a single ring cannot express.
using asym_pipe = named_barrier_pipeline<ring_spec<3, 1, 2>, ring_spec<2, 1, 2>>;
using asym_a    = asym_pipe::ring<0>;
using asym_b    = asym_pipe::ring<1>;

// The largest pipeline the default pool holds.
using full_pipe = named_barrier_pipeline<ring_spec<3, 4, 1>>;
using full_ring = full_pipe::ring<0>;

// The shape the gfx1250 GEMM pipeline is planned around.
using gemm_pipe = named_barrier_pipeline<ring_spec<2, 2, 2>>;
using gemm_ring = gemm_pipe::ring<0>;

} // namespace

TEST(NamedBarrierPipeline, RingsTileTheArenaWithoutOverlap)
{
    constexpr bool kAok    = ids_are_bijective_and_addressable<asym_a, 1>();
    constexpr bool kBok    = ids_are_bijective_and_addressable<asym_b, 7>();
    constexpr bool kFullOk = ids_are_bijective_and_addressable<full_ring, 1>();
    constexpr bool kGemmOk = ids_are_bijective_and_addressable<gemm_ring, 1>();

    EXPECT_TRUE(kAok) << "ring<0> ids alias or leave its slice";
    EXPECT_TRUE(kBok) << "ring<1> ids alias or leave its slice";
    EXPECT_TRUE(kFullOk) << "the saturating ring escapes [1, 15]";
    EXPECT_TRUE(kGemmOk) << "the 2P/2C ring aliases or escapes its slice";

    // The pipeline is exactly the sum of its rings, so the slices tile it with no gap.
    EXPECT_EQ(asym_a::kNumBarriers, 6);
    EXPECT_EQ(asym_b::kNumBarriers, 4);
    EXPECT_EQ(asym_pipe::kNumBarriers, 10);
    EXPECT_EQ(gemm_ring::kNumBarriers, 6);
    EXPECT_EQ(full_ring::kNumBarriers, 15);
    EXPECT_LE(full_pipe::kNumBarriers, kMaxNamedBarrierId);
}

TEST(NamedBarrierPipeline, BasesAreComputedNotWritten)
{
    // ring<1> starts where ring<0> ends. This is the invariant the backend must match, and
    // the extra parens keep the preprocessor from splitting on the template argument comma.
    EXPECT_EQ(asym_a::data_id<0>(), 1);
    EXPECT_EQ((asym_a::free_id<0, 2>()), 6);
    EXPECT_EQ(asym_b::data_id<0>(), 7);
    EXPECT_EQ((asym_b::free_id<0, 1>()), 10);
}

TEST(NamedBarrierSlotRing, MemberCountsIncludeTheWaiter)
{
    // Consumers wait on DATA, so every producer and every consumer must be a member. For the
    // 2P/2C pipeline that is what makes one wait mean "A and B are both ready".
    EXPECT_EQ(gemm_ring::kDataMemberCount, 2u + 2u);
    // One producer waits on its own FREE, so it and every consumer must be a member.
    EXPECT_EQ(gemm_ring::kFreeMemberCount, 1u + 2u);
}

TEST(NamedBarrierSlotRing, OccupiesNoLds)
{
    // Named barriers are a separate hardware pool; a pipeline aggregating LDS must see zero.
    EXPECT_EQ(asym_a::GetSmemSize(), 0);
    EXPECT_EQ(full_ring::GetSmemSize(), 0);
}

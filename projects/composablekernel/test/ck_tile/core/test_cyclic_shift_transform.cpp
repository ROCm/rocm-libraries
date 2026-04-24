// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/sequence.hpp"

using namespace ck_tile;

// ---------------------------------------------------------------------------
// Test parameters: different (dim0_len, dim1_len) sizes
// ---------------------------------------------------------------------------

template <int Dim0_, int Dim1_>
struct CyclicShiftParam
{
    static constexpr int Dim0 = Dim0_;
    static constexpr int Dim1 = Dim1_;
};

using CyclicShiftTypes = ::testing::Types<
    CyclicShiftParam<10, 8>,  // BLOCK_W=10, BLOCK_C8=8 (typical 4c, C=64)
    CyclicShiftParam<20, 16>, // BLOCK_W=20, BLOCK_C8=16 (typical 4c, C=128)
    CyclicShiftParam<4, 4>,   // small square
    CyclicShiftParam<8, 1>,   // degenerate: dim1=1 (shift is always 0)
    CyclicShiftParam<1, 8>    // degenerate: dim0=1 (no shift)
    >;

template <typename P>
class CyclicShiftTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CyclicShiftTest, CyclicShiftTypes);

// ---------------------------------------------------------------------------
// Helper: create the transform and compute lower index for (x, c8)
// ---------------------------------------------------------------------------
template <typename P>
static auto compute_lower(int x, int c8)
{
    constexpr auto lengths = make_tuple(number<P::Dim0>{}, number<P::Dim1>{});
    auto transform         = make_cyclic_shift_transform(lengths);

    auto idx_up  = make_multi_index(x, c8);
    auto idx_low = make_zero_multi_index<2>();
    transform.calculate_lower_index(idx_low, idx_up);
    return idx_low;
}

// ---------------------------------------------------------------------------
// 1. Basic mapping: idx_low(0) == x, idx_low(1) == (c8 + x) % Dim1
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, BasicMapping)
{
    constexpr int D0 = TypeParam::Dim0;
    constexpr int D1 = TypeParam::Dim1;

    for(int x = 0; x < D0; x++)
    {
        for(int c8 = 0; c8 < D1; c8++)
        {
            auto idx_low = compute_lower<TypeParam>(x, c8);
            EXPECT_EQ(idx_low[0], x) << "x=" << x << " c8=" << c8;
            EXPECT_EQ(idx_low[1], (c8 + x) % D1) << "x=" << x << " c8=" << c8;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Per-row bijection: for each row x, the dim1 outputs are a permutation
//    of 0..Dim1-1
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, PerRowBijection)
{
    constexpr int D0 = TypeParam::Dim0;
    constexpr int D1 = TypeParam::Dim1;

    for(int x = 0; x < D0; x++)
    {
        std::set<int> seen;
        for(int c8 = 0; c8 < D1; c8++)
        {
            auto idx_low = compute_lower<TypeParam>(x, c8);
            EXPECT_TRUE(seen.insert(idx_low[1]).second)
                << "Duplicate dim1 output at x=" << x << " c8=" << c8;
        }
        EXPECT_EQ(static_cast<int>(seen.size()), D1) << "x=" << x;
    }
}

// ---------------------------------------------------------------------------
// 3. Row 0 is identity: (0 + c8) % Dim1 == c8
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, Row0IsIdentity)
{
    constexpr int D1 = TypeParam::Dim1;

    for(int c8 = 0; c8 < D1; c8++)
    {
        auto idx_low = compute_lower<TypeParam>(0, c8);
        EXPECT_EQ(idx_low[1], c8) << "c8=" << c8;
    }
}

// ---------------------------------------------------------------------------
// 4. Dim1 output is always in valid range [0, Dim1)
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, OutputInRange)
{
    constexpr int D0 = TypeParam::Dim0;
    constexpr int D1 = TypeParam::Dim1;

    for(int x = 0; x < D0; x++)
    {
        for(int c8 = 0; c8 < D1; c8++)
        {
            auto idx_low = compute_lower<TypeParam>(x, c8);
            EXPECT_GE(idx_low[1], 0) << "x=" << x << " c8=" << c8;
            EXPECT_LT(idx_low[1], D1) << "x=" << x << " c8=" << c8;
        }
    }
}

// ---------------------------------------------------------------------------
// 5. update_lower_index is consistent with calculate_lower_index
//    Start at (x0, c0), move by (dx, dc), verify the updated index matches
//    a fresh calculation at (x0+dx, c0+dc).
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, UpdateConsistentWithCalculate)
{
    constexpr int D0 = TypeParam::Dim0;
    constexpr int D1 = TypeParam::Dim1;

    constexpr auto lengths = make_tuple(number<D0>{}, number<D1>{});
    auto transform         = make_cyclic_shift_transform(lengths);

    // Test several (start, delta) combinations
    for(int x0 = 0; x0 < D0; x0++)
    {
        for(int c0 = 0; c0 < D1; c0++)
        {
            // Compute initial lower index
            auto idx_up_old  = make_multi_index(x0, c0);
            auto idx_low_old = make_zero_multi_index<2>();
            transform.calculate_lower_index(idx_low_old, idx_up_old);

            // Try several deltas
            for(int dx = -1; dx <= 1; dx++)
            {
                for(int dc = -1; dc <= 1; dc++)
                {
                    int x1 = x0 + dx;
                    int c1 = c0 + dc;
                    if(x1 < 0 || x1 >= D0 || c1 < 0 || c1 >= D1)
                        continue;

                    // Expected: fresh calculation at (x1, c1)
                    auto idx_up_new      = make_multi_index(x1, c1);
                    auto idx_low_expect  = make_zero_multi_index<2>();
                    transform.calculate_lower_index(idx_low_expect, idx_up_new);

                    // Actual: update from (x0, c0) by (dx, dc)
                    auto idx_diff_up  = make_multi_index(dx, dc);
                    auto idx_diff_low = make_zero_multi_index<2>();
                    auto idx_low_copy = idx_low_old;
                    transform.update_lower_index(idx_diff_low, idx_diff_up, idx_low_copy, idx_up_new);

                    EXPECT_EQ(idx_low_copy[0], idx_low_expect[0])
                        << "dim0 mismatch: (" << x0 << "," << c0 << ")+(" << dx << "," << dc << ")";
                    EXPECT_EQ(idx_low_copy[1], idx_low_expect[1])
                        << "dim1 mismatch: (" << x0 << "," << c0 << ")+(" << dx << "," << dc << ")";

                    // Also check the diff
                    EXPECT_EQ(idx_diff_low[0], idx_low_expect[0] - idx_low_old[0])
                        << "diff dim0: (" << x0 << "," << c0 << ")+(" << dx << "," << dc << ")";
                    EXPECT_EQ(idx_diff_low[1], idx_low_expect[1] - idx_low_old[1])
                        << "diff dim1: (" << x0 << "," << c0 << ")+(" << dx << "," << dc << ")";
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Validity predicates
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, ValidityPredicates)
{
    EXPECT_TRUE(
        (cyclic_shift_t<decltype(make_tuple(number<TypeParam::Dim0>{}, number<TypeParam::Dim1>{}))>::
             is_valid_upper_index_always_mapped_to_valid_lower_index()));

    constexpr auto lengths = make_tuple(number<TypeParam::Dim0>{}, number<TypeParam::Dim1>{});
    auto transform         = make_cyclic_shift_transform(lengths);
    auto idx               = make_multi_index(0, 0);
    EXPECT_TRUE(transform.is_valid_upper_index_mapped_to_valid_lower_index(idx));
}

// ---------------------------------------------------------------------------
// 7. Known at compile time when lengths are compile-time constants
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, KnownAtCompileTime)
{
    using Lengths = decltype(make_tuple(number<TypeParam::Dim0>{}, number<TypeParam::Dim1>{}));
    EXPECT_TRUE((cyclic_shift_t<Lengths>::is_known_at_compile_time()));
}

// ---------------------------------------------------------------------------
// 8. Enum value is correct
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, EnumValue)
{
    using Lengths = decltype(make_tuple(number<TypeParam::Dim0>{}, number<TypeParam::Dim1>{}));
    EXPECT_EQ(cyclic_shift_t<Lengths>::get_type_enum(), coord_transform_enum::cyclic_shift);
}

// ---------------------------------------------------------------------------
// 9. Upper lengths match construction arguments
// ---------------------------------------------------------------------------
TYPED_TEST(CyclicShiftTest, UpperLengths)
{
    constexpr auto lengths = make_tuple(number<TypeParam::Dim0>{}, number<TypeParam::Dim1>{});
    auto transform         = make_cyclic_shift_transform(lengths);
    auto up                = transform.get_upper_lengths();
    EXPECT_EQ(up[number<0>{}], TypeParam::Dim0);
    EXPECT_EQ(up[number<1>{}], TypeParam::Dim1);
}

// ---------------------------------------------------------------------------
// 10. Differs from XOR: verify cyclic_shift != xor for x >= 2, dim1 >= 4
//     (they agree at x=0 and x=1 for power-of-2 dim1, but diverge otherwise)
// ---------------------------------------------------------------------------
TEST(CyclicShiftVsXor, DiffersFromXor)
{
    constexpr auto lengths = make_tuple(number<10>{}, number<8>{});
    auto cs                = make_cyclic_shift_transform(lengths);
    auto xr                = make_xor_transform(lengths);

    bool found_difference = false;
    for(int x = 0; x < 10; x++)
    {
        for(int c8 = 0; c8 < 8; c8++)
        {
            auto idx_up   = make_multi_index(x, c8);
            auto low_cs   = make_zero_multi_index<2>();
            auto low_xor  = make_zero_multi_index<2>();
            cs.calculate_lower_index(low_cs, idx_up);
            xr.calculate_lower_index(low_xor, idx_up);

            if(low_cs[1] != low_xor[1])
            {
                found_difference = true;
            }
        }
    }
    EXPECT_TRUE(found_difference) << "cyclic_shift and xor should produce different mappings";
}

// ---------------------------------------------------------------------------
// 11. Factory function: make_cyclic_shift_transform returns correct type
// ---------------------------------------------------------------------------
TEST(CyclicShiftFactory, FactoryFunction)
{
    constexpr auto lengths = make_tuple(number<8>{}, number<4>{});
    auto transform         = make_cyclic_shift_transform(lengths);

    // Verify it works by computing a known value
    auto idx_up  = make_multi_index(3, 2);
    auto idx_low = make_zero_multi_index<2>();
    transform.calculate_lower_index(idx_low, idx_up);

    // (2 + 3) % 4 = 1
    EXPECT_EQ(idx_low[0], 3);
    EXPECT_EQ(idx_low[1], 1);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/numeric/math.hpp"

using namespace ck_tile;

// ============================================================================
// sequence::modify tests
// ============================================================================

TEST(CkTileSequence, ModifyFirstElement)
{
    constexpr auto result = sequence<1, 2, 3, 4>{}.modify(number<0>{}, number<99>{});
    static_assert(std::is_same_v<decltype(result), const sequence<99, 2, 3, 4>>);
}

TEST(CkTileSequence, ModifyLastElement)
{
    constexpr auto result = sequence<1, 2, 3, 4>{}.modify(number<3>{}, number<99>{});
    static_assert(std::is_same_v<decltype(result), const sequence<1, 2, 3, 99>>);
}

TEST(CkTileSequence, ModifyMiddleElement)
{
    constexpr auto result = sequence<5, 5, 5>{}.modify(number<1>{}, number<0>{});
    static_assert(std::is_same_v<decltype(result), const sequence<5, 0, 5>>);
}

TEST(CkTileSequence, ModifySingleElement)
{
    constexpr auto result = sequence<42>{}.modify(number<0>{}, number<99>{});
    static_assert(std::is_same_v<decltype(result), const sequence<99>>);
}

// ============================================================================
// sequence_gen tests
// ============================================================================

TEST(CkTileSequence, SequenceGenZero)
{
    using Result = typename sequence_gen<0, identity>::type;
    static_assert(std::is_same_v<Result, sequence<>>);
}

TEST(CkTileSequence, SequenceGenIdentity)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i; }
    };
    using Result = typename sequence_gen<5, F>::type;
    static_assert(std::is_same_v<Result, sequence<0, 1, 2, 3, 4>>);
}

TEST(CkTileSequence, SequenceGenDouble)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i * 2; }
    };
    using Result = typename sequence_gen<4, F>::type;
    static_assert(std::is_same_v<Result, sequence<0, 2, 4, 6>>);
}

TEST(CkTileSequence, SequenceGenSingle)
{
    struct F
    {
        constexpr index_t operator()(index_t) const { return 42; }
    };
    using Result = typename sequence_gen<1, F>::type;
    static_assert(std::is_same_v<Result, sequence<42>>);
}

TEST(CkTileSequence, SequenceGenLarger)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i * i; }
    };
    using Result = typename sequence_gen<8, F>::type;
    static_assert(std::is_same_v<Result, sequence<0, 1, 4, 9, 16, 25, 36, 49>>);
}

// ============================================================================
// uniform_sequence_gen tests
// ============================================================================

TEST(CkTileSequence, UniformSequenceGenZero)
{
    using Result = typename uniform_sequence_gen<0, 7>::type;
    static_assert(std::is_same_v<Result, sequence<>>);
}

TEST(CkTileSequence, UniformSequenceGenSingle)
{
    using Result = typename uniform_sequence_gen<1, 99>::type;
    static_assert(std::is_same_v<Result, sequence<99>>);
}

TEST(CkTileSequence, UniformSequenceGenMultiple)
{
    using Result = typename uniform_sequence_gen<4, 0>::type;
    static_assert(std::is_same_v<Result, sequence<0, 0, 0, 0>>);
}

TEST(CkTileSequence, UniformSequenceGenLarger)
{
    using Result = typename uniform_sequence_gen<8, 3>::type;
    static_assert(std::is_same_v<Result, sequence<3, 3, 3, 3, 3, 3, 3, 3>>);
}

// ============================================================================
// sequence_reverse_inclusive_scan tests
// ============================================================================

TEST(CkTileSequence, ReverseInclusiveScanProduct)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<1, 2, 3, 4>,
                                                            multiplies<index_t>,
                                                            1>::type;
    // reverse inclusive scan with multiply, init=1:
    //   result[3] = 4*1 = 4
    //   result[2] = 3*4 = 12
    //   result[1] = 2*12 = 24
    //   result[0] = 1*24 = 24
    static_assert(std::is_same_v<Result, sequence<24, 24, 12, 4>>);
}

TEST(CkTileSequence, ReverseInclusiveScanSum)
{
    using Result =
        typename sequence_reverse_inclusive_scan<sequence<1, 2, 3, 4>, plus<index_t>, 0>::type;
    // reverse inclusive scan with add, init=0:
    //   result[3] = 4+0 = 4
    //   result[2] = 3+4 = 7
    //   result[1] = 2+7 = 9
    //   result[0] = 1+9 = 10
    static_assert(std::is_same_v<Result, sequence<10, 9, 7, 4>>);
}

TEST(CkTileSequence, ReverseInclusiveScanSingleElement)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<5>, plus<index_t>, 0>::type;
    static_assert(std::is_same_v<Result, sequence<5>>);
}

TEST(CkTileSequence, ReverseInclusiveScanEmpty)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<>, plus<index_t>, 0>::type;
    static_assert(std::is_same_v<Result, sequence<>>);
}

// ============================================================================
// sequence_map_inverse tests
// ============================================================================

TEST(CkTileSequence, MapInverseIdentity)
{
    using Result = typename sequence_map_inverse<sequence<0, 1, 2, 3>>::type;
    static_assert(std::is_same_v<Result, sequence<0, 1, 2, 3>>);
}

TEST(CkTileSequence, MapInverseSwap)
{
    using Result = typename sequence_map_inverse<sequence<1, 0>>::type;
    static_assert(std::is_same_v<Result, sequence<1, 0>>);
}

TEST(CkTileSequence, MapInversePermutation)
{
    using Input  = sequence<2, 0, 1>;
    using Result = typename sequence_map_inverse<Input>::type;
    // inverse of (2,0,1): pos0->2, pos1->0, pos2->1
    // so result[2]=0, result[0]=1, result[1]=2 -> (1, 2, 0)
    static_assert(std::is_same_v<Result, sequence<1, 2, 0>>);

    // Verify: input[result[i]] == i for all i
    static_assert(Input::at(number<Result::at(number<0>{})>{}) == 0);
    static_assert(Input::at(number<Result::at(number<1>{})>{}) == 1);
    static_assert(Input::at(number<Result::at(number<2>{})>{}) == 2);
}

TEST(CkTileSequence, MapInverseEmpty)
{
    using Result = typename sequence_map_inverse<sequence<>>::type;
    static_assert(std::is_same_v<Result, sequence<>>);
}

TEST(CkTileSequence, MapInverseSingle)
{
    using Result = typename sequence_map_inverse<sequence<0>>::type;
    static_assert(std::is_same_v<Result, sequence<0>>);
}

TEST(CkTileSequence, MapInverseRotation)
{
    using Input  = sequence<1, 2, 0>;
    using Result = typename sequence_map_inverse<Input>::type;
    static_assert(std::is_same_v<Result, sequence<2, 0, 1>>);
}

TEST(CkTileSequence, MapInverse4D)
{
    using Input  = sequence<2, 0, 3, 1>;
    using Result = typename sequence_map_inverse<Input>::type;
    // Verify round-trip
    static_assert(Input::at(number<Result::at(number<0>{})>{}) == 0);
    static_assert(Input::at(number<Result::at(number<1>{})>{}) == 1);
    static_assert(Input::at(number<Result::at(number<2>{})>{}) == 2);
    static_assert(Input::at(number<Result::at(number<3>{})>{}) == 3);
}

// ============================================================================
// make_index_sequence tests
// ============================================================================

TEST(CkTileSequence, MakeIndexSequenceZero)
{
    using Result = make_index_sequence<0>;
    static_assert(std::is_same_v<Result, sequence<>>);
}

TEST(CkTileSequence, MakeIndexSequenceSmall)
{
    using Result = make_index_sequence<5>;
    static_assert(std::is_same_v<Result, sequence<0, 1, 2, 3, 4>>);
}

// ============================================================================
// detail::index_array tests
// ============================================================================

TEST(CkTileSequence, IndexArrayConstruction)
{
    constexpr detail::index_array<3> arr{};
    static_assert(arr[0] == 0);
    static_assert(arr[1] == 0);
    static_assert(arr[2] == 0);
}

TEST(CkTileSequence, IndexArrayInitialized)
{
    constexpr detail::index_array<3> arr = {{1, 2, 3}};
    static_assert(arr[0] == 1);
    static_assert(arr[1] == 2);
    static_assert(arr[2] == 3);
}

TEST(CkTileSequence, IndexArrayZeroSize)
{
    // Should compile without error
    [[maybe_unused]] constexpr detail::index_array<0> arr{};
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mosaic/stride.hpp>

using namespace mosaic;

TEST(Stride, Crd2Idx) {
    // We define a bijection between coordinates of a shape and indices via
    // "colexicographic" ordering i.e column major.
    // This mapping has nothing to do with a stride.
    // It can also be viewed as a layout function with column major strides.
    auto shape = std::make_tuple(Int<4>{}, Int<8>{});

    auto coord0 = std::make_tuple(Int<0>{}, Int<0>{});
    EXPECT_EQ(crd2idx(coord0, shape), Int<0>{});

    auto coord1 = std::make_tuple(Int<1>{}, Int<0>{});
    EXPECT_EQ(crd2idx(coord1, shape), Int<1>{});

    auto coord2 = std::make_tuple(Int<0>{}, Int<1>{});
    EXPECT_EQ(crd2idx(coord2, shape), Int<4>{});
}

TEST(Stride, Crd2IdxIdx2CrdRoundTrip) {
    auto shape = std::make_tuple(Int<4>{}, Int<8>{});
    for (int i = 0; i < size(shape); ++i) {
        auto coord = idx2crd(i, shape);
        auto idx = crd2idx(coord, shape);
        EXPECT_EQ(idx, i);
    }
}

TEST(Stride, Crd2IdxStride) {
    // When both a shape and stride are provided, we compute the inner product
    // of the coordinate and stride. This is precisely the layout function.
    auto shape = std::make_tuple(Int<4>{}, Int<8>{});
    auto stride = std::make_tuple(Int<8>{}, Int<1>{});

    auto coord0 = std::make_tuple(Int<0>{}, Int<4>{});
    EXPECT_EQ(crd2idx(coord0, shape, stride), Int<4>{});
}

// ===================================================================
// compact_col_major(shape, base) — base-stride overload for complement.
// ===================================================================

TEST(Stride, CompactColMajorBaseScalar) {
    // shape is scalar → returns base directly.
    auto s = compact_col_major(Int<7>{}, Int<3>{});
    static_assert(std::is_same_v<decltype(s), Int<3>>);
}

TEST(Stride, CompactColMajorBaseTuple) {
    // shape = (4, 3, 2), base = 5 → strides = (5, 5*4, 5*4*3) = (5, 20, 60).
    auto shape = std::make_tuple(Int<4>{}, Int<3>{}, Int<2>{});
    auto strides = compact_col_major(shape, Int<5>{});
    static_assert(std::is_same_v<decltype(strides), std::tuple<Int<5>, Int<20>, Int<60>>>);
}

TEST(Stride, CompactColMajorBaseDefault) {
    // The base-stride overload with base=1 should match the existing
    // no-base overload.
    auto shape = std::make_tuple(Int<4>{}, Int<3>{});
    auto a = compact_col_major(shape, Int<1>{});
    auto b = compact_col_major(shape);
    static_assert(std::is_same_v<decltype(a), decltype(b)>);
}

// ===================================================================
// Hierarchical (nested) compact strides.
//
// Regression for the bug where compact_col_major / compact_row_major did
// not recurse into nested sub-shapes, producing a *non-congruent* stride
// (a scalar stride paired with a tuple-shaped mode). That broke crd2idx,
// which walks coord/shape/stride mode-by-mode and requires congruence.
//
// Expected values are CUTLASS/pycute ground truth:
//   Layout(((32,4),8)).stride         == ((1,32),128)
//   Layout((((2,3),4),(5,6))).stride  == (((1,2),6),(24,120))
//   Layout(((32,4),8,(2,3))).stride   == ((1,32),128,(1024,2048))
// ===================================================================

TEST(Stride, CompactColMajorNested) {
    // shape ((32,4),8) → stride ((1,32),128).
    auto shape = std::make_tuple(std::make_tuple(Int<32>{}, Int<4>{}), Int<8>{});
    auto strides = compact_col_major(shape);
    static_assert(
        std::is_same_v<decltype(strides), std::tuple<std::tuple<Int<1>, Int<32>>, Int<128>>>);
}

TEST(Stride, CompactColMajorDeeplyNested) {
    // shape (((2,3),4),(5,6)) → stride (((1,2),6),(24,120)).
    auto shape = std::make_tuple(std::make_tuple(std::make_tuple(Int<2>{}, Int<3>{}), Int<4>{}),
                                 std::make_tuple(Int<5>{}, Int<6>{}));
    auto strides = compact_col_major(shape);
    static_assert(
        std::is_same_v<decltype(strides), std::tuple<std::tuple<std::tuple<Int<1>, Int<2>>, Int<6>>,
                                                     std::tuple<Int<24>, Int<120>>>>);
}

TEST(Stride, CompactColMajorMixedNested) {
    // shape ((32,4),8,(2,3)) → stride ((1,32),128,(1024,2048)).
    auto shape = std::make_tuple(std::make_tuple(Int<32>{}, Int<4>{}), Int<8>{},
                                 std::make_tuple(Int<2>{}, Int<3>{}));
    auto strides = compact_col_major(shape);
    static_assert(
        std::is_same_v<decltype(strides), std::tuple<std::tuple<Int<1>, Int<32>>, Int<128>,
                                                     std::tuple<Int<1024>, Int<2048>>>>);
}

TEST(Stride, CompactRowMajorNested) {
    // pycute: Layout(((32,4),8)) row-major stride == ((32,8),1).
    auto shape = std::make_tuple(std::make_tuple(Int<32>{}, Int<4>{}), Int<8>{});
    auto strides = compact_row_major(shape);
    static_assert(
        std::is_same_v<decltype(strides), std::tuple<std::tuple<Int<32>, Int<8>>, Int<1>>>);
}

TEST(Stride, CompactRowMajorDeeplyNested) {
    // row-major (((2,3),4),(5,6)) == (((360,120),30),(6,1)).
    auto shape = std::make_tuple(std::make_tuple(std::make_tuple(Int<2>{}, Int<3>{}), Int<4>{}),
                                 std::make_tuple(Int<5>{}, Int<6>{}));
    auto strides = compact_row_major(shape);
    static_assert(std::is_same_v<decltype(strides),
                                 std::tuple<std::tuple<std::tuple<Int<360>, Int<120>>, Int<30>>,
                                            std::tuple<Int<6>, Int<1>>>>);
}

TEST(Stride, CompactColMajorFlatUnchanged) {
    // Flat shapes must keep their original (non-nested) stride — no regression.
    auto shape = std::make_tuple(Int<4>{}, Int<3>{}, Int<2>{});
    auto strides = compact_col_major(shape);
    static_assert(std::is_same_v<decltype(strides), std::tuple<Int<1>, Int<4>, Int<12>>>);
}

TEST(Stride, NestedRoundTrip) {
    // crd2idx(idx2crd(i, shape), shape) == i over a nested shape. This is the
    // end-to-end path that previously failed to even compile.
    auto shape = std::make_tuple(std::make_tuple(Int<32>{}, Int<4>{}), Int<8>{});
    for (int i = 0; i < size(shape); ++i) {
        auto coord = idx2crd(i, shape);
        auto idx = crd2idx(coord, shape);
        EXPECT_EQ(idx, i);
    }
}

TEST(Stride, NestedCrd2IdxKnownValues) {
    // shape ((32,4),8), compact col-major stride ((1,32),128).
    // coord ((cu,wf),xcd) → cu*1 + wf*32 + xcd*128.
    auto shape = std::make_tuple(std::make_tuple(Int<32>{}, Int<4>{}), Int<8>{});

    auto c0 = std::make_tuple(std::make_tuple(1, 0), 0);
    EXPECT_EQ(crd2idx(c0, shape), 1);

    auto c1 = std::make_tuple(std::make_tuple(0, 1), 0);
    EXPECT_EQ(crd2idx(c1, shape), 32);

    auto c2 = std::make_tuple(std::make_tuple(0, 0), 1);
    EXPECT_EQ(crd2idx(c2, shape), 128);

    auto c3 = std::make_tuple(std::make_tuple(5, 2), 3);
    EXPECT_EQ(crd2idx(c3, shape), 5 + 2 * 32 + 3 * 128);
}

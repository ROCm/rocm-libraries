// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <mosaic/tensor.hpp>

using namespace mosaic;

// ===================================================================
// Tensor<Engine, Layout>
//   T(c) = engine[layout(c)]
//
// Tests verify:
//   - factory functions construct correct (engine, layout) pairs
//   - operator() reads/writes through the engine (here: a raw pointer)
//   - slicing returns a sub-Tensor with the engine shifted to absorb
//     the bound coords' offsets (so sub-tensor evaluation indexes from 0)
// ===================================================================

// Helper: build a flat array [0, 1, 2, ..., N-1] for engine-by-pointer.
template <std::size_t N>
std::array<int, N> ramp() {
    std::array<int, N> a{};
    for (std::size_t i = 0; i < N; ++i) a[i] = static_cast<int>(i);
    return a;
}

// -------------------------------------------------------------------
// Construction + size + raw evaluation
// -------------------------------------------------------------------

TEST(Tensor, ConstructFromShape) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));
    EXPECT_EQ(static_cast<int>(size(T.layout())), 32);
    EXPECT_EQ(T(0, 0), 0);
    EXPECT_EQ(T(3, 7), 31);  // compact column-major: 3*1 + 7*4 = 31
}

TEST(Tensor, ConstructFromShapeAndStride) {
    // Row-major view of the same buffer
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8), make_stride(8, 1));
    EXPECT_EQ(T(0, 0), 0);
    EXPECT_EQ(T(0, 7), 7);
    EXPECT_EQ(T(1, 0), 8);
    EXPECT_EQ(T(3, 7), 31);
}

TEST(Tensor, ConstructFromLayout) {
    auto data = ramp<32>();
    auto L = make_layout(make_shape(4, 8), make_stride(1, 4));
    auto T = make_tensor(data.data(), L);
    EXPECT_EQ(T(2, 3), data[L(2, 3)]);
}

// -------------------------------------------------------------------
// Evaluation modes — variadic, tuple coord, flat index
// -------------------------------------------------------------------

TEST(Tensor, EvalVariadic) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));  // col-major default
    EXPECT_EQ(T(2, 3), 2 + 3 * 4);                        // 14
}

TEST(Tensor, EvalTupleCoord) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));
    EXPECT_EQ(T(make_coord(2, 3)), 14);
}

TEST(Tensor, EvalFlatIndex) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));
    // Compact col-major: T(idx) == data[idx]
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(T(i), data[i]) << "at i=" << i;
    }
}

// -------------------------------------------------------------------
// Write through Tensor — mutate via operator()
// -------------------------------------------------------------------

TEST(Tensor, MutationThroughOperator) {
    std::array<int, 32> data{};
    auto T = make_tensor(data.data(), make_shape(4, 8));
    T(2, 3) = 99;
    EXPECT_EQ(data[T.layout()(2, 3)], 99);
}

// -------------------------------------------------------------------
// Slicing — the underscore path
//
// Slicing a Tensor returns a sub-Tensor whose engine has been shifted
// by the offset of the bound coords. The sub-Tensor reads from the
// correct positions of the original buffer, indexed from 0.
// -------------------------------------------------------------------

TEST(Tensor, SliceColumn) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));  // col-major
    // Column 3 in col-major: occupies offsets 12..15 of the buffer.
    auto col3 = T(_, 3);
    EXPECT_EQ(static_cast<int>(size(col3.layout())), 4);
    EXPECT_EQ(col3(0), 12);
    EXPECT_EQ(col3(1), 13);
    EXPECT_EQ(col3(2), 14);
    EXPECT_EQ(col3(3), 15);
}

TEST(Tensor, SliceRow) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));  // col-major
    // Row 2 in col-major: occupies offsets 2, 6, 10, 14, 18, 22, 26, 30.
    auto row2 = T(2, _);
    EXPECT_EQ(static_cast<int>(size(row2.layout())), 8);
    EXPECT_EQ(row2(0), 2);
    EXPECT_EQ(row2(1), 6);
    EXPECT_EQ(row2(2), 10);
    EXPECT_EQ(row2(7), 30);
}

TEST(Tensor, SliceRowMajorTensor) {
    auto data = ramp<32>();
    // Row-major view: stride (8, 1)
    auto T = make_tensor(data.data(), make_shape(4, 8), make_stride(8, 1));
    auto row2 = T(2, _);
    // Row 2 in row-major: offsets 16, 17, ..., 23
    EXPECT_EQ(row2(0), 16);
    EXPECT_EQ(row2(7), 23);
}

TEST(Tensor, SliceWritesBackThroughOriginalEngine) {
    std::array<int, 32> data{};
    auto T = make_tensor(data.data(), make_shape(4, 8));
    auto col3 = T(_, 3);
    col3(2) = 777;
    EXPECT_EQ(data[14], 777);  // col 3, row 2 → offset 12 + 2 = 14
}

TEST(Tensor, SliceRoundsTripWithOriginal) {
    auto data = ramp<32>();
    auto T = make_tensor(data.data(), make_shape(4, 8));
    auto col3 = T(_, 3);
    for (int i = 0; i < size(col3.layout()); ++i) {
        EXPECT_EQ(col3(i), T(i, 3)) << "at i=" << i;
    }
}

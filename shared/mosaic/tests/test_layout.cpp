// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mosaic/layout.hpp>

using namespace mosaic;

TEST(Layout, LayoutFunction) {
    auto shape = make_shape(2, 7);
    auto stride = make_stride(4, 5);
    auto layout = make_layout(shape, stride);

    EXPECT_EQ(layout(1, 2), 1 * 4 + 2 * 5);
    EXPECT_EQ(layout(0, 6), 0 * 4 + 6 * 5);
}

TEST(Layout, LayoutFunctionConst) {
    auto shape = make_shape(Int<2>{}, Int<7>{});
    auto stride = make_stride(Int<4>{}, Int<5>{});
    auto layout = make_layout(shape, stride);

    EXPECT_EQ(layout(Int<1>{}, Int<2>{}), Int<1 * 4 + 2 * 5>{});
    EXPECT_EQ(layout(Int<0>{}, Int<6>{}), Int<0 * 4 + 6 * 5>{});
}

// Hierarchical shape: a shape-only layout must build a congruent compact
// column-major stride and evaluate nested coords correctly. This is the
// motivating gemm.cpp case: shape ((cu,wf),xcd) → compact stride ((1,32),128).
TEST(Layout, NestedShapeOnly) {
    auto layout = make_layout(make_shape(make_shape(32, 4), 8));

    // crd2idx of the nested coord ((cu,wf),xcd) == cu + wf*32 + xcd*128.
    auto coord = make_coord(make_coord(5, 2), 3);
    EXPECT_EQ(layout(coord), 5 + 2 * 32 + 3 * 128);

    // Round-trip over the whole domain via idx2crd.
    for (int i = 0; i < layout.size(); ++i) {
        auto c = idx2crd(i, layout.shape());
        EXPECT_EQ(layout(c), i);
    }
}

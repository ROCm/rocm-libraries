// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <mosaic/algorithms.hpp>
#include <mosaic/layout.hpp>
#include <mosaic/tensor.hpp>
#include <vector>

using namespace mosaic;

// =====================================================================
// clear(tensor) — zeroes every element.
// Provides a rank/layout agnostic way to zero out a tensor.
// =====================================================================

TEST(Clear, ZeroesPointerBackedTensor) {
    std::vector<float> buf(16);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = float(i) + 1.0f;

    auto t = make_tensor(
        buf.data(), make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})));

    clear(t);

    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_EQ(buf[i], 0.0f) << "buf[" << i << "] not zeroed";
    }
}

TEST(Clear, ZeroesStackArrayBackedTensor) {
    std::array<float, 8> frag;
    for (std::size_t i = 0; i < frag.size(); ++i) frag[i] = float(i) + 1.0f;

    auto t = make_tensor(frag.data(), make_layout(make_shape(Int<8>{})));

    clear(t);

    for (std::size_t i = 0; i < frag.size(); ++i) {
        EXPECT_EQ(frag[i], 0.0f) << "frag[" << i << "] not zeroed";
    }
}

TEST(Clear, WorksForIntegerType) {
    std::vector<int> buf{1, 2, 3, 4, 5, 6, 7, 8};
    auto t = make_tensor(buf.data(), make_layout(make_shape(Int<2>{}, Int<4>{})));
    clear(t);
    for (int v : buf) EXPECT_EQ(v, 0);
}

TEST(Clear, IsIdempotent) {
    std::vector<float> buf(8, 0.0f);
    auto t = make_tensor(buf.data(), make_layout(make_shape(Int<8>{})));
    clear(t);
    clear(t);  // second time should still leave it zero
    for (float v : buf) EXPECT_EQ(v, 0.0f);
}

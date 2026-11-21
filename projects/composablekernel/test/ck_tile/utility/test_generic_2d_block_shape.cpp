// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"

// For rows narrower than a wave (ThreadPerBlock_N < warp size), Generic2dBlockShape used to
// count every row of ThreadPerBlock_N lanes as its own wave, so a block got 2-4x the threads of
// the shape and its extra waves wrote rows of the next workgroup
// (https://github.com/ROCm/rocm-libraries/issues/15811). The block must have exactly the
// shape's threads, each wave holding warp size / ThreadPerBlock_N rows. Host only, on wave64.

namespace {

// The rmsnorm2d and layernorm2d generators' block shapes: Repeat_M = Repeat_N = 1.
template <ck_tile::index_t ThreadPerBlock_M,
          ck_tile::index_t ThreadPerBlock_N,
          ck_tile::index_t Vector_N>
using NormShape =
    ck_tile::Generic2dBlockShape<ck_tile::sequence<ThreadPerBlock_M, ThreadPerBlock_N * Vector_N>,
                                 ck_tile::sequence<ThreadPerBlock_M, ThreadPerBlock_N>,
                                 ck_tile::sequence<1, Vector_N>>;

struct Expected
{
    ck_tile::index_t block_size;
    ck_tile::index_t warp_per_block_m;
    ck_tile::index_t warp_per_block_n;
    ck_tile::index_t warp_m;
    ck_tile::index_t thread_per_warp_m;
    ck_tile::index_t thread_per_warp_n;
};

template <typename Shape>
void Check(const Expected& expected)
{
    EXPECT_EQ(Shape::BlockSize, expected.block_size);
    EXPECT_EQ(Shape::template GetBlockSize<false>(), expected.block_size);
    EXPECT_EQ(Shape::template GetBlockSize<true>(), expected.block_size);
    EXPECT_EQ(Shape::WarpPerBlock_M, expected.warp_per_block_m);
    EXPECT_EQ(Shape::WarpPerBlock_N, expected.warp_per_block_n);
    EXPECT_EQ(Shape::Warp_M, expected.warp_m);
    EXPECT_EQ(Shape::ThreadPerWarp_M, expected.thread_per_warp_m);
    EXPECT_EQ(Shape::ThreadPerWarp_N, expected.thread_per_warp_n);
}

} // namespace

// {BlockSize, WarpPerBlock_M, WarpPerBlock_N, Warp_M, ThreadPerWarp_M, ThreadPerWarp_N}

TEST(Generic2dBlockShape, RowsNarrowerThanAWave)
{
    // 4 x 16 threads: one wave of 4 rows (it used to be 4 waves, 256 threads).
    Check<NormShape<4, 16, 4>>({64, 1, 1, 4, 4, 16});
    Check<NormShape<4, 16, 8>>({64, 1, 1, 4, 4, 16});
    // 8 x 8 threads: one wave of 8 rows (it used to be 8 waves, 512 threads).
    Check<NormShape<8, 8, 8>>({64, 1, 1, 8, 8, 8});
    // 8 x 32 threads: 4 waves of 2 rows (it used to be 8 waves, 512 threads).
    Check<NormShape<8, 32, 8>>({256, 4, 1, 2, 2, 32});
}

TEST(Generic2dBlockShape, RowsOfAWaveOrWider)
{
    // One wave per row, and one row over several waves, are unchanged.
    Check<NormShape<4, 64, 8>>({256, 4, 1, 1, 1, 64});
    Check<NormShape<1, 256, 8>>({256, 1, 4, 1, 1, 64});
}

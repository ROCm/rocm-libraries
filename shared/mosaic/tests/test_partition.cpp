// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// test_partition.cpp
//
// Tests for include/mosaic/partition.hpp.
//
// `local_tile(tensor, tiler, coord, step)` extracts a workgroup's tile of a
// global tensor, projecting the (rank-K) tiler/coord onto the source's
// (rank-N) modes via the Step<...> mask.
//
// Test layers:
//   (1) Step<...> filtering helpers — drop X-positions from a tuple.
//   (2) local_tile on a 2D source with a 2D tiler (no Step mask
//       projection needed) — simple shape + per-coord offset check.
//   (3) local_tile on a 2D source with a rank-3 tiler + Step mask —
//       the canonical GEMM kernel pattern (A/B/D operand variants).

#include <gtest/gtest.h>

#include <mosaic/layout.hpp>
#include <mosaic/partition.hpp>
#include <mosaic/slice.hpp>
#include <mosaic/tensor.hpp>
#include <set>
#include <vector>

using namespace mosaic;

// =====================================================================
// (1) Step<...> filtering
// =====================================================================

TEST(LocalTile, FilterStep_Drops_X_Positions) {
    auto t = std::make_tuple(Int<10>{}, Int<20>{}, Int<30>{});
    auto kept = detail::filter_step<Step<Int<1>, X, Int<1>>>(t);
    EXPECT_EQ(std::tuple_size_v<decltype(kept)>, 2u);
    EXPECT_EQ(std::get<0>(kept).value, 10);
    EXPECT_EQ(std::get<1>(kept).value, 30);
}

TEST(LocalTile, FilterStep_KeepsAll_When_NoX) {
    auto t = std::make_tuple(Int<5>{}, Int<7>{}, Int<11>{});
    auto kept = detail::filter_step<Step<Int<1>, Int<1>, Int<1>>>(t);
    EXPECT_EQ(std::tuple_size_v<decltype(kept)>, 3u);
}

TEST(LocalTile, FilterStep_DropsAll_When_AllX) {
    auto t = std::make_tuple(Int<5>{}, Int<7>{});
    auto kept = detail::filter_step<Step<X, X>>(t);
    EXPECT_EQ(std::tuple_size_v<decltype(kept)>, 0u);
}

// =====================================================================
// (2) local_tile — rank-2 source, rank-2 tiler, no Step projection needed
// =====================================================================

TEST(LocalTile, Rank2Source_Rank2Tiler_AllKept) {
    // Source: 16x12 row-major. Tiler: (4, 3). Coord: (1, 2) — slice both
    // rest dims.
    std::vector<float> buf(16 * 12);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = float(i);

    auto t = make_tensor(buf.data(), make_layout(make_shape(Int<16>{}, Int<12>{}),
                                                 make_stride(Int<12>{}, Int<1>{})));
    auto tiler = std::make_tuple(Int<4>{}, Int<3>{});
    auto wg = local_tile(t, tiler, std::make_tuple(1, 2), Step<Int<1>, Int<1>>{});

    // After both rest modes sliced, the result should be the 4x3 tile at
    // coord (1, 2). That covers rows [4, 8), cols [6, 9) of the source.
    // For (m=0, k=0): source row 4, col 6 → buf[4*12 + 6] = 54.
    EXPECT_EQ(wg(0, 0), 54.0f);
    EXPECT_EQ(wg(0, 1), 55.0f);
    EXPECT_EQ(wg(0, 2), 56.0f);
    EXPECT_EQ(wg(1, 0), 66.0f);
    EXPECT_EQ(wg(3, 2), buf[7 * 12 + 8]);  // = 92.0f
}

TEST(LocalTile, Rank2Source_KeepsRestMode_WithUnderscore) {
    // Source: 8x12 row-major. Tiler: (4, 3). Coord: (1, _) — keep the K
    // rest mode for iteration. Result shape: (4, 3, K/3=4).
    std::vector<float> buf(8 * 12);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = float(i);

    auto t = make_tensor(
        buf.data(), make_layout(make_shape(Int<8>{}, Int<12>{}), make_stride(Int<12>{}, Int<1>{})));
    auto tiler = std::make_tuple(Int<4>{}, Int<3>{});
    auto wg = local_tile(t, tiler, std::make_tuple(1, _), Step<Int<1>, Int<1>>{});

    // For each k_iter in [0, 4), the (0, 0, k_iter) cell of wg should
    // be the source value at row 4, col k_iter * 3.
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
        auto const expected = buf[4 * 12 + k_iter * 3];
        EXPECT_EQ(wg(0, 0, k_iter), expected) << "k_iter=" << k_iter;
    }
    // And (m=1, k=2, k_iter=3) = source[5][3*3+2] = source[5][11] = 71.
    EXPECT_EQ(wg(1, 2, 3), buf[5 * 12 + 11]);
}

// =====================================================================
// (3) local_tile — Step<> projects a rank-3 tiler onto a rank-2 source
//                  (the canonical GEMM kernel pattern)
// =====================================================================

TEST(LocalTile, GemmPattern_A_Operand) {
    // gA: rank-2 (M=16, K=12). wg_tiler: rank-3 (BLK_M=4, BLK_N, BLK_K=3).
    // Step<_1, X, _1>{} drops the N axis → sub_tiler = (4, 3).
    // Coord (wg_m=1, _, _) → sub_coord = (1, _) — slice M-iter, keep K-iter.
    // Same expected shape (4, 3, 4) as the prior test.
    std::vector<float> buf(16 * 12);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = float(i);

    auto gA = make_tensor(buf.data(), make_layout(make_shape(Int<16>{}, Int<12>{}),
                                                  make_stride(Int<12>{}, Int<1>{})));
    auto wg_tiler = std::make_tuple(Int<4>{}, Int<8>{}, Int<3>{});

    auto wg = local_tile(gA, wg_tiler, std::make_tuple(1, _, _), Step<Int<1>, X, Int<1>>{});

    // Result should be a tensor of shape (4, 3, k_tiles=4) for the
    // M-tile at wg_m=1: covers source rows [4, 8).
    for (int m = 0; m < 4; ++m) {
        for (int k = 0; k < 3; ++k) {
            for (int k_iter = 0; k_iter < 4; ++k_iter) {
                auto const expected = buf[(4 + m) * 12 + k_iter * 3 + k];
                EXPECT_EQ(wg(m, k, k_iter), expected)
                    << "(m=" << m << ", k=" << k << ", k_iter=" << k_iter << ")";
            }
        }
    }
}

TEST(LocalTile, GemmPattern_D_Operand_AllSliced) {
    // gD: rank-2 (M=16, N=12). wg_tiler: rank-3 (BLK_M=4, BLK_N=3, BLK_K).
    // Step<_1, _1, X>{} drops K → sub_tiler = (4, 3).
    // Coord (wg_m=2, wg_n=1, _) → sub_coord = (2, 1) — both rest dims sliced.
    // Result: tensor of shape (4, 3) at the (M=2, N=1) output tile.
    std::vector<float> buf(16 * 12);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = float(i);

    auto gD = make_tensor(buf.data(), make_layout(make_shape(Int<16>{}, Int<12>{}),
                                                  make_stride(Int<12>{}, Int<1>{})));
    auto wg_tiler = std::make_tuple(Int<4>{}, Int<3>{}, Int<8>{});

    auto wg = local_tile(gD, wg_tiler, std::make_tuple(2, 1, _), Step<Int<1>, Int<1>, X>{});

    // wg_m=2 → source rows [8, 12). wg_n=1 → source cols [3, 6).
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 3; ++n) {
            auto const expected = buf[(8 + m) * 12 + (3 + n)];
            EXPECT_EQ(wg(m, n), expected) << "(m=" << m << ", n=" << n << ")";
        }
    }
}

// =====================================================================
// (4) End-to-end coverage check: walking every cell of the kept tile
//     reaches a distinct source offset (bijection on the tile domain).
// =====================================================================

TEST(LocalTile, TileBijection_FixedKIter) {
    // Slice gA at (wg_m=2, _, k_iter=1) for a 16x12 source with
    // (4, *, 3) tiling. The resulting (4, 3) slice (at k_iter fixed)
    // should hit 12 distinct offsets.
    std::vector<int> buf(16 * 12);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = int(i);

    auto gA = make_tensor(buf.data(), make_layout(make_shape(Int<16>{}, Int<12>{}),
                                                  make_stride(Int<12>{}, Int<1>{})));
    auto wg_tiler = std::make_tuple(Int<4>{}, Int<5>{}, Int<3>{});

    auto wg_full = local_tile(gA, wg_tiler, std::make_tuple(2, _, _), Step<Int<1>, X, Int<1>>{});
    // wg_full has shape (4, 3, 4). Fix k_iter=1 by sub-slicing.
    std::set<int> seen;
    for (int m = 0; m < 4; ++m) {
        for (int k = 0; k < 3; ++k) {
            seen.insert(wg_full(m, k, 1));
        }
    }
    EXPECT_EQ(seen.size(), 12u);
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/utils.hpp"

// Test to verify deterministic tile selection when tiles have same latency and arithmetic intensity
// This test ensures that the tie-breaking logic works correctly regardless of input order
TEST(OrigamiTileSelection, DeterministicTieBreaking) {
    // Setup hardware (gfx942)
    auto gfx942arch = origami::hardware_t::arch_name_to_enum("gfx942");
    auto hardware = origami::hardware_t(gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.7, 1,
                                        std::make_tuple(0, 0.015, 0));

    // Square problem size
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    size_t batch = 1;
    bool transA = false;
    bool transB = false;

    // Two tiles with same arithmetic intensity: 256x64x32 and 64x256x32
    // AI = (2 * MT_M * MT_N * MT_K) / (MT_M*MT_K + MT_N*MT_K + MT_M*MT_N)
    // Both have AI = 1048576 / 26624 = 39.38

    // List 1: Tile A first, then Tile B
    const std::vector<origami::tile_tuple> MT_list_A_first = {
        {256, 64, 32, 32, 32, 8, 1, 6, 0, 0},  // Tile A
        {64, 256, 32, 32, 32, 8, 1, 6, 0, 0}   // Tile B
    };

    // List 2: Tile B first, then Tile A (reversed order)
    const std::vector<origami::tile_tuple> MT_list_B_first = {
        {64, 256, 32, 32, 32, 8, 1, 6, 0, 0},  // Tile B
        {256, 64, 32, 32, 32, 8, 1, 6, 0, 0}   // Tile A
    };

    // Call select_best_macro_tile_size with both orderings
    auto results_A_first = origami::select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list_A_first, 16, 16, 16,
        origami::data_type_t::BFloat16, 0, 0.8, false, 6);

    auto results_B_first = origami::select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list_B_first, 16, 16, 16,
        origami::data_type_t::BFloat16, 0, 0.8, false, 6);

    // Extract the best tile from each result
    auto best_tile_A_first = results_A_first[0];
    auto best_tile_B_first = results_B_first[0];

    size_t MT_M_A_first = std::get<1>(best_tile_A_first);
    size_t MT_N_A_first = std::get<2>(best_tile_A_first);
    size_t MT_K_A_first = std::get<3>(best_tile_A_first);

    size_t MT_M_B_first = std::get<1>(best_tile_B_first);
    size_t MT_N_B_first = std::get<2>(best_tile_B_first);
    size_t MT_K_B_first = std::get<3>(best_tile_B_first);

    // Verify deterministic selection: both should select the same tile (256x64x32)
    // regardless of input order, using the final tie-breaker (prefer larger MT_M)
    EXPECT_EQ(MT_M_A_first, MT_M_B_first) << "Selected tile MT_M should be consistent";
    EXPECT_EQ(MT_N_A_first, MT_N_B_first) << "Selected tile MT_N should be consistent";
    EXPECT_EQ(MT_K_A_first, MT_K_B_first) << "Selected tile MT_K should be consistent";

    // Verify it selected the tile with larger MT_M (256 > 64)
    EXPECT_EQ(MT_M_A_first, 256) << "Should prefer tile with larger MT_M";
    EXPECT_EQ(MT_N_A_first, 64) << "Should prefer tile with larger MT_M";
}

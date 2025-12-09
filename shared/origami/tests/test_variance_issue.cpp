// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/utils.hpp"

// Test to verify deterministic tile selection
TEST(OrigamiTileSelection, VarianceEffect) {
    // Setup hardware
    auto gfx950arch = origami::hardware_t::arch_name_to_enum("gfx950");
    // auto hardware = origami::hardware_t(gfx950arch, 256, 163840, 8, 7727.272727, 2993.42, 3157.89, 4194304, 2.2, 4,
    //                                     std::make_tuple(0, 0.008, 0));
    auto hardware = origami::hardware_t(gfx950arch, 256, 163840, 8, 7727.272727, 2000.42, 2000.89, 4194304, 2.2, 4,
                                        std::make_tuple(0, 0.008, 0));

    // Square problem size
    size_t M = 42598;
    size_t N = 153;
    size_t K = 128;
    size_t batch = 1;
    bool transA = false;
    bool transB = true;

    // List 1: Only two tiles
    const std::vector<origami::tile_tuple> MT_list1 = {
        {256, 160, 32, 16, 16, 32, 1, 6, 0, 0},  // Tile A
        {192, 160, 64, 16, 16, 32, 1, 6, 0, 0}   // Tile B
    };

    // List 2: Previous two tiles + a new one
    const std::vector<origami::tile_tuple> MT_list2 = {
        {256, 160, 32, 16, 16, 32, 1, 6, 0, 0},  // Tile A
        {192, 160, 64, 16, 16, 32, 1, 6, 0, 0},  // Tile B
        {192, 160, 32, 16, 16, 32, 1, 6, 0, 0}   // Tile C
    };

    // Call select_best_macro_tile_size with both tile lists
    auto results1 = origami::select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list1, 16, 16, 32,
        origami::data_type_t::BFloat16, 0, 0.8, false, 6);

    auto results2 = origami::select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list2, 16, 16, 32,
        origami::data_type_t::BFloat16, 0, 0.8, false, 6);

    // Extract the best tile from each result
    auto best_tile1 = results1[0];
    auto best_tile2 = results2[0];

    size_t MT_M1 = std::get<1>(best_tile1);
    size_t MT_N1 = std::get<2>(best_tile1);
    size_t MT_K1 = std::get<3>(best_tile1);

    size_t MT_M2 = std::get<1>(best_tile2);
    size_t MT_N2 = std::get<2>(best_tile2);
    size_t MT_K2 = std::get<3>(best_tile2);
    
    // At this time, the model predicts following latencies:
    // TileA: 35803.3
    // TileB: 36088
    // TileC: 35452.4
    // Hence, for list1 TileB is the winner as it has the largest AI.
    // For list2, either TileB (previous winner) or TileC (new tile) should be the winner.
    // Note that adding TileC (with a reasonable variance) eliminates TileB from the
    // pool of selected kernels, and TileC is the winner.
    std::cout << std::get<0>(results2[0]) << " " << std::get<1>(results2[0]) << " " << std::get<3>(results2[0]) << std::endl;
    std::cout << std::get<0>(results2[1]) << " " << std::get<1>(results2[1]) << " " << std::get<3>(results2[1]) << std::endl;
    std::cout << std::get<0>(results2[2]) << " " << std::get<1>(results2[2]) << " " << std::get<3>(results2[2]) << std::endl;
    
    // Verify deterministic selection
    // After adding a new tile, we don't want the selection to change from 
    // tile A to tile B (or the other way around).
    // We accept if the new tile is the winner, though!
    EXPECT_THAT(MT_M2, ::testing::AnyOf(::testing::Eq(MT_M1), ::testing::Eq(std::get<0>(MT_list2[2])))) << "Winner is not acceptable";
    EXPECT_THAT(MT_N2, ::testing::AnyOf(::testing::Eq(MT_N1), ::testing::Eq(std::get<1>(MT_list2[2])))) << "Winner is not acceptable";
    EXPECT_THAT(MT_K2, ::testing::AnyOf(::testing::Eq(MT_K1), ::testing::Eq(std::get<2>(MT_list2[2])))) << "Winner is not acceptable";
}

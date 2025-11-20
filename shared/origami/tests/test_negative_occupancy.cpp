// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/utils.hpp"

TEST(OrigamiTileSelection, NegativeOccupancy) {
    // Setup hardware (gfx942)
    auto gfx942arch = origami::hardware_t::arch_name_to_enum("gfx942");
    auto hardware = origami::hardware_t(gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.7, 1,
                                        std::make_tuple(0, 0.015, 0));
    
    // Square problem size
    size_t M = 32;
    size_t N = 800000;
    size_t K = 16;
    size_t batch = 1;
    bool transA = false;
    bool transB = true;

    // List 1: Tile A first, then Tile B
    const std::vector<origami::tile_tuple> MT_list = {
        {256, 256, 32, 16, 16, 32, -1, 6, 0, 0},  // Tile A
        {32, 256, 16, 32, 32, 8, 2, 6, 0, 0}   // Tile B
    };

    auto results = origami::select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list, 32, 32, 32,
        origami::data_type_t::XFloat32, 0, 0.8, false, 6);
    
    auto best_tile = results[0];
    size_t MT_M = std::get<1>(best_tile);
    size_t MT_N = std::get<2>(best_tile);
    size_t MT_K = std::get<3>(best_tile);
    EXPECT_EQ(MT_M, 32) << "MT_M should be 32";
    EXPECT_EQ(MT_N, 256) << "MT_N should be 256";
    EXPECT_EQ(MT_K, 16) << "MT_K should be 16";
}
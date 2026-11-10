// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Batched Stream-K: the output tiles of all batches form one tile space, so data-parallel tiles
// and Stream-K segments run across batch boundaries. A fixed workgroup budget (num_cu with
// occupancy one) keeps the partition independent of the device.

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_OneTilePerBatch)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // 11 one-tile batches on 8 workgroups: Stream-K only, and every segment boundary inside a
    // tile is a partial tile of some batch.
    this->Run(M_Tile, N_Tile, 10 * K_Tile, 0, 0, 0, /*num_cu=*/8, /*require_split=*/true, 11);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_DataParallelAndStreamK)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // 7 batches of 2 x 3 tiles on 8 workgroups: 32 data-parallel tiles, then 10 Stream-K tiles
    // that span the last two batches.
    this->Run(2 * M_Tile, 3 * N_Tile, 6 * K_Tile, 0, 0, 0, /*num_cu=*/8, /*require_split=*/true, 7);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_PaddedTiles)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // Partial M, N and K tiles in every batch: 5 batches of 3 x 2 tiles on 7 workgroups, then the
    // same on the device's workgroup budget.
    this->Run(2 * M_Tile + 16, N_Tile + 32, 7 * K_Tile + 8, 0, 0, 0, /*num_cu=*/7, true, 5);
    this->Run(2 * M_Tile + 16, N_Tile + 32, 7 * K_Tile + 8, 0, 0, 0, 0, false, 5);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_SingleBatch)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // One batch is the single-GEMM partition.
    this->Run(M_Tile, 3 * N_Tile, 9 * K_Tile, 0, 0, 0, /*num_cu=*/4, /*require_split=*/true, 1);
}

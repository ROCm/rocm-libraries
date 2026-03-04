// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include <gtest/gtest.h>
#include "test_mx_flatmm_fixtures.hpp"

// FP8 x FP8 -> FP16
// N_Tile = 256, K must be a multiple of 32.
// clang-format off
using FP8FP8Types = ::testing::Types<
    std::tuple<FP8, FP8, FP16, MXFlatmm_GFX950_FP8FP8_Traits>
>;
// clang-format on

TYPED_TEST_SUITE(TestMXFlatmm, FP8FP8Types);

TYPED_TEST(TestMXFlatmm, SmallMNK) { this->run_test_with_validation(128, 256, 256); }

TYPED_TEST(TestMXFlatmm, MediumMNK) { this->run_test_with_validation(256, 512, 512); }

// K=768 -> num_loop=3: has_hot_loop=true, tail=ODD
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopOdd) { this->run_test_with_validation(128, 256, 768); }

// K=1024 -> num_loop=4: has_hot_loop=true, tail=EVEN
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopEven) { this->run_test_with_validation(128, 256, 1024); }

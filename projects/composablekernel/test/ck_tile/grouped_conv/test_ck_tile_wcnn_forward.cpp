// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test_wcnn_forward_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;

using I1  = ck_tile::number<1>;
using I2  = ck_tile::number<2>;
using I4  = ck_tile::number<4>;
using I8  = ck_tile::number<8>;
using I16 = ck_tile::number<16>;
using I32 = ck_tile::number<32>;

// clang-format off
using KernelTypesWcnnFwd = ::testing::Types<
    //         InType, WeiType, AccType, OutType, HPerBlock, WPerBlock, CPerBlock, KPerBlock, HPerWcnn, WPerWcnn, WarpsInH, WarpsInW, WarpsInK
    std::tuple<  F16,    F16,     F32,     F16,       I4,       I2,        I8,       I16,       I4,       I2,       I1,       I1,       I1>,
    std::tuple<  F16,    F16,     F32,     F16,       I8,       I4,        I8,       I16,       I4,       I2,       I2,       I2,       I1>,
    std::tuple<  F16,    F16,     F32,     F16,       I16,      I8,        I8,       I16,       I4,       I2,       I2,       I2,       I1>,
    std::tuple<  F16,    F16,     F16,     F16,       I16,      I8,        I8,       I16,       I4,       I4,       I2,       I2,       I1>,
    std::tuple<  F16,    F16,     F16,     F16,       I16,      I16,       I8,       I16,       I8,       I4,       I2,       I2,       I1>
>;
// clang-format on

template <typename T>
class TestCkTileWcnnFwd : public TestCkTileWcnnForward<T>
{
};

TYPED_TEST_SUITE(TestCkTileWcnnFwd, KernelTypesWcnnFwd);

TYPED_TEST(TestCkTileWcnnFwd, SingleTile)
{
    // G=1, N=1, K=KPerBlock, C=CPerBlock, Hi=HPerBlock, Wi=WPerBlock
    this->Run(1,
              1,
              TestFixture::KPerBlock,
              TestFixture::CPerBlock,
              TestFixture::HPerBlock,
              TestFixture::WPerBlock);
}

TYPED_TEST(TestCkTileWcnnFwd, MultiCLoop)
{
    // C > CPerBlock to exercise the C-loop
    this->Run(1,
              1,
              TestFixture::KPerBlock,
              TestFixture::CPerBlock * 4,
              TestFixture::HPerBlock,
              TestFixture::WPerBlock);
}

TYPED_TEST(TestCkTileWcnnFwd, MultiBlock)
{
    // C > CPerBlock to exercise the C-loop
    this->Run(1,
              1,
              TestFixture::KPerBlock,
              TestFixture::CPerBlock * 4,
              TestFixture::HPerBlock * 4,
              TestFixture::WPerBlock * 8);
}

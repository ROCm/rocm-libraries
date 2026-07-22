// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmQuick : public TestCkTileMxGemmPipeline<T, TestCkTileMxGemmQuick<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

using KernelTypesMxQuick = ::testing::Types<std::tuple<Row,
                                                       Col,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I64,
                                                       I64,
                                                       I256,
                                                       I16,
                                                       I16,
                                                       CompAsync,
                                                       I32>,
                                            std::tuple<Row,
                                                       Row,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I128,
                                                       I128,
                                                       I256,
                                                       I32,
                                                       I32,
                                                       CompAsync,
                                                       I32>,
                                            std::tuple<Col,
                                                       Row,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I128,
                                                       I128,
                                                       I256,
                                                       I32,
                                                       I32,
                                                       CompAsync,
                                                       I32>,
                                            std::tuple<Col,
                                                       Col,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I128,
                                                       I128,
                                                       I256,
                                                       I32,
                                                       I32,
                                                       CompAsync,
                                                       I32>,
                                            std::tuple<Row,
                                                       Col,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I128,
                                                       I256,
                                                       I256,
                                                       I16,
                                                       I16,
                                                       WeightPreshuffle,
                                                       I32>,
                                            std::tuple<Row,
                                                       Col,
                                                       Row,
                                                       F8,
                                                       F8,
                                                       E8M0,
                                                       E8M0,
                                                       F32,
                                                       F16,
                                                       I128,
                                                       I256,
                                                       I256,
                                                       I16,
                                                       I16,
                                                       WeightPreshuffle,
                                                       I32,
                                                       std::true_type>>;

TYPED_TEST_SUITE(TestCkTileMxGemmQuick, KernelTypesMxQuick);

TYPED_TEST(TestCkTileMxGemmQuick, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmQuick, DoubleK)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, 2 * TestFixture::K_Tile);
}

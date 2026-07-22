// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_batched_gemm_util.hpp"

using F8  = ck_tile::fp8_t;
using F16 = ck_tile::half_t;
using F32 = float;
using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using KernelTypesQuick = ::testing::Types<std::tuple<Row, Row, Row, F16, F16, F32, F16>,
                                          std::tuple<Row, Col, Row, F16, F16, F32, F16>,
                                          std::tuple<Col, Row, Row, F16, F16, F32, F16>,
                                          std::tuple<Col, Col, Row, F16, F16, F32, F16>,
                                          std::tuple<Row, Row, Row, F8, F8, F32, F16>,
                                          std::tuple<Row, Col, Row, F8, F8, F32, F16>,
                                          std::tuple<Col, Row, Row, F8, F8, F32, F16>,
                                          std::tuple<Col, Col, Row, F8, F8, F32, F16>>;

template <typename Tuple>
class TestCkTileBatchedGemmQuick : public TestCkTileBatchedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileBatchedGemmQuick, KernelTypesQuick);

TYPED_TEST(TestCkTileBatchedGemmQuick, Batch3K64)
{
    this->Run(256, 256, 64, 0, 0, 0, 256 * 64, 64 * 256, 256 * 256, 3);
}

TYPED_TEST(TestCkTileBatchedGemmQuick, Batch2K128)
{
    this->Run(256, 256, 128, 0, 0, 0, 256 * 128, 128 * 256, 256 * 256, 2);
}

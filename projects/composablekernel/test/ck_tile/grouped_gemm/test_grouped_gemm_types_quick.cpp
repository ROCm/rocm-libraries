// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using F8    = ck_tile::fp8_t;
using F16   = ck_tile::half_t;
using F32   = float;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

using KernelTypesQuick = ::testing::Types<std::tuple<Row, Col, Row, F16, F16, F32, F16, True>,
                                          std::tuple<Row, Col, Row, F8, F8, F32, F16, False>,
                                          std::tuple<Col, Col, Row, F16, F16, F32, F16, False>,
                                          std::tuple<Col, Col, Row, F8, F8, F32, F16, True>,
                                          std::tuple<Row, Row, Row, F16, F16, F32, F16, True>,
                                          std::tuple<Row, Row, Row, F8, F8, F32, F16, False>,
                                          std::tuple<Col, Row, Row, F16, F16, F32, F16, False>,
                                          std::tuple<Col, Row, Row, F8, F8, F32, F16, True>>;

template <typename Tuple>
class TestCkTileGroupedGemmQuick : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmQuick, KernelTypesQuick);

TYPED_TEST(TestCkTileGroupedGemmQuick, MixedShapes)
{
    constexpr int group_count = 4;
    std::vector<int> ms{64, 128, 192, 256};
    std::vector<int> ns{128, 192, 256, 320};
    std::vector<int> ks{64, 96, 128, 160};
    std::vector<int> stride_as(group_count, 0);
    std::vector<int> stride_bs(group_count, 0);
    std::vector<int> stride_cs(group_count, 0);
    this->Run(ms, ns, ks, stride_as, stride_bs, stride_cs, 1, group_count);
}

TYPED_TEST(TestCkTileGroupedGemmQuick, ZeroM)
{
    constexpr int group_count = 4;
    std::vector<int> ms{128, 0, 256, 0};
    std::vector<int> ns{128, 192, 256, 320};
    std::vector<int> ks{128, 128, 256, 256};
    std::vector<int> stride_as(group_count, 0);
    std::vector<int> stride_bs(group_count, 0);
    std::vector<int> stride_cs(group_count, 0);
    this->Run(ms, ns, ks, stride_as, stride_bs, stride_cs, 1, group_count);
}

TYPED_TEST(TestCkTileGroupedGemmQuick, SplitK)
{
    constexpr int group_count = 4;
    std::vector<int> ms{128, 192, 256, 320};
    std::vector<int> ns{128, 192, 256, 320};
    std::vector<int> ks{128, 192, 256, 320};
    std::vector<int> stride_as(group_count, 0);
    std::vector<int> stride_bs(group_count, 0);
    std::vector<int> stride_cs(group_count, 0);
    this->Run(ms, ns, ks, stride_as, stride_bs, stride_cs, 2, group_count);
}

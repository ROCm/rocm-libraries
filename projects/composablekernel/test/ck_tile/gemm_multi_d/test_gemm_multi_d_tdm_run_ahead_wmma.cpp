// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-D GEMM with the run-ahead TDM pipeline (CompTDMRunAhead) and TdmMultiDEpilogue, full tile
// and staged, and the row-broadcast D0 path of TdmMultiDEpilogue (DsRowBroadcastMask) with the
// run-ahead and CompTDMV1 pipelines (gfx1250).

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_d_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using NoTransC = std::false_type;

using Tile64  = ck_tile::sequence<64, 64, 32>;
using Tile128 = ck_tile::sequence<128, 128, 64>;
using Tile256 = ck_tile::sequence<256, 256, 32>;

using D0Full = std::false_type;
using D0Row  = std::true_type;

// clang-format off
using KernelTypes = ::testing::Types<
    //          ALayout, BLayout, D0Layout, D1Layout, ELayout, ADataType, BDataType, D0DataType, D1DataType, AccDataType, EDataType, CDElementWiseFn,   Pipeline,          TransposeC, Tile,    D0
    // Run-ahead pipeline, full D tensors. Tile64 (K = one warp tile) takes the pipeline's odd
    // sub-tile path, Tile128 the even one.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile64,  D0Full>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile128, D0Full>,
    std::tuple<    Row,     Row,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile128, D0Full>,
    std::tuple<    Col,     Row,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile128, D0Full>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmRA,       NoTransC,   Tile128, D0Full>,
    // fp32 E also instantiates the (unreachable) atomic-add epilogue branch of the kernel.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmRA,       NoTransC,   Tile64,  D0Full>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRAStaged, NoTransC,   Tile128, D0Full>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRAStaged, NoTransC,   Tile256, D0Full>,

    // Row-broadcast D0 (a bias), D1 a full tensor.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile64,  D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile128, D0Row>,
    std::tuple<    Col,     Row,     Row,      Row,      Row,      BF16,      BF16,      BF16,       BF16,       F32,        BF16, ElementWiseAddAdd, MultiDTdmRA,       NoTransC,   Tile128, D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmRA,       NoTransC,   Tile128, D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmRAStaged, NoTransC,   Tile128, D0Row>,
    std::tuple<    Row,     Row,     Row,      Row,      Row,      BF16,      BF16,      BF16,       BF16,       F32,        BF16, ElementWiseAddAdd, MultiDTdmRAStaged, NoTransC,   Tile256, D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1,       NoTransC,   Tile128, D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1,       std::true_type, Tile64, D0Row>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1Staged, NoTransC,   Tile128, D0Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiD, KernelTypes);

#include "test_gemm_multi_d_tdm_cases.inc"

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-D GEMM with the TDM pipelines (CompTDMV1 / CompTDMV2 / producer/consumer) and
// TdmMultiDEpilogue, full tile and staged (gfx1250).

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
using TransC   = std::true_type;

using Tile64  = ck_tile::sequence<64, 64, 32>;
using Tile128 = ck_tile::sequence<128, 128, 64>;
using Tile256 = ck_tile::sequence<256, 256, 32>;

// clang-format off
using KernelTypes = ::testing::Types<
    //          ALayout, BLayout, D0Layout, D1Layout, ELayout, ADataType, BDataType, D0DataType, D1DataType, AccDataType, EDataType, CDElementWiseFn,   Pipeline,    TransposeC, Tile
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, TransC,     Tile128>,

    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile128>,

    // fp32 E also instantiates the (unreachable) atomic-add epilogue branch of the kernel.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile128>,

    // The producer/consumer pipeline: only its math waves run TdmMultiDEpilogue.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmPC, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmPC, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmPC, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmPC, TransC,     Tile64>,

    // Staged Multi-D TdmEpilogue
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,      BF16,      F32,        F16,  ElementWiseAddAdd, MultiDTdmV1Staged, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,      BF16,      F32,        F16,  ElementWiseAddAdd, MultiDTdmV1Staged, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,      BF16,      F32,        F16,  ElementWiseAddAdd, MultiDTdmV1Staged, NoTransC,   Tile256>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,      BF16,      F32,        F16,  ElementWiseAddAdd, MultiDTdmV1Staged, TransC,     Tile256>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,       F16,       F32,        F16,  MultiplyMultiply,  MultiDTdmV1Staged, TransC,     Tile256>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,       F32,       F32,        F32,  MultiplyMultiply,  MultiDTdmV1Staged, NoTransC,   Tile128>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiD, KernelTypes);

#include "test_gemm_multi_d_tdm_cases.inc"

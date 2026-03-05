// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <tuple>
#include <type_traits>
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "test_gemm_streamk_util.hpp"

using F8   = ck_tile::fp8_t;
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using Mem    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;

using I32  = ck_tile::number<32>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

// clang-format off
using KernelTypesStreamKFp16Persistent = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  Persistent    Pipeline

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     Persistent,   Mem>
>;

using KernelTypesStreamKBf16Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    Persistent,   Mem>
>;

using KernelTypesStreamKBf8Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    Persistent,   Mem>
>;

using KernelTypesStreamKFp8Persistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,    Persistent,   Mem>
>;

using KernelTypesStreamKFp16NonPersistent = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  Persistent    Pipeline

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,     NonPersistent,   Mem>
>;

using KernelTypesStreamKBf16NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,    NonPersistent,   Mem>
>;

using KernelTypesStreamKBf8NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Row,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,    NonPersistent,   Mem>
>;

using KernelTypesStreamKFp8NonPersistent = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   Mem>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,    NonPersistent,   Mem>
>;

// clang-format on

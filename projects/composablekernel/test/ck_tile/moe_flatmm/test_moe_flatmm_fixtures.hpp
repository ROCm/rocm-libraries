// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

// Convenience dtype aliases used in typed-test tuples.
using FP16 = ck_tile::half_t;
using BF16 = ck_tile::bfloat16_t;
using FP8  = ck_tile::fp8_t;
using BF8  = ck_tile::bf8_t;
using FP4  = ck_tile::pk_fp4_t;

// Wrap a MoeFlatmmKind value as a type so it can appear inside ::testing::Types tuples.
template <ck_tile::MoeFlatmmKind Kind>
struct MoeKind
{
    static constexpr ck_tile::MoeFlatmmKind value = Kind;
};

using GateOnly = MoeKind<ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>;
using GateUp   = MoeKind<ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>;
using Gemm2    = MoeKind<ck_tile::MoeFlatmmKind::kFFN_gemm2>;
using SplitK   = MoeKind<ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k>;

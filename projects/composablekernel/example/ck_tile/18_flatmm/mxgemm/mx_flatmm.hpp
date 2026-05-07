// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "mx_flatmm_arch_traits.hpp"

// FlatmmConfig types are now defined via CurrentArchTraits:
//   - MXfp4_FlatmmConfig16  = CurrentArchTraits::Fp4Fp4Config
//   - MXfp6_FlatmmConfig16  = CurrentArchTraits::Fp6Fp6Config
//   - MXfp8_FlatmmConfig16  = CurrentArchTraits::Fp8Fp8Config
//   - MXf8f4_FlatmmConfig16 = CurrentArchTraits::F8F4Config
//   - MXf4f8_FlatmmConfig16 = CurrentArchTraits::F4F8Config
//
// For GFX1250 TDM: these map to MXFlatmmConfigBase32TDM (32x32 warp tile)
// For others:  these map to MXFlatmmConfigBase16 (16x16 warp tile)
// GEMM config with 16x16 warp tile

template <typename MXFlatmmArchTraits,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise,
          bool Splitk,
          bool HasHotLoop,
          ck_tile::TailNumber TailNum>
float mx_flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                     const ck_tile::stream_config& s);

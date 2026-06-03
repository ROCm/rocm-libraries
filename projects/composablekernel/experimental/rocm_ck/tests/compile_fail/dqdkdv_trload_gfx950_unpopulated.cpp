// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: gfx950 is TrLoad-eligible, but the gfx950 TrLoad tile table is
// intentionally not populated yet (this is the non-TrLoad scope). getTileConfig
// rejects tr_load == true for gfx950 at compile time until the table lands.
// Expected error: "gfx950 TrLoad tile configs are not yet populated"
//
// NOTE: the compile_fail harness only asserts the TU fails to build (WILL_FAIL);
// it does not match this message. (hdim_q, hdim_v) are symmetric and valid so
// the not-yet-populated throw is the only reachable failure.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx950, /*tr_load=*/true);

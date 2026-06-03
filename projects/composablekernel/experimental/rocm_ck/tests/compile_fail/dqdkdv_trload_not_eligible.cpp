// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: TrLoad is only available on gfx950. Requesting tr_load == true for
// a non-eligible gfx9 target (gfx942) is rejected at compile time by
// getTileConfig via TargetSet::trload_eligible().
// Expected error: "TrLoad is only available on gfx950"
//
// NOTE: the compile_fail harness only asserts the TU fails to build (WILL_FAIL);
// it does not match this message. (hdim_q, hdim_v) are symmetric and valid so
// the trload eligibility throw is the only reachable failure.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942, /*tr_load=*/true);

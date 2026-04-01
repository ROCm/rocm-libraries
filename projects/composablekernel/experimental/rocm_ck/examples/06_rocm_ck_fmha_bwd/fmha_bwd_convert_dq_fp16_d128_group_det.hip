// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/convert_dq_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// FP16, hdim_q=128, group, deterministic. Paired with det dQ/dK/dV.
// All modes use generic rocm_ck::Args (no per-mode Args type).
// Compilation proof only — numerical verification deferred.
static constexpr rocm_ck::FmhaBwdConvertDQSpec spec =
    rocm_ck::fmha_bwd_convert_dq_variant_spec("fmha_bwd_convert_dq_fp16_d128_group_det");

extern "C" __global__ __launch_bounds__(spec.block_size, spec.block_per_cu)
void fmha_bwd_convert_dq_fp16_d128_group_det(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdConvertDQ<spec>(args);
}

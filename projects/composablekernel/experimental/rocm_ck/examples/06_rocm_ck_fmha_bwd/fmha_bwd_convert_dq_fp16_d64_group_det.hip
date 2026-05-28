// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/convert_dq_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// FP16, hdim_q=64, group, deterministic. Will pair with fp16 group det dQ/dK/dV.
// Compilation proof only -- numerical verification deferred.
static constexpr rocm_ck::FmhaBwdConvertDQSpec spec =
    rocm_ck::fmha_bwd_convert_dq_variant_spec("fmha_bwd_convert_dq_fp16_d64_group_det");

extern "C" __global__ __launch_bounds__(
    spec.block_size,
    spec.block_per_cu) void fmha_bwd_convert_dq_fp16_d64_group_det(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdConvertDQ<spec>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/dqdkdv_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// FP16, hdim_q/v=64, batch, no mask/dropout/bias/deterministic.
static constexpr rocm_ck::FmhaBwdDQDKDVSpec spec =
    rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d64_batch");

extern "C" __global__
__launch_bounds__(spec.block_size,
                  spec.block_per_cu) void fmha_bwd_dqdkdv_fp16_d64_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdDQDKDV<spec>(args);
}

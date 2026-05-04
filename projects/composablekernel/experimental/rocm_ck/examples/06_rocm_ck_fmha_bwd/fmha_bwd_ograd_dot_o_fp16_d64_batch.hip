// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// FP16, hdim_v=64, batch mode, padded. hdim axis variant.
static constexpr rocm_ck::FmhaBwdOGradDotOSpec spec =
    rocm_ck::fmha_bwd_ograd_dot_o_variant_spec("fmha_bwd_ograd_dot_o_fp16_d64_batch");

extern "C" __global__
__launch_bounds__(spec.block_size,
                  spec.block_per_cu) void fmha_bwd_ograd_dot_o_fp16_d64_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdOGradDotO<spec>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// BF16, hdim_v=128, batch mode, padded. dtype axis variant.
static constexpr rocm_ck::FmhaBwdOGradDotOSpec spec =
    rocm_ck::fmha_bwd_ograd_dot_o_variant_spec("fmha_bwd_ograd_dot_o_bf16_d128_batch");

extern "C" __global__
__launch_bounds__(spec.block_size,
                  spec.block_per_cu) void fmha_bwd_ograd_dot_o_bf16_d128_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdOGradDotO<spec>(args);
}

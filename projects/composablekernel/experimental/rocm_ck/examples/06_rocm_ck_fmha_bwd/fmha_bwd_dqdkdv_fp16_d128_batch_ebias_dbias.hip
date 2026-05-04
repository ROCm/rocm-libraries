// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/dqdkdv_dev.hpp>

#include "rocm_fmha_bwd_registry.hpp"

// FP16, hdim_q/v=128, batch, elementwise bias + bias gradient.
// Compilation proof only -- numerical verification deferred (NV-3).
static constexpr rocm_ck::FmhaBwdDQDKDVSpec spec =
    rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_ebias_dbias");

extern "C" __global__ __launch_bounds__(
    spec.block_size,
    spec.block_per_cu) void fmha_bwd_dqdkdv_fp16_d128_batch_ebias_dbias(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdDQDKDV<spec>(args);
}

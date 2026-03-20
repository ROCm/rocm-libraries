// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_dev.hpp"

// FP16, hdim_v=128, batch mode, no padding. No-pad axis variant.
// Use when seqlen_q is a multiple of block_size and hdim_v matches exactly.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::FmhaBwdOGradDotOConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_v = 128, .mode = rocm_ck::FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});

extern "C" __global__ __launch_bounds__(kernel.block_size, kernel.block_per_cu)
void fmha_bwd_ograd_dot_o_fp16_d128_batch_npad(rocm_ck::FmhaBwdOGradDotOBatchArgs args)
{
    rocm_ck::runFmhaBwdOGradDotO<kernel>(args);
}

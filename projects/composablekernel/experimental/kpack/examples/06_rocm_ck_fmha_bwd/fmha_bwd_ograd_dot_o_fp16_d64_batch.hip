// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_dev.hpp"

// FP16, hdim_v=64, batch mode, padded. Smaller head dimension axis variant.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::FmhaBwdOGradDotOConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_v = 64, .mode = rocm_ck::FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

extern "C" __global__ __launch_bounds__(kernel.block_size, kernel.block_per_cu)
void fmha_bwd_ograd_dot_o_fp16_d64_batch(rocm_ck::FmhaBwdOGradDotOBatchArgs args)
{
    rocm_ck::runFmhaBwdOGradDotO<kernel>(args);
}

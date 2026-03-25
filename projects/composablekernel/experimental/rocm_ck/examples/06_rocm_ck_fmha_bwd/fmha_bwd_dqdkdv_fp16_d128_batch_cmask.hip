// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_dqdkdv_dev.hpp"

// FP16, hdim_q/v=128, batch, causal mask. Mask axis variant.
// Compilation proof only — numerical verification deferred.
static constexpr rocm_ck::FmhaBwdDQDKDVKernel kernel =
    rocm_ck::make_kernel(
    rocm_ck::FmhaBwdDQDKDVConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = rocm_ck::FmhaMode::BATCH},
        .algorithm = {.has_mask = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}});

extern "C" __global__ __launch_bounds__(kernel.block_size, kernel.block_per_cu)
void fmha_bwd_dqdkdv_fp16_d128_batch_cmask(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdDQDKDV<kernel>(args);
}

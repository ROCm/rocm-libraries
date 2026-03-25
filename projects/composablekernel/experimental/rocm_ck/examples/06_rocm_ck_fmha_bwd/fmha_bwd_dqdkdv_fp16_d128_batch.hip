// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_dqdkdv_dev.hpp"

// FP16, hdim_q/v=128, batch, no mask/dropout/bias/deterministic. Baseline.
static constexpr rocm_ck::FmhaBwdDQDKDVKernel kernel =
    rocm_ck::make_kernel(
    rocm_ck::FmhaBwdDQDKDVConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = rocm_ck::FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

extern "C" __global__ __launch_bounds__(256, 1)
void fmha_bwd_dqdkdv_fp16_d128_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdDQDKDV<kernel>(args);
}

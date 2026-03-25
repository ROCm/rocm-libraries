// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_convert_dq_dev.hpp"

// FP16, hdim_q=128, batch, deterministic. Paired with det dQ/dK/dV.
// Compilation proof only — numerical verification deferred.
static constexpr rocm_ck::FmhaBwdConvertDQKernel kernel =
    rocm_ck::make_kernel(
    rocm_ck::FmhaBwdConvertDQConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_q = 128,
                      .mode = rocm_ck::FmhaMode::BATCH},
        .algorithm = {}});

extern "C" __global__ __launch_bounds__(kernel.block_size, kernel.block_per_cu)
void fmha_bwd_convert_dq_fp16_d128_batch_det(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdConvertDQ<kernel>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_convert_dq_dev.hpp"

// FP16, hdim_q=128, group, deterministic. Paired with det dQ/dK/dV.
// All modes use generic rocm_ck::Args (no per-mode Args type).
// Compilation proof only — numerical verification deferred.
static constexpr rocm_ck::FmhaBwdConvertDQKernel kernel =
    rocm_ck::make_kernel(
    rocm_ck::FmhaBwdConvertDQConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16,
                      .hdim_q = 128,
                      .mode = rocm_ck::FmhaMode::GROUP},
        .algorithm = {}});

extern "C" __global__ __launch_bounds__(kernel.block_size, kernel.block_per_cu)
void fmha_bwd_convert_dq_fp16_d128_group_det(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdConvertDQ<kernel>(args);
}

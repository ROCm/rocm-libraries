// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_ograd_dot_o_dev.hpp"

static constexpr rocm_ck::FmhaBwdOGradDotOKernel kernel =
    rocm_ck::make_kernel(
        rocm_ck::FmhaBwdOGradDotOConfig{
            .signature = {.dtype = rocm_ck::DataType::BF16,
                          .hdim_v = 128,
                          .mode = rocm_ck::FmhaMode::BATCH},
            .algorithm = {.pad_seqlen_q = true,
                          .pad_hdim_v = true}});

extern "C" __global__ void
fmha_bwd_ograd_dot_o_bf16_d128_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdOGradDotO<kernel>(args);
}

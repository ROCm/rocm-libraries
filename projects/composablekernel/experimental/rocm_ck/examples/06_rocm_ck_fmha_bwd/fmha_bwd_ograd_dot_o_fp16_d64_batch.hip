// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_fmha_bwd_ograd_dot_o_dev.hpp"

static constexpr rocm_ck::FmhaBwdOGradDotOKernel kernel =
    rocm_ck::make_kernel(
        rocm_ck::FmhaBwdOGradDotOConfig{
            .signature = {.dtype = rocm_ck::DataType::FP16,
                          .hdim_v = 64,
                          .mode = rocm_ck::FmhaMode::BATCH},
            .algorithm = {.pad_seqlen_q = true,
                          .pad_hdim_v = true}});

extern "C" __global__ void
fmha_bwd_ograd_dot_o_fp16_d64_batch(rocm_ck::Args args)
{
    rocm_ck::runFmhaBwdOGradDotO<kernel>(args);
}

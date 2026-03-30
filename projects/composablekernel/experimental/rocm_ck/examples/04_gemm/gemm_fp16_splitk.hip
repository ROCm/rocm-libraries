// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_variants.hpp"
#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::gemm_variant_spec("gemm_fp16_splitk");

extern "C" __global__ void gemm_fp16_splitk(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

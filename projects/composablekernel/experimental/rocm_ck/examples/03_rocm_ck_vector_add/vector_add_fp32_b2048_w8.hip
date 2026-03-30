// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "vector_add_variants.hpp"
#include <rocm_ck/elementwise_dev.hpp>

static constexpr rocm_ck::ElementwiseSpec spec =
    rocm_ck::vector_add_variant_spec("vector_add_fp32_b2048_w8");

extern "C" __global__ void vector_add_fp32_b2048_w8(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

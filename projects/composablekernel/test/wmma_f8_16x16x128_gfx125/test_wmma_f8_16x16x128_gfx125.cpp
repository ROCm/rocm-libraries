// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-only: every FP8/BF8 16x16x128 dense WMMA wrapper in CK builds for every gfx125 target.
// gfx1250-strict has no V_WMMA_F{16,32}_16X16X128_{FP8,BF8}_{FP8,BF8}, so the wrappers must not
// emit those builtins there (ROCM-35020).

#include "ck/utility/amd_wmma.hpp"

namespace {

template <typename Intrin, typename AVec, typename BVec, typename CType>
__device__ void run_wmma(void* p)
{
    auto* a = static_cast<AVec*>(p);
    auto* b = reinterpret_cast<BVec*>(a + 1);
    auto* c = reinterpret_cast<ck::vector_type<CType, 8>*>(b + 1);
    Intrin::Run(*a, *b, *c);
}

} // namespace

__global__ void test_wmma_f8_16x16x128_gfx125(void* p)
{
    using ck::bf8x64_t;
    using ck::f8x64_t;
    using ck::half_t;
    run_wmma<ck::intrin_wmma_f32_16x16x128_f8f8<16, 16>, f8x64_t, f8x64_t, float>(p);
    run_wmma<ck::intrin_wmma_f32_16x16x128_f8bf8<16, 16>, f8x64_t, bf8x64_t, float>(p);
    run_wmma<ck::intrin_wmma_f32_16x16x128_bf8f8<16, 16>, bf8x64_t, f8x64_t, float>(p);
    run_wmma<ck::intrin_wmma_f32_16x16x128_bf8bf8<16, 16>, bf8x64_t, bf8x64_t, float>(p);
    run_wmma<ck::intrin_wmma_f16_16x16x128_f8f8<16, 16>, f8x64_t, f8x64_t, half_t>(p);
    run_wmma<ck::intrin_wmma_f16_16x16x128_f8bf8<16, 16>, f8x64_t, bf8x64_t, half_t>(p);
    run_wmma<ck::intrin_wmma_f16_16x16x128_bf8f8<16, 16>, bf8x64_t, f8x64_t, half_t>(p);
    run_wmma<ck::intrin_wmma_f16_16x16x128_bf8bf8<16, 16>, bf8x64_t, bf8x64_t, half_t>(p);
}

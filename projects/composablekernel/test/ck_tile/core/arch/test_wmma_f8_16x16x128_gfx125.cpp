// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-only: every FP8/BF8 16x16x128 dense WMMA wrapper in CK and CK Tile builds for every
// gfx125 target. gfx1250-strict has no V_WMMA_F{16,32}_16X16X128_{FP8,BF8}_{FP8,BF8}, so the
// wrappers must not emit those builtins there (ROCM-35020).

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace {

namespace mma    = ck_tile::core::arch::mma;
using Target1250 = decltype(ck_tile::core::arch::make_amdgcn_gfx1250_target<
                            ck_tile::core::arch::amdgcn_target_id::GFX1250>());

template <typename A, typename B, typename C>
__device__ void tile_mma(void* p)
{
    using Op = mma::amdgcn_mma<A, B, C, 16u, 16u, 128u, Target1250, mma::MmaOpFamily::DENSE>;
    auto* a  = static_cast<typename Op::AVecType*>(p);
    auto* b  = reinterpret_cast<typename Op::BVecType*>(a + 1);
    auto* c  = reinterpret_cast<typename Op::CVecType*>(b + 1);
    *c       = Op::exec(*a, *b, *c);
}

template <typename A, typename B>
__device__ void warp_wmma(void* p)
{
    using Traits = ck_tile::WmmaTraits<ck_tile::gfx125_t, A, B, float, 16, 16, 128>;
    auto* a      = static_cast<typename Traits::AVecType*>(p);
    auto* b      = reinterpret_cast<typename Traits::BVecType*>(a + 1);
    auto* c      = reinterpret_cast<typename Traits::CVecType*>(b + 1);
    *c           = Traits::template wmma_intrinsic<>(*a, *b, *c);
}

template <typename Intrin, typename AVec, typename BVec, typename CType>
__device__ void ck_wmma(void* p)
{
    auto* a = static_cast<AVec*>(p);
    auto* b = reinterpret_cast<BVec*>(a + 1);
    auto* c = reinterpret_cast<ck::vector_type<CType, 8>*>(b + 1);
    Intrin::Run(*a, *b, *c);
}

} // namespace

__global__ void test_wmma_f8_16x16x128_gfx125(void* p)
{
    using ck_tile::bf8_t;
    using ck_tile::fp16_t;
    using ck_tile::fp32_t;
    using ck_tile::fp8_t;
    tile_mma<fp8_t, fp8_t, fp32_t>(p);
    tile_mma<fp8_t, bf8_t, fp32_t>(p);
    tile_mma<bf8_t, fp8_t, fp32_t>(p);
    tile_mma<bf8_t, bf8_t, fp32_t>(p);
    tile_mma<fp8_t, fp8_t, fp16_t>(p);
    tile_mma<fp8_t, bf8_t, fp16_t>(p);
    tile_mma<bf8_t, fp8_t, fp16_t>(p);
    tile_mma<bf8_t, bf8_t, fp16_t>(p);
    warp_wmma<fp8_t, fp8_t>(p);
    warp_wmma<fp8_t, bf8_t>(p);
    warp_wmma<bf8_t, fp8_t>(p);
    warp_wmma<bf8_t, bf8_t>(p);

    using ck::bf8x64_t;
    using ck::f8x64_t;
    using ck::half_t;
    ck_wmma<ck::intrin_wmma_f32_16x16x128_f8f8<16, 16>, f8x64_t, f8x64_t, float>(p);
    ck_wmma<ck::intrin_wmma_f32_16x16x128_f8bf8<16, 16>, f8x64_t, bf8x64_t, float>(p);
    ck_wmma<ck::intrin_wmma_f32_16x16x128_bf8f8<16, 16>, bf8x64_t, f8x64_t, float>(p);
    ck_wmma<ck::intrin_wmma_f32_16x16x128_bf8bf8<16, 16>, bf8x64_t, bf8x64_t, float>(p);
    ck_wmma<ck::intrin_wmma_f16_16x16x128_f8f8<16, 16>, f8x64_t, f8x64_t, half_t>(p);
    ck_wmma<ck::intrin_wmma_f16_16x16x128_f8bf8<16, 16>, f8x64_t, bf8x64_t, half_t>(p);
    ck_wmma<ck::intrin_wmma_f16_16x16x128_bf8f8<16, 16>, bf8x64_t, f8x64_t, half_t>(p);
    ck_wmma<ck::intrin_wmma_f16_16x16x128_bf8bf8<16, 16>, bf8x64_t, bf8x64_t, half_t>(p);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "wmma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_wmma for fp16_t, fp16_t, fp32_t MMA operation on GFX12
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12
// signature: V8fV8sV8sV8f
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12
// signature: V8hV8hV8hV8h
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12
// signature: V8sV8sV8sV8s
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
// signature: V8iIbV2iIbV2iV8iIb
// flags: sign, sign, clamp
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<uint8_t, uint8_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<uint8_t, uint8_t, int32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(0, aVec, 0, bVec, cVec, 0)};
    }
};

// __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
// signature: V8fV2iV2iV8f
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12
// signature: V8fV2iV2iV8f
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12
// signature: V8fV2iV2iV8f
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(aVec, bVec, cVec)};
    }
};

// __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12
// signature: V8fV2iV2iV8f
// flags: n/a
// layouts: A=K{8} L{K1M} V{K0} B=K{8} L{K1M} V{K0} C/D=M{8} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(aVec, bVec, cVec)};
    }
};

} // namespace ck_tile::core::arch::mma

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include <cstdint>

namespace {

/**
 * @struct RegisterIdxPair
 * @brief Small helper struct to hold a pair of lane and vector index values
 */
struct RegisterIdxPair
{
    const uint32_t lane;
    const uint32_t vecIdx;
};

/**
 * @class RegisterMap
 * @brief Maps matrix coordinates to (lane, vecIdx) pairs for a given MmaOp
 *
 * Test-only register mapping utilities for MMA layout validation. Each MMA intrinsic (MFMA, WMMA)
 * defines a specific mapping from logical matrix coordinates (m, k) or (m, n) to hardware
 * registers: which lane holds the value and at which vector index within that lane's register.
 * These mappings are architecture-specific and are encoded here as RegisterMap specializations, one
 * per target.
 *
 * Fails intentionally if left unimplemented.
 *
 * @tparam MmaOp amdgcn_mma specialization
 */
template <typename MmaOp>
struct RegisterMap
{
    static_assert(false, "RegisterMap always requires a specialization");

    CK_TILE_HOST_DEVICE static RegisterIdxPair A2RegisterMap(const uint32_t, const uint32_t)
    {
        return {0u, 0u};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair B2RegisterMap(const uint32_t, const uint32_t)
    {
        return {0u, 0u};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair C2RegisterMap(const uint32_t, const uint32_t)
    {
        return {0u, 0u};
    }
};

/**
 * @class RegisterMap (GFX12 WMMA specialization)
 * @brief Register mapping for __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
 *
 * TODO: At the moment, the mapping is hardcoded and derived based on unmerge/merge operations. In
 * the future, we want to use amdgcn_mma internals to compute it.
 *
 * A/B layout unmerge/merge: K{2, 4} L{K1M} V{K2K0}
 *
 * Unmerge K = 16 into K0 = K % 4, K1 = (K / 4) % 2, K2 = K / 8
 *
 *   lane   = 16 * k1 + m
 *   vecIdx = k0 + k2 * 2
 *
 * C layout descriptor:  M{8} L{M1N} V{M0}
 *
 * Unmerge M=16 into (M0 = M % 8) and (M1 = M / 8)
 *
 *   lane   = 16 * m1 + n
 *   vecIdx = m0
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMap<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx12_t<CompilerTarget>>>
{
    CK_TILE_HOST_DEVICE static RegisterIdxPair A2RegisterMap(const uint32_t m, const uint32_t k)
    {
        const uint32_t lane   = 16u * ((k / 4u) % 2u) + m;
        const uint32_t vecIdx = (k % 4) + (k / 8) * 4;
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair B2RegisterMap(const uint32_t k, const uint32_t n)
    {
        const uint32_t lane   = 16u * ((k / 4u) % 2u) + n;
        const uint32_t vecIdx = (k % 4) + (k / 8) * 4;
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair C2RegisterMap(const uint32_t m, const uint32_t n)
    {
        const uint32_t lane   = 16u * (m / 8u) + n;
        const uint32_t vecIdx = m % 8u;
        return {lane, vecIdx};
    }
};

/**
 * @class RegisterMap (GFX9 MFMA specialization)
 * @brief Register mapping for __builtin_amdgcn_mfma_f32_16x16x16f16
 *
 * TODO: At the moment, the mapping is hardcoded and derived based on unmerge/merge operations. In
 * the future, we want to use amdgcn_mma internals to compute it.
 *
 * A/B layout unmerge/merge: K{4} L{K1 M} V{K0}
 *
 * Unmerge K = 16 into (K0 = K % 4) and (K1 = K / 4)
 *
 *   lane   = 16 * k1 + m
 *   vecIdx = k0
 *
 * C layout unmerge/merge: M{4} L{M1 N} V{M0}
 *
 * Unmerge M = 16 into (M0 = M % 4) and (M1 = M / 4)
 *
 *   lane   = 16 * m1 + n
 *   vecIdx = m0
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMap<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx9_t<CompilerTarget>>>
{
    CK_TILE_HOST_DEVICE static RegisterIdxPair A2RegisterMap(const uint32_t m, const uint32_t k)
    {
        const uint32_t lane =
            16u * (k / 4u) + m; // NOTE: this can be derived with kAMLane * (k / kABKPerLane) + m
        const uint32_t vecIdx = k % 4u; // NOTE: this can be derived with k % kABKPerLane
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair B2RegisterMap(const uint32_t k, const uint32_t n)
    {
        const uint32_t lane =
            16u * (k / 4u) + n; // NOTE: this can be derived with kAMLane * (k / kABKPerLane) + n
        const uint32_t vecIdx = k % 4u; // NOTE: this can be derived with k % kABKPerLane
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair C2RegisterMap(const uint32_t m, const uint32_t n)
    {
        const uint32_t lane =
            16u * (m / 4u) + n; // NOTE: this can be derived with kCNLane * (m / kCM1PerLane) + n
        const uint32_t vecIdx = m % 4u; // NOTE: this can be derived with m % kCM1PerLane
        return {lane, vecIdx};
    }
};

/**
 * @class RegisterMap (GFX11 WMMA specialization)
 * @brief Register mapping for __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
 *
 * TODO: At the moment, the mapping is hardcoded and derived based on unmerge/merge operations. In
 * the future, we want to use amdgcn_mma internals to compute it.
 *
 * A/B layout unmerge/merge: L{M} V{K}
 *
 *   lane   = m
 *   vecIdx = k
 *
 * C layout unmerge/merge: M{2} L{M0N} V{M1}
 *
 * Unmerge M=16 into (M0 = M % 2) and (M1 = M / 2)
 *
 *   lane   = 16 * m0 + n
 *   vecIdx = m1
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMap<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx11_t<CompilerTarget>>>
{
    CK_TILE_HOST_DEVICE static RegisterIdxPair A2RegisterMap(const uint32_t m, const uint32_t k)
    {
        const uint32_t lane   = m;
        const uint32_t vecIdx = k;
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair B2RegisterMap(const uint32_t k, const uint32_t n)
    {
        const uint32_t lane   = n;
        const uint32_t vecIdx = k;
        return {lane, vecIdx};
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair C2RegisterMap(const uint32_t m, const uint32_t n)
    {
        const uint32_t lane   = (16u * (m % 2u)) + n;
        const uint32_t vecIdx = m / 2u;
        return {lane, vecIdx};
    }
};

} // namespace


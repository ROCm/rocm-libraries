// swmmac_traits.hpp — SWMMAC IO adapter traits
//
// Bridges rocWMMA's fragment IO system with SWMMAC's asymmetric register layout.
// Customizes per-thread element counts to match hardware: A=2×i32, B=4×i32.
//
// AMD vendor-level integration. Complements swmmac_impl.hpp + swmmac.hpp.
//
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "config.hpp"
#include "constants.hpp"
#include "io_shape.hpp"
#include "pack_util.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "vector_traits.hpp"

namespace rocwmma {

// ============================================================================
// SWMMAC register traits — hardware-level register counts per wave,
// overriding the standard IO formula (BlockDim × KDim / WaveSize / PackRatio).
// ============================================================================
template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct SwmmacRegTraits;

// SWMMAC INT4 A-side: 2×i32 per thread (16 INT4 values, nibble-packed)
template <uint32_t BlockM, uint32_t BlockK>
struct SwmmacRegTraits<matrix_a, BlockM, 16, BlockK>
{
    static constexpr uint32_t PackedSize   = 2;  // <2×i32> register
    static constexpr uint32_t PackedVRegs  = 2;
    static constexpr uint32_t UnpackedSize = 8;  // 2×4 values (PackRatio=4)
    using PackedT   = int32_t;
    using UnpackedT = int4_t;
};

// SWMMAC INT4 B-side: 4×i32 per thread (32 INT4 values, nibble-packed)
template <uint32_t BlockN, uint32_t BlockK>
struct SwmmacRegTraits<matrix_b, 16, BlockN, BlockK>
{
    static constexpr uint32_t PackedSize   = 4;  // <4×i32> register
    static constexpr uint32_t PackedVRegs  = 4;
    static constexpr uint32_t UnpackedSize = 16; // 4×4 values (PackRatio=4)
    using PackedT   = int32_t;
    using UnpackedT = int4_t;
};

// SWMMAC accum: 8×i32 per thread
template <uint32_t BlockM, uint32_t BlockN>
struct SwmmacRegTraits<accumulator, BlockM, BlockN, 64>
{
    static constexpr uint32_t PackedSize   = 8;  // <8×i32> register
    static constexpr uint32_t PackedVRegs  = 8;
    static constexpr uint32_t UnpackedSize = 8;
    using PackedT   = int32_t;
    using UnpackedT = int32_t;
};

// INT8 variants
template <uint32_t BlockM, uint32_t BlockK>
struct SwmmacRegTraits<matrix_a, BlockM, 16, BlockK>
{
    // Override for INT8 — same layout as INT4 on A-side
    static constexpr uint32_t PackedSize   = 2;
    static constexpr uint32_t PackedVRegs  = 2;
    static constexpr uint32_t UnpackedSize = 8;
    using PackedT   = int32_t;
    using UnpackedT = int8_t;
};

// ============================================================================
// SWMMAC Mma adapter — skips BlockK iteration, directly calls backend
//
// Standard Mma<> iterates BlocksK = FragK/BlockK times.
// SWMMAC completes K=64 in ONE instruction → skip iteration.
// ============================================================================
template <typename SwmmacBackend>
struct SwmmacMmaAdapter
{
    using Backend = SwmmacBackend;
    using ARegsT  = typename Backend::ARegsT;
    using BRegsT  = typename Backend::BRegsT;
    using CRegsT  = typename Backend::CRegsT;
    using DRegsT  = typename Backend::DRegsT;

    static constexpr uint32_t BlockM = Backend::BlockM;
    static constexpr uint32_t BlockN = Backend::BlockN;
    static constexpr uint32_t BlockK = Backend::BlockK;

    // Direct exec — no BlockK iteration needed
    // Fragment storage is passed as raw pointers to packed registers
    template <typename VecTA, typename VecTB, typename VecTC>
    __device__ static inline auto exec(
        VecTA&& a_storage, VecTB&& b_storage, VecTC& accum_storage)
    {
        // Extract registers from fragment packed storage
        ARegsT const& ra = reinterpret_cast<ARegsT const&>(a_storage);
        BRegsT const& rb = reinterpret_cast<BRegsT const&>(b_storage);
        CRegsT const& rc = reinterpret_cast<CRegsT const&>(accum_storage);
        DRegsT result = Backend::exec(ra, rb, rc);
        accum_storage = reinterpret_cast<VecTC&>(result);
        return accum_storage;
    }
};

// ============================================================================
// Convenience: SwmmacTraits — unified backend + register traits
// ============================================================================
template <typename InputTA,   typename InputTB,   typename ComputeT,
          uint32_t BlockM,    uint32_t BlockN,    uint32_t BlockK>
struct SwmmacTraits
{
    using Backend = detail::amdgcn_swmmac<InputTA, InputTB, ComputeT,
                                          BlockM, BlockN, BlockK>;
    using MmaDriver = SwmmacMmaAdapter<Backend>;

    using ARegTraits = SwmmacRegTraits<matrix_a, BlockM, BlockN, BlockK>;
    using BRegTraits = SwmmacRegTraits<matrix_b, BlockM, BlockN, BlockK>;
    using CRegTraits = SwmmacRegTraits<accumulator, BlockM, BlockN, BlockK>;

    static constexpr bool is_swmmac = true;
};

} // namespace rocwmma

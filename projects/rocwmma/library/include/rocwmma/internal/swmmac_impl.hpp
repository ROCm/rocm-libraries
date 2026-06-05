// swmmac_impl.hpp — SWMMAC backend implementation (amdgcn_swmmac)
// Follows amdgcn_mfma pattern: wave-level matrix multiply-accumulate.
// AMD vendor-level integration into rocWMMA core.
//
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "constants.hpp"
#include "types.hpp"
#include "utility/type_traits.hpp"
#include "utility/vector.hpp"

namespace rocwmma {
namespace detail {

struct Unsupported;

// Target enablers
template <uint32_t TargetId, uint32_t... TargetIds>
using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;

template <uint32_t TargetId, bool Cond = true>
using enable_gfx12_swmmac_t
    = enable_if_t<contains_number_v<uint32_t,
                                    TargetId,
                                    Constants::AMDGCN_ARCH_ID_GFX1200,
                                    Constants::AMDGCN_ARCH_ID_GFX1201> && Cond>;

// ============================================================================
// amdgcn_swmmac — wave-level SWMMAC backend
//
// Template params: InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK.
// Each specialization maps to one v_swmmac instruction variant.
// ============================================================================
template <typename InputTA,   typename InputTB,   typename ComputeT,
          uint32_t BlockM,    uint32_t BlockN,    uint32_t BlockK,
          uint32_t GfxTargetId = Constants::AMDGCN_CURRENT_ARCH_ID,
          typename Enabler      = void>
struct amdgcn_swmmac {
    using Unsupported = Unsupported;

private:
    using PackTraitsA   = PackTraits<InputTA>;
    using PackTraitsB   = PackTraits<InputTB>;
    using PackTraitsAcc = PackTraits<ComputeT>;

    static constexpr uint32_t InputASize
        = BlockM * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsA::PackRatio);
    static constexpr uint32_t InputBSize
        = BlockN * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsB::PackRatio);
    static constexpr uint32_t AccumSize
        = BlockM * BlockN / (Constants::AMDGCN_WAVE_SIZE * PackTraitsAcc::PackRatio);

public:
    using ARegsT = VecT<typename PackTraitsA::PackedT, InputASize>;
    using BRegsT = VecT<typename PackTraitsB::PackedT, InputBSize>;
    using CRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
    using DRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
};

// ============================================================================
// INT4 SWMMAC: 16×16×64 (gfx1200 wave32)
// ============================================================================
template <uint32_t GfxTargetId>
struct amdgcn_swmmac<int4_t, int4_t, int32_t, 16u, 16u, 64u, GfxTargetId,
                     enable_gfx12_swmmac_t<GfxTargetId>>
{
    using ARegsT = VecT<int32_t, 2>;
    using BRegsT = VecT<int32_t, 4>;
    using CRegsT = VecT<int32_t, 8>;
    using DRegsT = VecT<int32_t, 8>;

    // IQ format helpers: convert fragment format (PackRatio=4, 4 values/i32)
    // to hardware format (nibble-packed, 8 values/i32)
    __device__ static inline ARegsT to_hw_a(ARegsT const& frag_a) {
        // fragment: [a3,a2,a1,a0,0,0,0,0], [a7,a6,a5,a4,0,0,0,0]
        //          (2 unused high-nibble pairs per i32 due to PackRatio=4)
        // hardware: [a7,a6,a5,a4,a3,a2,a1,a0] (nibble-packed)
        ARegsT r;
        r[0] = (frag_a[0] & 0x0F0F0F0Fu) | ((frag_a[1] & 0x0F0F0F0Fu) << 4);
        r[1] = 0;  // pad — 16 INT4 values for A, only 2 i32 needed
        return r;
    }
    __device__ static inline BRegsT to_hw_b(BRegsT const& frag_b) {
        // fragment: 4 values/i32 × 4 elements = 16 values (need 32 for HW)
        // hardware needs 8 values/i32 × 4 = 32 values
        BRegsT r;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            r[i] = (frag_b[i*2] & 0x0F0F0F0Fu) | ((frag_b[i*2+1] & 0x0F0F0F0Fu) << 4);
        return r;
    }

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b, CRegsT const& c)
    {
        DRegsT result;
        ARegsT hw_a = to_hw_a(a);
        BRegsT hw_b = to_hw_b(b);
        to_native_vector(result)
            = {__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(
                true, to_native_vector(hw_a),
                true, to_native_vector(hw_b),
                to_native_vector(c), 0, true)};
        return result;
    }
};

// ============================================================================
// INT8 SWMMAC: 16×16×32 (gfx1200 wave32)
// ============================================================================
template <uint32_t GfxTargetId>
struct amdgcn_swmmac<int8_t, int8_t, int32_t, 16u, 16u, 32u, GfxTargetId,
                     enable_gfx12_swmmac_t<GfxTargetId>>
{
    using ARegsT = VecT<int32_t, 2>;
    using BRegsT = VecT<int32_t, 4>;
    using CRegsT = VecT<int32_t, 8>;
    using DRegsT = VecT<int32_t, 8>;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b, CRegsT const& c)
    {
        DRegsT result;
        // Pad A from 2 to 4 elements if needed (following MFMA concat pattern)
        to_native_vector(result)
            = {__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(
                true, to_native_vector(a),
                true, to_native_vector(b),
                to_native_vector(c), 0, true)};
        return result;
    }
};

} // namespace detail
} // namespace rocwmma

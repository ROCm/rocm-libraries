// rocwmma_gfx11_fallback.hpp — WMMA fallback for RDNA3 (gfx11)
//
// When SWMMAC is not available (gfx1100/gfx1101/gfx1102/gfx1150/gfx1151),
// falls back to WMMA instructions with equivalent functionality.
//
// WMMA gfx11 register layout (differs from SWMMAC gfx12):
//   INT8: A=<4×i32>, B=<4×i32>, C/D=<8×i32>, K=16
//   INT4: A=<2×i32>, B=<2×i32>, C/D=<8×i32>, K=16
//   FP16: A=<16×f16>, B=<16×f16>, C/D=<8×f32>, K=16
//
// Compared to SWMMAC:
//   SWMMAC INT8: A=<2×i32>, B=<4×i32>, K=32  (1/2 K, different A width)
//   SWMMAC INT4: A=<2×i32>, B=<4×i32>, K=64  (1/4 K)
//
// Throughput on gfx11: ~50% of gfx12 SWMMAC (due to K=16 vs K=32/64)
//
// Usage: #include <rocwmma/rocwmma_gfx11_fallback.hpp>
//        Uses chain_pipeline_dispatch<> to auto-select SWMMAC or WMMA.
#pragma once

#include "rocwmma_int4.hpp"
#include "internal/utility/vector.hpp"
#include "internal/vector_traits.hpp"

// Only include SWMMAC backends if available (gfx12)
#if ROCWMMA_HAS_SWMMAC
#include "rocwmma_swmmac.hpp"
#include "rocwmma_16chain.hpp"
#endif

// Architecture detection
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) \
    || defined(__gfx1150__) || defined(__gfx1151__)
#define ROCWMMA_IS_GFX11 1
#else
#define ROCWMMA_IS_GFX11 0
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)
#define ROCWMMA_IS_GFX12 1
#else
#define ROCWMMA_IS_GFX12 0
#endif

namespace rocwmma {

// ============================================================================
// WMMA INT8 backend (gfx11: 16×16×16)
// ============================================================================
template <bool ASign = true, bool BSign = true, bool CSign = true>
struct WmmaInt8Gfx11 {
    using ARegsT = VecT<int32_t, 4>;    // <4×i32> (wider than SWMMAC <2×i32>)
    using BRegsT = VecT<int32_t, 4>;    // <4×i32>
    using CRegsT = VecT<int32_t, 8>;    // <8×i32> (same)
    using DRegsT = VecT<int32_t, 8>;

    static constexpr uint32_t BlockM = 16;
    static constexpr uint32_t BlockN = 16;
    static constexpr uint32_t BlockK = 16;  // half of SWMMAC INT8 K=32

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c)
    {
        DRegsT result;
        to_native_vector(result)
            = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                ASign, to_native_vector(a),
                BSign, to_native_vector(b),
                to_native_vector(c), CSign)};
        return result;
    }
};

using WmmaI8Gfx11 = WmmaInt8Gfx11<true, true, true>;

// ============================================================================
// WMMA INT4 backend (gfx11: 16×16×16)
// ============================================================================
template <bool ASign = true, bool BSign = true, bool CSign = true>
struct WmmaInt4Gfx11 {
    using ARegsT = VecT<int32_t, 2>;    // <2×i32> (same width)
    using BRegsT = VecT<int32_t, 2>;    // <2×i32> (narrower than SWMMAC <4×i32>)
    using CRegsT = VecT<int32_t, 8>;    // <8×i32> (same)
    using DRegsT = VecT<int32_t, 8>;

    static constexpr uint32_t BlockM = 16;
    static constexpr uint32_t BlockN = 16;
    static constexpr uint32_t BlockK = 16;  // 1/4 of SWMMAC INT4 K=64

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c)
    {
        DRegsT result;
        to_native_vector(result)
            = {__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                ASign, to_native_vector(a),
                BSign, to_native_vector(b),
                to_native_vector(c), CSign)};
        return result;
    }
};

using WmmaI4Gfx11 = WmmaInt4Gfx11<true, true, true>;

// ============================================================================
// gfx11 WMMA ChainPipeline
//
// WMMA doesn't have sparse_idx, so the step() signature differs from SWMMAC.
// This adapter provides a step(A,B) with the same interface.
// ============================================================================
template <uint32_t NCHAINS = 16, typename Backend = WmmaI4Gfx11>
struct ChainPipelineGfx11 {
    static_assert(NCHAINS <= 16, "WMMA gfx11 supports up to 16 chains");

    // Types
    using ARegsT = typename Backend::ARegsT;
    using BRegsT = typename Backend::BRegsT;
    using CRegsT = typename Backend::CRegsT;
    using AccumT = typename VecTraits<CRegsT>::DataT;
    using APtrT  = typename VecTraits<ARegsT>::DataT;
    using BPtrT  = typename VecTraits<BRegsT>::DataT;

    static constexpr uint32_t BLOCK_K  = Backend::BlockK;
    static constexpr uint32_t BLOCK_M  = Backend::BlockM;
    static constexpr double   OPS      = BLOCK_M * BLOCK_M * BLOCK_K * 2.0;

    alignas(32) AccumT acc[NCHAINS][8];

    __device__ void zero() {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] = AccumT{0};
    }

    __device__ void store(AccumT* C) const {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) C[c * 8 + i] = acc[c][i];
    }

    // WMMA step (no sparse_idx — gfx11 WMMA doesn't support sparsity)
    __device__ __attribute__((always_inline))
    void step(APtrT const* __restrict__ A, BPtrT const* __restrict__ B) {
        ARegsT const& a = *reinterpret_cast<ARegsT const*>(A);
        BRegsT const& b = *reinterpret_cast<BRegsT const*>(B);
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c) {
            CRegsT& accum = *reinterpret_cast<CRegsT*>(acc[c]);
            accum = Backend::exec(a, b, accum);
        }
    }
};

// ============================================================================
// Architecture dispatch alias
//   AutoChain<16> → SWMMAC ChainPipeline on gfx12, WMMA ChainPipeline on gfx11
// ============================================================================
#if ROCWMMA_HAS_SWMMAC
template <uint32_t NCHAINS = 16>
using AutoChain = ChainPipeline<NCHAINS, SwmmacI4>;
#else
template <uint32_t NCHAINS = 16>
using AutoChain = ChainPipelineGfx11<NCHAINS, WmmaI4Gfx11>;
#endif

} // namespace rocwmma

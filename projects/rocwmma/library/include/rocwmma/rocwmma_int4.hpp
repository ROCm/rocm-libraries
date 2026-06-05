// rocwmma_int4.hpp — INT4/INT8 type extension for rocWMMA
// Layer 1 of 3: Adds 4-bit integer type + SWMMAC register types.
//
// INT4 packing: 8 signed 4-bit values per 32-bit register.
// INT8 packing: 4 signed 8-bit values per 32-bit register.
// Both share the same SWMMAC register layout on gfx1200:
//   A: <2 x i32>  B: <4 x i32>  C/D: <8 x i32>
//
// Usage:
//   #include <rocwmma/rocwmma_int4.hpp>
//   rocwmma::int4_t x = rocwmma::int4_t(3);
//   rocwmma::SwmmacAccumT acc{};
//   rocwmma::SwmmacARegsT a = { pack0, pack1 };
#pragma once

#include "rocwmma.hpp"
#include "internal/vector.hpp"
#include "internal/vector_types.hpp"

// ============================================================================
// Architecture feature detection
// SWMMAC instructions (v_swmmac_*) are gfx1200/gfx1201 only.
// On gfx11 (RDNA3), WMMA is available but SWMMAC is not.
// On gfx9/CDNA, MFMA is the only matrix instruction.
//
// HIP two-pass compilation:
//   Device pass (__HIP_DEVICE_COMPILE__): arch macros defined → check gfx12
//   Host pass: always enable types for template visibility
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)
  // Device code generation — architecture macros are authoritative
  #if defined(__gfx1200__) || defined(__gfx1201__)
    #define ROCWMMA_HAS_SWMMAC 1
  #else
    #define ROCWMMA_HAS_SWMMAC 0
  #endif
#else
  // Host or non-HIP compilation — enable types, device check catches errors
  #define ROCWMMA_HAS_SWMMAC 1
#endif

#if defined(__HIP_DEVICE_COMPILE__) && !ROCWMMA_HAS_SWMMAC
#pragma message "rocwmma_int4.hpp: SWMMAC not available on this target. "\
                    "Use rocWMMA WMMA backend instead. "\
                    "SWMMAC requires gfx1200/gfx1201 (RDNA4)."
#endif

namespace rocwmma {

// ============================================================================
// INT4 value type — 4-bit signed integer (-8..7), sign-extended in int8_t
// ============================================================================

namespace detail {
// Sign-extend a 4-bit value from bit 3 to full int8_t
__host__ __device__ constexpr int8_t sext4(int8_t v) {
    int8_t lo = static_cast<int8_t>(v & 0xF);
    return (lo & 0x8) ? static_cast<int8_t>(lo | static_cast<int8_t>(0xF0)) : lo;
}
} // namespace detail

struct int4_t {
    int8_t data;  // always sign-extended: -8..7

    __host__ __device__ constexpr int4_t() : data(0) {}
    __host__ __device__ constexpr explicit int4_t(int8_t v)
        : data(detail::sext4(v)) {}
    __host__ __device__ constexpr explicit int4_t(int v)
        : data(detail::sext4(static_cast<int8_t>(v))) {}

    __host__ __device__ constexpr operator int8_t()  const { return data; }
    __host__ __device__ constexpr operator int32_t() const { return static_cast<int32_t>(data); }

    __host__ __device__ constexpr int4_t operator-() const {
        return int4_t(static_cast<int8_t>(-data));
    }
    __host__ __device__ constexpr int4_t operator+(int4_t rhs) const {
        return int4_t(static_cast<int8_t>(data + rhs.data));
    }
    __host__ __device__ constexpr int4_t operator-(int4_t rhs) const {
        return int4_t(static_cast<int8_t>(data - rhs.data));
    }
    __host__ __device__ constexpr int4_t operator*(int4_t rhs) const {
        return int4_t(static_cast<int8_t>(data * rhs.data));
    }

    __host__ __device__ constexpr bool operator==(int4_t rhs) const {
        return (data & 0xF) == (rhs.data & 0xF);
    }
    __host__ __device__ constexpr bool operator!=(int4_t rhs) const {
        return (data & 0xF) != (rhs.data & 0xF);
    }
};

// ============================================================================
// INT4 ↔ i32 packing — hardware nibble order for SWMMAC A/B operands
//
// Each i32 packs 8 INT4 values:
//   bits [3:0]   = value 0   bits [19:16] = value 4
//   bits [7:4]   = value 1   bits [23:20] = value 5
//   bits [11:8]  = value 2   bits [27:24] = value 6
//   bits [15:12] = value 3   bits [31:28] = value 7
//
// Usage:
//   uint32_t a0 = rocwmma::pack_int4x8(v0, v1, v2, v3, v4, v5, v6, v7);
//   SwmmacARegsT A = {{(int32_t)a0, (int32_t)a1}};
// ============================================================================

__host__ __device__ inline uint32_t pack_int4x8(
    int4_t v0, int4_t v1, int4_t v2, int4_t v3,
    int4_t v4, int4_t v5, int4_t v6, int4_t v7)
{
    return (static_cast<uint32_t>(v0.data) & 0xFu)
        | ((static_cast<uint32_t>(v1.data) & 0xFu) << 4)
        | ((static_cast<uint32_t>(v2.data) & 0xFu) << 8)
        | ((static_cast<uint32_t>(v3.data) & 0xFu) << 12)
        | ((static_cast<uint32_t>(v4.data) & 0xFu) << 16)
        | ((static_cast<uint32_t>(v5.data) & 0xFu) << 20)
        | ((static_cast<uint32_t>(v6.data) & 0xFu) << 24)
        | ((static_cast<uint32_t>(v7.data) & 0xFu) << 28);
}

// Unpack one INT4 value from nibble position n (0..7)
__host__ __device__ inline int4_t unpack_int4_nibble(uint32_t packed, int n) {
    return int4_t(static_cast<int8_t>((packed >> (n * 4)) & 0xFu));
}

// Fill an entire i32 with the same INT4 value repeated 8 times
__host__ __device__ inline int32_t broadcast_int4(int4_t v) {
    uint32_t nib = static_cast<uint32_t>(v.data) & 0xFu;
    uint32_t pat = nib | (nib << 4) | (nib << 8) | (nib << 12)
                 | (nib << 16) | (nib << 20) | (nib << 24) | (nib << 28);
    return static_cast<int32_t>(pat);
}

// ============================================================================
// Sparse mode selector — 2:4 structured sparsity
//
// In 2:4 sparsity, every group of 4 consecutive values has exactly 2 non-zero
// values. The sparse_idx encodes which pair is selected in each group.
//   idx=0: dense mode (all values used, no sparsity)
//   idx≠0: sparse mode (hardware skips zero-value pairs)
//
// Expected throughput: 2× dense for sparse workloads (half the MACs skipped).
// ============================================================================
enum class SparseSel : int32_t {
    DENSE  =  0,   // all values used
    SPARSE = -1,   // auto / non-zero = 2:4 structured sparsity
};

// ============================================================================
// SWMMAC hardware register types (gfx1200 wave32)
// Shared by both INT4 and INT8 SWMMAC instructions.
//
// INT4: A=<2×i32> = 16 values (8/i32), B=<4×i32> = 32 values (8/i32)
// INT8: A=<2×i32> =  8 values (4/i32), B=<4×i32> = 16 values (4/i32)
// C/D: <8 x i32> accumulator matrix output
// ============================================================================
using SwmmacARegsT = VecT<int32_t, 2>;    // <2 x i32>
using SwmmacBRegsT = VecT<int32_t, 4>;    // <4 x i32>
using SwmmacAccumT = VecT<int32_t, 8>;    // <8 x i32>
using SwmmacIdxT   = int32_t;              // sparse index

// ============================================================================
// INT4 SWMMAC constants
// ============================================================================
struct SwmmacConstants {
    static constexpr uint32_t BlockM     = 16;
    static constexpr uint32_t BlockN     = 16;
    static constexpr uint32_t BlockK     = 64;
    static constexpr uint32_t MACs       = 16384;  // 16×16×64
    static constexpr uint32_t Ops        = 32768;  // ×2 ops
    static constexpr uint32_t XDLDepth   = 16;     // pipeline depth
};

// ============================================================================
// INT8 SWMMAC constants (same register layout, K=32)
// ============================================================================
struct SwmmacConstantsInt8 {
    static constexpr uint32_t BlockM     = 16;
    static constexpr uint32_t BlockN     = 16;
    static constexpr uint32_t BlockK     = 32;
    static constexpr uint32_t MACs       = 8192;   // 16×16×32
    static constexpr uint32_t Ops        = 16384;  // ×2 ops
    static constexpr uint32_t XDLDepth   = 16;     // pipeline depth
};

// ============================================================================
// FP SWMMAC constants — all are 16×16×K with f32 accumulate
// ============================================================================
struct SwmmacConstantsFp {
    static constexpr uint32_t BlockM     = 16;
    static constexpr uint32_t BlockN     = 16;
    static constexpr uint32_t BlockK     = 32;
    static constexpr uint32_t MACs       = 8192;   // 16×16×32
    static constexpr uint32_t Ops        = 16384;  // ×2 ops
    static constexpr uint32_t XDLDepth   = 16;
};

// ============================================================================
// FP accumulator type — shared by FP8/BF8/FP16/BF16 SWMMAC
// ============================================================================
using SwmmacFpAccumT = VecT<float32_t, 8>;       // <8 x f32>

// ============================================================================
// FP16/BF16 SWMMAC register types (different from integer A/B layout!)
//
// FP16 SWMMAC: A=<8×f16>, B=<16×f16>
// BF16 SWMMAC: A=<8×i16>, B=<16×i16> (bf16 stored as i16 for builtin compat)
// ============================================================================
using SwmmacF16ARegsT  = VecT<float16_t, 8>;     // <8 x f16>
using SwmmacF16BRegsT  = VecT<float16_t, 16>;    // <16 x f16>
using SwmmacBf16ARegsT = VecT<int16_t, 8>;       // <8 x i16> (bf16 bitcast)
using SwmmacBf16BRegsT = VecT<int16_t, 16>;      // <16 x i16>

} // namespace rocwmma

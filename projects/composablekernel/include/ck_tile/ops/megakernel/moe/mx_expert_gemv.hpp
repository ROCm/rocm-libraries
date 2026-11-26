// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @file
/// Weight-streaming MX GEMV for MoE experts at decode M (a few tokens per expert), gfx1250.
///
/// One wave computes C[16 weight rows, 16 tokens] += W[rows, K] * X[tokens, K]^T with
/// v_wmma_scale_f32_16x16x128_f8f6f4: W is MXFP4 (e2m1, 2 per byte, row-major [N, K/2]) and is
/// the WMMA A operand; X is MX fp8 (row-major [rows, K]) and is the B operand. Both carry one e8m0
/// scale per 32 along K ([rows, K/32]). The 16 WMMA columns are tokens, so with 1-2 tokens per
/// expert most of B is padding, but the streamed operand (W) is dense and each lane's C holds 8
/// consecutive output features of one token (one 8 to 16 byte store in the epilogue).
///
/// Register layouts follow ck_tile's gfx125 WmmaTraits (LayoutFromNon8BitMixPrec for e2m1,
/// LayoutFrom8BitMixPrec for fp8; lane = half * 16 + row):
///   A (e2m1): lane (row, h) dword d, nibble i (low nibble first): K = 64 (d / 4) + 32 h + 8 (d %
///   4) + i B (fp8):  lane (col, h) byte i:                                K = 32 (i / 16) + 16 h +
///   i % 16 scale:    byte b of a lane's int32 scales K block [32 b, 32 b + 32) (same in both
///   halves) C:        lane (col, h) element v: row 8 h + v, column col
///
/// The K order inside a WMMA only has to agree between A, B and the scales, so a lane loads a
/// contiguous 64-byte run of its weight row: a "window" is 256 K (two WMMAs) split into eight
/// 32-K blocks, and logical block b of WMMA j is block window_block(j, b) of the window. Lane half
/// h then holds window blocks 4h..4h+3 of A (bytes [64 h, 64 h + 64) of the 128-byte row window);
/// B and the scales follow the same block permutation (b_chunk_byte, select_scales).
///
/// The A/B/scale layouts above were verified on MI450 (single-product probes are exact). The
/// instruction does not add its 128 scaled products exactly: with a wide dynamic range among
/// them the result comes out low (truncation) by up to ~2^-9.5 of the largest product, which the
/// example's reference tolerance models (WmmaErr in example 62).
/// TODO(megakernel): bf16 activations (dequantise W to bf16, v_wmma_f32_16x16x32_bf16) and A4W4
/// (e2m1 B, same instruction).
namespace moe_mx {

inline constexpr index_t kWmmaK       = 128;
inline constexpr index_t kWindowK     = 256;
inline constexpr index_t kScaleBlock  = 32;
inline constexpr index_t kWindowBlock = kWindowK / kScaleBlock; // 8
inline constexpr index_t kWaveRows    = 16;                     // weight rows (features) per wave
inline constexpr index_t kWaveTokens  = 16;                     // token columns per WMMA

/// Window block (0..7) that holds logical K block @p b (0..3) of WMMA @p j (0..1) of a window.
CK_TILE_HOST_DEVICE constexpr index_t window_block(index_t j, index_t b)
{
    return 4 * (b % 2) + 2 * j + b / 2;
}

/// The lane's int32 scale operand of WMMA @p j from the window's 8 scale bytes (lo = blocks 0-3,
/// hi = blocks 4-7): bytes (lo[2j], hi[2j], lo[2j+1], hi[2j+1]).
CK_TILE_HOST_DEVICE uint32_t select_scales(uint32_t lo, uint32_t hi, index_t j)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(hi, lo, j == 0 ? 0x05010400u : 0x07030602u);
#else
    uint32_t r = 0;
    for(index_t b = 0; b < 4; ++b)
    {
        const index_t blk = window_block(j, b);
        const uint32_t v  = ((blk < 4 ? lo : hi) >> (8 * (blk % 4))) & 0xFFu;
        r |= v << (8 * b);
    }
    return r;
#endif
}

/// Byte offset in the 128-byte row window (256 K of e2m1) of the 16 bytes that fill dwords
/// [4 c, 4 c + 4) of A operand @p j (8 dwords) in lane half @p half. A lane's two operands are
/// its contiguous 64 bytes [64 half, 64 half + 64).
CK_TILE_HOST_DEVICE constexpr index_t a_chunk_byte(index_t j, index_t c, index_t half)
{
    return 64 * half + 32 * j + 16 * c;
}

/// Byte offset in the 256-byte fp8 window (256 K) of the 16 bytes that fill dwords
/// [4 s, 4 s + 4) of B operand @p j (16 dwords) in lane half @p half.
CK_TILE_HOST_DEVICE constexpr index_t b_chunk_byte(index_t j, index_t s, index_t half)
{
    return 32 * window_block(j, s) + 16 * half;
}

template <typename T>
struct f8f6f4_format;
template <>
struct f8f6f4_format<fp8_t>
{
    static constexpr int value = 0; // E4M3
};
template <>
struct f8f6f4_format<bf8_t>
{
    static constexpr int value = 1; // E5M2
};
template <>
struct f8f6f4_format<pk_fp4_t>
{
    static constexpr int value = 4; // E2M1
};

/// e8m0 scale for an MX block whose largest magnitude is @p amax, as cast_tile_mx_wmma rounds it:
/// the smallest power of two >= amax / max(T), at least 2^-126. Returns the float's bits.
template <typename T>
CK_TILE_HOST_DEVICE uint32_t mx_scale_bits(float amax)
{
    constexpr float rcp_max = 1.0f / (std::is_same_v<T, fp8_t> ? 448.0f : 57344.0f);
    const uint32_t b        = (bit_cast<uint32_t>(amax * rcp_max) + 0x007FFFFFu) & 0x7F800000u;
    return b < 0x00800000u ? 0x00800000u : b;
}

/// Round to bf16, nearest even (ck_tile's default bf16 conversion truncates). Returns the bits.
CK_TILE_HOST_DEVICE uint16_t bf16_rne_bits(float f)
{
    const uint32_t u = bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

CK_TILE_HOST_DEVICE float bf16_round(float f)
{
    return bit_cast<float>(static_cast<uint32_t>(bf16_rne_bits(f)) << 16);
}

/// One wave's share of an expert GEMV: kNumMats weight matrices (gate and up share the tokens)
/// times one group of up to 16 tokens over K range [k_begin, k_end), multiples of kWindowK.
///
/// @tparam ActType fp8_t or bf8_t (MX, e8m0 per 32).
template <typename ActType, index_t kNumMats>
struct mx_gemv_wave
{
    static_assert(std::is_same_v<ActType, fp8_t> || std::is_same_v<ActType, bf8_t>,
                  "TODO: bf16 activations (dequantised W) and A4W4 (e2m1 B)");

    /// One window's operands, loaded straight into WMMA operand order.
    struct Window
    {
        int32x8_t w[kNumMats][2]; // A of WMMA j
        uint32x2_t ws[kNumMats];  // weight scales of window blocks 0-3, 4-7
        int32x16_t x[2];          // B of WMMA j
        uint32x2_t xs;
    };

    CK_TILE_DEVICE static void ClearTokens(Window& win)
    {
        win.x[0] = int32x16_t{0};
        win.x[1] = int32x16_t{0};
        win.xs   = uint32x2_t{0, 0};
    }

    /// @param w_row   per matrix, this lane's weight row (K/2 bytes)
    /// @param ws_row  per matrix, this lane's weight scale row (K/32 bytes)
    /// @param x_row   this lane's token row (K bytes); only read when @p x_valid
    /// @param xs_row  this lane's token scale row (K/32 bytes)
    CK_TILE_DEVICE static void Load(Window& win,
                                    const uint8_t* const (&w_row)[kNumMats],
                                    const uint8_t* const (&ws_row)[kNumMats],
                                    const uint8_t* x_row,
                                    const uint8_t* xs_row,
                                    bool x_valid,
                                    index_t half,
                                    index_t k)
    {
        static_for<0, kNumMats, 1>{}([&](auto m) {
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 2, 1>{}([&](auto c) {
                    // Weights are read once per launch: stream them past the caches.
                    const int32x4_t v =
                        __builtin_nontemporal_load(reinterpret_cast<const int32x4_t*>(
                            w_row[m] + k / 2 + a_chunk_byte(j, c, half)));
                    static_for<0, 4, 1>{}(
                        [&](auto d) { win.w[m][j][4 * c.value + d.value] = v[d.value]; });
                });
            });
            win.ws[m] = *reinterpret_cast<const uint32x2_t*>(ws_row[m] + k / kScaleBlock);
        });
        // Padding token lanes load nothing; their C columns are discarded.
        if(x_valid)
        {
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 4, 1>{}([&](auto s) {
                    const int32x4_t v =
                        *reinterpret_cast<const int32x4_t*>(x_row + k + b_chunk_byte(j, s, half));
                    static_for<0, 4, 1>{}(
                        [&](auto d) { win.x[j][4 * s.value + d.value] = v[d.value]; });
                });
            });
            win.xs = *reinterpret_cast<const uint32x2_t*>(xs_row + k / kScaleBlock);
        }
    }

    CK_TILE_DEVICE static void Compute(const Window& win, fp32x8_t (&acc)[kNumMats])
    {
#if defined(__gfx125__)
        static_for<0, 2, 1>{}([&](auto j) {
            const uint32_t b_sc = select_scales(win.xs[0], win.xs[1], j);
            static_for<0, kNumMats, 1>{}([&](auto m) {
                const uint32_t a_sc = select_scales(win.ws[m][0], win.ws[m][1], j);
                // e2m1 uses the low 8 dwords of the 16-dword operand.
                const int32x16_t a = __builtin_shufflevector(win.w[m][j],
                                                             int32x8_t{0},
                                                             0,
                                                             1,
                                                             2,
                                                             3,
                                                             4,
                                                             5,
                                                             6,
                                                             7,
                                                             8,
                                                             9,
                                                             10,
                                                             11,
                                                             12,
                                                             13,
                                                             14,
                                                             15);
                acc[m] =
                    __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(f8f6f4_format<pk_fp4_t>::value,
                                                                     a,
                                                                     f8f6f4_format<ActType>::value,
                                                                     win.x[j],
                                                                     0, // C modifier
                                                                     acc[m],
                                                                     0, // scale A lane select
                                                                     0, // scale A format: e8m0
                                                                     static_cast<int32_t>(a_sc),
                                                                     0, // scale B lane select
                                                                     0, // scale B format: e8m0
                                                                     static_cast<int32_t>(b_sc),
                                                                     false,  // reuse A
                                                                     false); // reuse B
            });
        });
#else
        ignore = win;
        ignore = acc;
#endif
    }

    /// acc[m] += W_m[rows] * X[tokens]^T over K [k_begin, k_begin + kKPerWave): fully unrolled,
    /// two windows ping-ponged so one window of loads is in flight during the WMMAs.
    template <index_t kKPerWave>
    CK_TILE_DEVICE static void Run(const uint8_t* const (&w_row)[kNumMats],
                                   const uint8_t* const (&ws_row)[kNumMats],
                                   const uint8_t* x_row,
                                   const uint8_t* xs_row,
                                   bool x_valid,
                                   index_t k_begin,
                                   fp32x8_t (&acc)[kNumMats])
    {
        static_assert(kKPerWave % kWindowK == 0);
        constexpr index_t kWindows = kKPerWave / kWindowK;
        const index_t half         = static_cast<index_t>(__lane_id()) / kWaveTokens;
        Window win[2];
        ClearTokens(win[0]);
        ClearTokens(win[1]);
        Load(win[0], w_row, ws_row, x_row, xs_row, x_valid, half, k_begin);
        static_for<0, kWindows, 1>{}([&](auto i) {
            if constexpr(i.value + 1 < kWindows)
                Load(win[(i.value + 1) % 2],
                     w_row,
                     ws_row,
                     x_row,
                     xs_row,
                     x_valid,
                     half,
                     k_begin + (i.value + 1) * kWindowK);
            Compute(win[i.value % 2], acc);
        });
    }
};

} // namespace moe_mx
} // namespace ck_tile

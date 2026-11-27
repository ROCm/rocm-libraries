// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_load.hpp"

namespace ck_tile {

// Same intrinsic as __builtin_amdgcn_s_buffer_load_f32, which older clang (e.g. ROCm 10.0) lacks.
CK_TILE_DEVICE_EXTERN __attribute__((noconvergent)) float llvm_amdgcn_s_buffer_load_fp32(
    int32x4_t srsrc, index_t soffset, index_t glc_slc) __asm("llvm.amdgcn.s.buffer.load.f32");

// Uses the clang builtin when present, else the intrinsic declaration (e.g. ROCm 10.0).
CK_TILE_DEVICE float amd_s_buffer_load_fp32(int32x4_t rsrc, index_t offset)
{
#if __has_builtin(__builtin_amdgcn_s_buffer_load_f32)
    return __builtin_amdgcn_s_buffer_load_f32(cast_to_amdgpu_buffer_rsrc_t(rsrc), offset, 0);
#else
    return llvm_amdgcn_s_buffer_load_fp32(rsrc, offset, 0);
#endif
}

// MX: one E8M0 per 32 elements along each GEMM's K dimension; strides are in scales.
struct FmhaTdmSchedFp8MxDescales
{
    const e8m0_t* q_ptr = nullptr;
    const e8m0_t* k_ptr = nullptr;
    const e8m0_t* v_ptr = nullptr;
    index_t stride_q    = 0;
    index_t stride_k    = 0;
    index_t stride_v    = 0;
    index_t seqlen_q    = 1;
};

// Blockscale and MX descale handling and the P quantization helpers of the FP8 sched pipeline.
template <typename Geometry, index_t kBlockSize, index_t kK1, index_t kQKHeaddim>
struct FmhaTdmSchedFp8QuantScale
{
    static constexpr index_t kN0 = Geometry::kN;

    // A scale operand packs kScaleBytes E8M0 bytes; a gemm_1 byte covers kPScaleGranularity K.
    static constexpr index_t kScaleBytes        = 4;
    static constexpr index_t kPScaleGranularity = kK1 / kScaleBytes;
    // K descale is one scalar per QK stage, V descale one E8M0 byte per kPScaleGranularity rows.
    static constexpr index_t kKVScaleAlign = Geometry::kQkSuColumns;
    // K and V descales of one KV tile, one per kKVScaleAlign-wide group; both share the index.
    static constexpr index_t kKVScaleGroups = kN0 / kKVScaleAlign;
    using KVDescales                        = array<float, kKVScaleGroups>;

    // Neutral scale operand, four E8M0 1.0 bytes; a zeroed one decodes to 2^-127.
    static constexpr int32_t kE8m0One = 0x7f7f7f7f;

    // Stochastic-rounding seed: a mixed query-row hash plus a Weyl step per KV tile.
    static constexpr uint32_t kPSeedStep = 0x9e3779b9u;
    CK_TILE_DEVICE static uint32_t PSeed(index_t row, index_t kv_tile)
    {
        auto x = static_cast<uint32_t>(row);
        x      = (x ^ (x >> 16)) * 0x85ebca6bu;
        x      = (x ^ (x >> 13)) * 0xc2b2ae35u;
        return (x ^ (x >> 16)) + static_cast<uint32_t>(kv_tile) * kPSeedStep;
    }

    // 256 * P to fp8 with stochastic rounding, so small P are kept on average instead of flushed.
    CK_TILE_DEVICE static fp8x8_t QuantizePStochastic(const fp32x8_t& v, uint32_t seed, float scale)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        return bit_cast<fp8x8_t>(__builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(v, seed, scale));
#else
        ignore = seed;
        return fp32x8_to_fp8x8(v, scale);
#endif
    }

    // MX: one E8M0 per kMxGranularity elements along each GEMM's K dimension.
    static constexpr index_t kMxGranularity = 32;
    // E8M0 bytes of one Q/K scale row, and of one QK WMMA K step (K64 steps: two).
    static constexpr index_t kMxQkScaleBytes   = kQKHeaddim / kMxGranularity;
    static constexpr index_t kMxStepScaleBytes = Geometry::kQkWmmaK / kMxGranularity;
    // One scale dword per QK WMMA K step; D192/D256 keep each step's Q/K scales in its own plane.
    static constexpr index_t kMxQkScaleDwords   = Geometry::kQkKIter;
    static constexpr index_t kMxKScalesPerPlane = Geometry::kQkStages * Geometry::kQkSuNIter;
    // K64 steps pad the two scale bytes of their zero K half to 1.0.
    static constexpr int32_t kMxQkPadScale = kMxStepScaleBytes == 2 ? 0x7f7f0000 : 0;
    // Pointer and stride alignment of the Q/K scale rows read by LoadMxRowScale.
    static constexpr index_t kMxQkScaleAlign = min(kMxStepScaleBytes, kScaleBytes);
    // V scale rows each thread copies: two at D256; K rows: kN / kBlockSize (one-wavefront: four).
    static constexpr index_t kMxVRowsPerThread = max(Geometry::kDv / kBlockSize, index_t{1});
    static constexpr index_t kMxKRowsPerThread = Geometry::kN / kBlockSize;

    // Blocks past the last KV block read 0 from the bounds check; only masked P/S use them.
    CK_TILE_DEVICE static int32x4_t
    MakeKVDescaleResource(const float* ptr, index_t kv_last, index_t block_scale_size_kv)
    {
        const index_t num_blocks = kv_last / block_scale_size_kv + 1;
        return make_wave_buffer_resource(ptr, num_blocks * sizeof(float), std::true_type{});
    }

    CK_TILE_DEVICE static float
    LoadKVDescale(const int32x4_t& res, index_t kv, index_t block_scale_size_kv)
    {
        return amd_s_buffer_load_fp32(
            res, kv / block_scale_size_kv * static_cast<index_t>(sizeof(float)));
    }

    CK_TILE_DEVICE static void LoadKVDescalesAt(const int32x4_t& res,
                                                index_t kv_tile_start,
                                                index_t block_scale_size_kv,
                                                KVDescales& descales)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) {
            descales(g) =
                LoadKVDescale(res, kv_tile_start + g * kKVScaleAlign, block_scale_size_kv);
        });
    }

    // Walks the KV groups in order, so each descale offset comes without a runtime division.
    struct KVScaleCursor
    {
        index_t offset;           // byte offset of the current group's descale
        index_t group_in_block;   // index of the current group within its descale block
        index_t groups_per_block; // block_scale_size_kv / kKVScaleAlign

        KVScaleCursor() = default;
        CK_TILE_DEVICE KVScaleCursor(index_t kv, index_t block_scale_size_kv)
        {
            const auto group  = static_cast<uint32_t>(kv) / kKVScaleAlign;
            groups_per_block  = block_scale_size_kv / kKVScaleAlign;
            const auto blocks = static_cast<uint32_t>(groups_per_block);
            offset            = static_cast<index_t>(group / blocks * sizeof(float));
            group_in_block    = static_cast<index_t>(group % blocks);
        }

        CK_TILE_DEVICE index_t Next()
        {
            const index_t current = offset;
            const bool wrap       = group_in_block + 1 == groups_per_block;
            group_in_block        = wrap ? 0 : group_in_block + 1;
            offset += wrap ? static_cast<index_t>(sizeof(float)) : 0;
            return current;
        }
    };

    using KVScaleOffsets = array<index_t, kKVScaleGroups>;

    CK_TILE_DEVICE static void AdvanceKVScaleCursor(KVScaleCursor& cursor, KVScaleOffsets& offsets)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) { offsets(g) = cursor.Next(); });
    }

    CK_TILE_DEVICE static void LoadKVDescalesFromOffsets(const int32x4_t& k_res,
                                                         const int32x4_t& v_res,
                                                         const KVScaleOffsets& offsets,
                                                         KVDescales& k_descales,
                                                         KVDescales& v_descales)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) {
            k_descales(g) = amd_s_buffer_load_fp32(k_res, offsets[g]);
            v_descales(g) = amd_s_buffer_load_fp32(v_res, offsets[g]);
        });
    }

    template <index_t Stage>
    CK_TILE_DEVICE static int32_t PackVScale(const KVDescales& v_descales, number<Stage>)
    {
        Packed4Scale_E8M0 packed;
        packed.data() = 0;
        static_for<0, kScaleBytes, 1>{}([&](auto i) {
            packed.pack_scale(e8m0_t(v_descales[number<Stage * kScaleBytes + i>{}]), i);
        });
        return FmhaTdmSchedFp8Pins::PinToVgpr(static_cast<int32_t>(packed.data()));
    }

    // Byte i of a WMMA scale operand covers K elements [32i, 32i + 32) of the lane's row.
    CK_TILE_DEVICE static int32_t LoadMxRowScale(const e8m0_t* ptr, index_t row, index_t stride)
    {
        if constexpr(kMxStepScaleBytes >= kScaleBytes)
            return *reinterpret_cast<const int32_t*>(ptr + row * stride);
        else
            return *reinterpret_cast<const uint16_t*>(ptr + row * stride) | kMxQkPadScale;
    }

    // D192 runs K64 steps 0+1 as one K128 WMMA, which takes its four scale bytes from plane 1.
    static constexpr bool kMxPairedK64 = kMxStepScaleBytes == 2 && kMxQkScaleDwords == 3;

    template <index_t N>
    CK_TILE_DEVICE static void PairMxQkScales(array<int32_t, N>& scales)
    {
        constexpr index_t kPlane = N / kMxQkScaleDwords;
        static_for<0, kPlane, 1>{}([&](auto i) {
            constexpr auto i1 = number<kPlane + i>{};
            scales(i1)        = (scales[i] & 0xffff) | (scales[i1] << 16);
        });
    }

    // K64-step K scale dwords hold two copied bytes; replace the upper two by the 1.0 pad scale.
    template <index_t N>
    CK_TILE_DEVICE static void PadMxKScales(array<int32_t, N>& scales)
    {
        if constexpr(kMxStepScaleBytes == 2)
            static_for<0, N, 1>{}(
                [&](auto i) { scales(i) = (scales[i] & 0xffff) | kMxQkPadScale; });
        if constexpr(kMxPairedK64)
            PairMxQkScales(scales);
    }

    // MX scales of a KV tile in LDS: kN K dwords per plane, then the V dwords (first kDv used).
    static constexpr index_t kMxKDwords       = Geometry::kN * kMxQkScaleDwords;
    static constexpr index_t kMxLdsTileDwords = kMxKDwords + kBlockSize * kMxVRowsPerThread;

    // core to_global is declared only for gfx125 device passes; this header parses in all passes.
    template <typename T>
    CK_TILE_DEVICE static T* ToGlobalPtr(const void* p)
    {
        return __builtin_bit_cast(T*, const_cast<void*>(p));
    }

    // Per-thread scale copy offsets that stay fixed across KV tiles.
    struct MxCopyOffsets
    {
        index_t k;        // offset of this thread's K scale row from the tile's first row
        index_t v;        // offset of this thread's V scale row
        index_t fast_end; // tiles starting below this are full and dword aligned
    };

    CK_TILE_DEVICE static MxCopyOffsets
    MakeMxCopyOffsets(const FmhaTdmSchedFp8MxDescales& mx, index_t kv_load_start, index_t kv_last)
    {
        const index_t t = get_thread_id();
        // Tiles advance by kN0 = kScaleBytes V blocks, so the V dword alignment never changes.
        static_assert(kN0 == kScaleBytes * kMxGranularity);
        const bool aligned =
            kv_load_start % kMxGranularity == 0 && mx.stride_v % kScaleBytes == 0 &&
            (reinterpret_cast<uintptr_t>(mx.v_ptr) + kv_load_start / kMxGranularity) %
                    kScaleBytes ==
                0;
        // Threads past kDv copy the last V row into spare dwords, which keeps the copy branch-free.
        const index_t v_row = Geometry::kDv == kBlockSize ? t : min(t, Geometry::kDv - 1);
        // Pinned so the compiler does not fold them back into per-tile multiplies (D128 only).
        const auto pin = [](index_t v) {
            return kMxQkScaleBytes == kScaleBytes ? FmhaTdmSchedFp8Pins::PinToVgpr(v) : v;
        };
        return {pin(t * mx.stride_k),
                pin(v_row * mx.stride_v),
                // kv_base < fast_end iff [kv_base, kv_base + kN0) ends by seqlen_k = kv_last + 1.
                aligned ? kv_last + 2 - kN0 : 0};
    }

    // Full tiles copy one K and one V dword per thread; edge tiles clamp K and copy V bytewise.
    template <bool kAllFast = false>
    CK_TILE_DEVICE static void IssueMxScalesToLds(const FmhaTdmSchedFp8MxDescales& mx,
                                                  const MxCopyOffsets& off,
                                                  int32_t* lds,
                                                  index_t kv_base,
                                                  index_t kv_last)
    {
        static_assert(Geometry::kN == kBlockSize * kMxKRowsPerThread &&
                      Geometry::kDv <= kBlockSize * kMxVRowsPerThread);
        using gint_t  = __attribute__((address_space(1))) int32_t;
        using gchar_t = __attribute__((address_space(1))) char;
        using lchar_t = __attribute__((address_space(3))) char;
        auto* k_dst   = fmha_tdm_sched_fp8_lds_ptr<int32_t>(fmha_tdm_sched_fp8_lds_offset(lds)) +
                      get_thread_id();
        auto* v_dst = k_dst + kMxKDwords;
        // Unsigned adds keep each lane offset one 32-bit value next to the scalar base.
        const auto lane = [](index_t tile, index_t lane_off) {
            return static_cast<index_t>(static_cast<uint32_t>(tile) +
                                        static_cast<uint32_t>(lane_off));
        };
        // A K64 step has two scale bytes, copied bytewise; PadMxKScales fills the rest.
        const auto copy_k = [&](const e8m0_t* src, auto* dst) {
            if constexpr(kMxQkScaleBytes == kScaleBytes)
                __builtin_amdgcn_global_load_async_to_lds_b32(ToGlobalPtr<gint_t>(src), dst, 0, 0);
            else if constexpr(kMxStepScaleBytes == kScaleBytes)
                static_for<0, kMxQkScaleDwords, 1>{}([&](auto j) {
                    __builtin_amdgcn_global_load_async_to_lds_b32(
                        ToGlobalPtr<gint_t>(src + j * kScaleBytes), dst + j * Geometry::kN, 0, 0);
                });
            else
                static_for<0, kMxQkScaleDwords, 1>{}([&](auto j) {
                    static_for<0, kMxStepScaleBytes, 1>{}([&](auto i) {
                        __builtin_amdgcn_global_load_async_to_lds_b8(
                            ToGlobalPtr<gchar_t>(src + j * kMxStepScaleBytes + i),
                            reinterpret_cast<lchar_t*>(dst + j * Geometry::kN) + i,
                            0,
                            0);
                    });
                });
        };
        const index_t k_off = lane(kv_base * mx.stride_k, off.k);
        // K row r of this thread is tile row t + r * kBlockSize.
        const auto k_row_off = [&](auto r) { return k_off + r * kBlockSize * mx.stride_k; };
        if(kAllFast || __builtin_expect(kv_base < off.fast_end, 1))
        {
            static_for<0, kMxKRowsPerThread, 1>{}(
                [&](auto r) { copy_k(mx.k_ptr + k_row_off(r), k_dst + r * kBlockSize); });
            __builtin_amdgcn_global_load_async_to_lds_b32(
                ToGlobalPtr<gint_t>(mx.v_ptr + lane(kv_base / kMxGranularity, off.v)), v_dst, 0, 0);
            static_for<1, kMxVRowsPerThread, 1>{}([&](auto r) {
                __builtin_amdgcn_global_load_async_to_lds_b32(
                    ToGlobalPtr<gint_t>(mx.v_ptr + lane(kv_base / kMxGranularity,
                                                        off.v + r * kBlockSize * mx.stride_v)),
                    v_dst + r * kBlockSize,
                    0,
                    0);
            });
        }
        else
        {
            static_for<0, kMxKRowsPerThread, 1>{}([&](auto r) {
                copy_k(mx.k_ptr + min(k_row_off(r), kv_last * mx.stride_k), k_dst + r * kBlockSize);
            });
            static_for<0, kMxVRowsPerThread, 1>{}([&](auto r) {
                static_for<0, kScaleBytes, 1>{}([&](auto i) {
                    const index_t blk = min(kv_base + i * kMxGranularity, kv_last) / kMxGranularity;
                    __builtin_amdgcn_global_load_async_to_lds_b8(
                        ToGlobalPtr<gchar_t>(mx.v_ptr +
                                             lane(blk, off.v + r * kBlockSize * mx.stride_v)),
                        reinterpret_cast<lchar_t*>(v_dst + r * kBlockSize) + i,
                        0,
                        0);
                });
            });
        }
    }

    // Lane l reads the scales of WMMA row/column i * 16 + l % 16; p points at its lane's first.
    template <typename LanePtr, index_t N>
    CK_TILE_DEVICE static void ReadMxScalesFromLds(LanePtr p, array<int32_t, N>& scales)
    {
        static_for<0, N, 1>{}([&](auto i) { scales(i) = p[i * Geometry::QkWmma::kN]; });
    }

    // LDS pointer from a byte address computed (and pinned) ahead of the barrier.
    CK_TILE_DEVICE static auto MxScaleLanePtr(int32_t lds_byte)
    {
        return fmha_tdm_sched_fp8_lds_ptr<const int32_t>(static_cast<uint32_t>(lds_byte));
    }

    CK_TILE_DEVICE static int32_t LdsByteAddress(const int32_t* p)
    {
        return static_cast<int32_t>(fmha_tdm_sched_fp8_lds_offset(p));
    }

    // Rescales row_sum like l and adds the exact all-ones-WMMA row sums of the fp8 P of gemm_1.
    template <typename Gemm1, typename PTile, typename RowTile>
    CK_TILE_DEVICE static void
    AccumulateQuantizedMass(const PTile& p_tile, RowTile& row_sum, float scale_m0, float scale_m1)
    {
        using WarpGemm    = typename Gemm1::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        static_assert(Gemm1::KIterPerWarp == 1 && Gemm1::MIterPerWarp == Geometry::kPvMIter &&
                      Geometry::kPvMIter == 2 && AWarpTensor::get_thread_buffer_size() == 64);
        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        // Four fp8 1.0 bytes per dword.
        const int32x8_t ones = int32x8_t{} + 0x38383838;
        auto& lq             = row_sum.get_thread_buffer();
        static_for<0, Geometry::kPvMIter, 1>{}([&](auto m_iter) {
            AWarpTensor a_warp_tensor;
            a_warp_tensor.get_thread_buffer() = p_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
            // gemm_1 runs P as WMMA source B, so each lane's sums belong to its query row.
            const auto p = bit_cast<int32x16_t>(a_warp_tensor.get_thread_buffer());
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
            fp32x8_t c = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                ones,
                __builtin_shufflevector(p, p, 0, 1, 2, 3, 4, 5, 6, 7),
                0,
                fp32x8_t{},
                false,
                false);
            c = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                ones,
                __builtin_shufflevector(p, p, 8, 9, 10, 11, 12, 13, 14, 15),
                0,
                c,
                false,
                false);
            const float s = c[0];
#else
            ignore        = p;
            ignore        = ones;
            const float s = 0.f;
#endif
            lq(m_iter) =
                __builtin_fmaf(decltype(m_iter)::value == 0 ? scale_m0 : scale_m1, lq[m_iter], s);
        });
    }
};

} // namespace ck_tile

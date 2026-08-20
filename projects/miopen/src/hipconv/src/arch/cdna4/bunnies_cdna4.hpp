#pragma once

#include "bunnies.hpp"
#include "detail.h"

#include <type_traits>

namespace bunnies
{

struct arch_cdna4
{
    static constexpr int wave_size = 64;

    // Products summed exactly within one f16xf16->f32 MFMA before it rounds.
    //
    // A lane's whole operand, so the cross-lane reduction is what rounds and one instruction costs
    // K/block - 1 roundings rather than K - 1. Measured on gfx950, identical for fp16 and bf16
    // operands; see docs/algorithms/direct/direct-wgrad-tolerance.md. The ISA documents neither
    // this nor its uniformity across operand types, so other MFMA families (fp8, fp64) and other
    // arches need their own measurement rather than inheriting this one.
    static constexpr int mfma_f16_f16_f32_exact_block = 8;

    using buffer_t = __amdgpu_buffer_rsrc_t;

    template <fpfmt Fmt, int Rows, int Cols, use Use>
    struct map_fun;
    template <>
    struct map_fun<fpfmt::e4m3, 16, 128, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 16 ^ x[1] / 16 * 64 ^ x[1] % 16};
        }
    };
    template <>
    struct map_fun<fpfmt::e4m3, 128, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 16 ^ x[1] / 16 * 64 ^ x[1] % 16, x[0] % 16};
        }
    };
    // 16-deep A, reached as the compressed base of the 1:4 staging operand: one 2-VGPR
    // block, so a lane-group holds a run of 4.
    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) struct map_fun<Fmt, 16, 16, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 4 ^ x[1]};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) struct map_fun<Fmt, 16, 32, use::A>
    {
        __device__ __host__ static constexpr auto
        map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 8 ^ x[1]};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) struct map_fun<Fmt, 32, 16, use::B>
    {
        __device__ __host__ static constexpr auto
        map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 8 ^ x[1], x[0] % 16};
        }
    };
    // smfmac_16x16x64 B (CDNA4 ISA 7.5.1.3): two stacked 32-deep halves, each laid out
    // like the dense 32x16 operand -- k = 32*(item/8) + 8*(L/16) + item%8, n = L%16.
    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) struct map_fun<Fmt, 64, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[1] / 8 * 32 ^ x[0] / 16 * 8 ^ x[1] % 8, x[0] % 16};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) struct map_fun<Fmt, 16, 16, use::Acc>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 4 ^ x[1], x[0] % 16};
        }
    };
    template <>
    struct map_fun<fpfmt::e8m23, 16, 16, use::Acc>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 4 ^ x[1], x[0] % 16};
        }
    };

    template <fpfmt Fmt, int Rows, int Cols, use Use>
    struct matrix
    {
        using arch                     = arch_cdna4;
        static constexpr fpfmt fmt     = Fmt;
        static constexpr int rows      = Rows;
        static constexpr int cols      = Cols;
        static constexpr use use_      = Use;
        static constexpr int num_items = Rows * Cols / wave_size;

        using base_storage_t = base_storage_type_t<fmt>;
        using storage_t      = storage_type_t<fmt, num_items>;
        storage_t data;

        __device__ __host__ static constexpr auto
        map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return map_fun<Fmt, Rows, Cols, Use>::map(x);
        }
    };

    // A structured-sparse operand: compressed values plus their sparsity index.
    //
    // Storage is the dense compressed matrix (K scaled by the compression ratio), while
    // rows/cols advertise the logical shape. `idx` follows the smfmac Src2 encoding: one
    // nibble per group, holding the 2-bit in-group position of each of its live slots.
    // The 1:4 staging form keeps that nibble spacing, so the cast to 2:4 copies it as is.
    template <fpfmt Fmt, int Rows, int Cols, use Use, sparsity Sprs>
    struct sparse_matrix
        : private matrix<Fmt, Rows, Cols * live_per_group(Sprs) / sparsity_group_size(Sprs), Use>
    {
        static constexpr int live_per_grp    = live_per_group(Sprs);
        static constexpr int group_size      = sparsity_group_size(Sprs);
        static constexpr int compressed_cols = Cols * live_per_grp / group_size;
        using base                           = matrix<Fmt, Rows, compressed_cols, Use>;
        static constexpr int cols            = Cols;

        using base::data;
        using base::fmt;
        using base::num_items;
        using typename base::arch;
        using typename base::base_storage_t;

        uint32_t idx;

        // (lane,item) -> (row, compressed col): the compressed matrix's own layout, so
        // load_tile / store_tile drive a sparse tile exactly like a dense one. The
        // logical column additionally needs the index field; see to_logical.
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return base::map(x);
        }

        // In-group position of compressed item `item`.
        //
        // Position fields sit at a fixed group_size / live_per_grp bit stride: 2 for 2:4
        // (both slots share a nibble), 4 for 1:4 (one slot leaves its nibble half empty).
        __device__ auto position(int item) const -> int
        {
            return (idx >> (group_size / live_per_grp * item)) & 3;
        }

        // (lane,item) -> logical (row, col), i.e. `map` composed with the index.
        __device__ auto to_logical(std::array<int, 2> const& x) const -> std::array<int, 2>
        {
            const auto coord = map(x);
            return {coord[0], coord[1] / live_per_grp * group_size + position(x[1])};
        }

        // Fill every compressed slot, map-driven like load_tile.
        //
        // `pos(row, group)` picks the live position within the group; `value` then
        // supplies the element at the resulting logical coordinate.
        template <typename Pos, typename Value>
        __device__ void fill(Pos&& pos, Value&& value)
        {
            static_assert(Sprs == sparsity::n1of4, "fill places one live value per group");
            const int lane = lane_id();
            idx            = 0;
            static_for<base::num_items>([&]<int item>() {
                const auto coord = map({lane, item});
                const int p      = pos(coord[0], coord[1]);
                idx |= p << (4 * item);
                this->data[item] = value(coord[0], coord[1] * group_size + p);
            });
        }
    };

    // 1:4 -> 2:4: spread each live value into the even slot of its 2:4 slot pair.
    //
    // Lane-local by construction (the 1:4 layout is the 2:4 one pulled back along
    // item -> 2*item) and the nibble spacing matches, so the index copies verbatim.
    // The zero padding slot keeps position 0, which may name the same dense column as
    // the live slot; harmless because its value is zero.
    template <fpfmt Fmt, int Rows, int Cols, use Use>
    inline __device__ static void
    matrix_cast(sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n2of4>& dest,
                sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n1of4> const& src)
    {
        using src_t  = sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n1of4>;
        using elem_t = base_storage_type_t<Fmt>;
        static_for<src_t::num_items>([&]<int item>() {
            dest.data[2 * item]     = src.data[item];
            dest.data[2 * item + 1] = elem_t(0);
        });
        dest.idx = src.idx;
    }

    template <fpfmt Fmt>
    requires(is_16bit<Fmt>) inline __device__
        static void matrix_cast(matrix<Fmt, 16, 16, use::Acc>& dest,
                                matrix<fpfmt::e8m23, 16, 16, use::Acc> const& src)
    {
        for(int item = 0; item < dest.matrix::num_items; ++item)
        {
            dest.data[item] = static_cast<base_storage_type_t<Fmt>>(src.data[item]);
        }
    }

    template <uint32_t flags = 0>
    struct mma
    {
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e5m10, 16, 32, use::A>& a,
                                    matrix<fpfmt::e5m10, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_mfma_f32_16x16x32_f16(a.data, b.data, c.data, 0, 0, 0);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e8m7, 16, 32, use::A>& a,
                                    matrix<fpfmt::e8m7, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a.data, b.data, c.data, 0, 0, 0);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e4m3, 16, 128, use::A>& a,
                                    matrix<fpfmt::e4m3, 128, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            constexpr int scale = 0;
            d.data              = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a.data, b.data, c.data, 0, 0, 0, scale, 0, scale);
        }
    };

    // 2:4 structured-sparse MFMA (V_SMFMAC_F32_16X16X64).
    //
    // A is a compressed sparse operand carrying its own sparsity index; B is the
    // dense 64x16 operand; D/C are 16x16. Only n2of4 is accepted, so a 1:4 staging
    // operand has to pass through matrix_cast first.
    struct smma
    {
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    sparse_matrix<fpfmt::e5m10, 16, 64, use::A, sparsity::n2of4>& a,
                                    matrix<fpfmt::e5m10, 64, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_smfmac_f32_16x16x64_f16(
                a.data, b.data, c.data, static_cast<int>(a.idx), 0, 0);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    sparse_matrix<fpfmt::e8m7, 16, 64, use::A, sparsity::n2of4>& a,
                                    matrix<fpfmt::e8m7, 64, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_smfmac_f32_16x16x64_bf16(
                a.data, b.data, c.data, static_cast<int>(a.idx), 0, 0);
        }
    };

    template <typename T>
    __device__ static auto make_buffer(T* global_ptr, int64_t global_size) -> buffer_t
    {
        constexpr std::int32_t data_format = 1 << 15;
        return __builtin_amdgcn_make_buffer_rsrc(const_cast<std::remove_const_t<T>*>(global_ptr),
                                                 0,
                                                 global_size * sizeof(T),
                                                 data_format);
    }

    template <int BytesPerLane>
    struct buffer_load_lds
    {
        __device__ static void load(buffer_t buffer, void* lds_ptr, int v_offset, int s_offset)
        {
            if constexpr(BytesPerLane == 16)
            {
                // Guarded here, not at the calling kernel: the literal size is checked
                // when this template is defined, so `if constexpr` does not hide it.
#ifdef __gfx950__
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    buffer, lds_ptr, 16, v_offset, s_offset, 0, 0);
#endif
            }
            else if constexpr(BytesPerLane == 4)
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    buffer, lds_ptr, 4, v_offset, s_offset, 0, 0);
            }
            else if constexpr(BytesPerLane == 2)
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    buffer, lds_ptr, 2, v_offset, s_offset, 0, 0);
            }
            else
            {
                static_assert(false, "BytesPerLane must be 2, 4, or 16");
            }
        }
    };

    template <int BytesPerLane>
    struct buffer_store
    {
        __device__ static void store(buffer_t buffer, void* src, int v_offset, int s_offset)
        {
            if constexpr(BytesPerLane == 16)
            {
                __builtin_amdgcn_raw_buffer_store_b128(
                    *static_cast<uint32x4*>(src), buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 12)
            {
                __builtin_amdgcn_raw_buffer_store_b96(
                    *static_cast<uint32x3*>(src), buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 8)
            {
                __builtin_amdgcn_raw_buffer_store_b64(
                    *static_cast<uint32x2*>(src), buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 4)
            {
                __builtin_amdgcn_raw_buffer_store_b32(
                    *static_cast<uint32_t*>(src), buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 2)
            {
                __builtin_amdgcn_raw_buffer_store_b16(
                    *static_cast<uint16_t*>(src), buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 1)
            {
                __builtin_amdgcn_raw_buffer_store_b8(
                    *static_cast<uint8_t*>(src), buffer, v_offset, s_offset, 0);
            }
            else
            {
                static_assert(false, "BytesPerLane must be 1, 2, 4, 8, 12, or 16.");
            }
        }
    };

    // Symmetric to buffer_store: a raw buffer (V#) load dispatched on per-lane byte
    // width, writing the result into `dest`.
    //
    // Buffer addressing gives free hardware bounds checking (OOB lanes read 0).
    template <int BytesPerLane>
    struct buffer_load
    {
        __device__ static void load(buffer_t buffer, void* dest, int v_offset, int s_offset)
        {
            if constexpr(BytesPerLane == 16)
            {
                *static_cast<uint32x4*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b128(buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 12)
            {
                *static_cast<uint32x3*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b96(buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 8)
            {
                *static_cast<uint32x2*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b64(buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 4)
            {
                *static_cast<uint32_t*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b32(buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 2)
            {
                *static_cast<uint16_t*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b16(buffer, v_offset, s_offset, 0);
            }
            else if constexpr(BytesPerLane == 1)
            {
                *static_cast<uint8_t*>(dest) =
                    __builtin_amdgcn_raw_buffer_load_b8(buffer, v_offset, s_offset, 0);
            }
            else
            {
                static_assert(false, "BytesPerLane must be 1, 2, 4, 8, 12, or 16.");
            }
        }
    };

    template <int BytesPerLane>
    struct global_or_ds_load
    {
        using type                         = packed_type<BytesPerLane>;
        static constexpr int bits_per_load = BytesPerLane * 8;
        inline __device__ static auto map(int lane, int item, int) -> std::array<int, 2>
        {
            return {lane, item};
        }
        inline __device__ static void load(void* ptr, void* dest)
        {
            *reinterpret_cast<type*>(dest) = *reinterpret_cast<type*>(ptr);
        }
    };
    template <int BytesPerLane>
    using ds_load      = global_or_ds_load<BytesPerLane>;
    using ds_load_b32  = ds_load<4>;
    using ds_load_b64  = ds_load<8>;
    using ds_load_b96  = ds_load<12>;
    using ds_load_b128 = ds_load<16>;
    template <int BytesPerLane>
    using global_load      = global_or_ds_load<BytesPerLane>;
    using global_load_b32  = global_load<4>;
    using global_load_b64  = global_load<8>;
    using global_load_b96  = global_load<12>;
    using global_load_b128 = global_load<16>;

    struct ds_read_b64_tr_b16
    {
        using type                         = int16x4;
        static constexpr int bits_per_load = 64;
        inline __device__ __host__ static constexpr auto
        map(int lane, int item, int bits_per_item) -> std::array<int, 2>
        {
            const auto num_items = bits_per_load / bits_per_item;
            const auto item0     = item / num_items * num_items;
            item                 = item % num_items;
            return {lane % 4 * 4 ^ lane / 16 * 16 ^ item, lane / 4 % 4 ^ item0};
        }
        inline __device__ static void load(void* lds_ptr, void* dest)
        {
            *reinterpret_cast<type*>(dest) =
                __builtin_amdgcn_ds_read_tr16_b64_v4i16(reinterpret_cast<type*>(lds_ptr));
        }
    };

    struct ds_read_b64_tr_b8
    {
        using type                         = int32x2;
        static constexpr int bits_per_load = 64;
        inline __device__ static auto
        map(int lane, int item, int bits_per_item) -> std::array<int, 2>
        {
            const auto num_items = bits_per_load / bits_per_item;
            const auto item0     = item / num_items * num_items;
            item                 = item % num_items;
            return {lane % 2 * 8 ^ lane / 16 * 16 ^ item, lane / 2 % 8 ^ item0};
        }
        inline __device__ static void load(void* lds_ptr, void* dest)
        {
            *reinterpret_cast<type*>(dest) =
                __builtin_amdgcn_ds_read_tr8_b64_v2i32(reinterpret_cast<type*>(lds_ptr));
        }
    };

    template <int BytesPerLane>
    struct global_or_ds_store
    {
        using type                          = packed_type<BytesPerLane>;
        static constexpr int bits_per_store = BytesPerLane * 8;
        inline __device__ static auto map(int lane, int item, int) -> std::array<int, 2>
        {
            return {lane, item};
        }
        inline __device__ static void store(void* lds_ptr, void* dest)
        {
            *reinterpret_cast<type*>(lds_ptr) = *reinterpret_cast<type*>(dest);
        }
    };
    template <int BytesPerLane>
    using ds_store      = global_or_ds_store<BytesPerLane>;
    using ds_store_b32  = ds_store<4>;
    using ds_store_b64  = ds_store<8>;
    using ds_store_b96  = ds_store<12>;
    using ds_store_b128 = ds_store<16>;
    template <int BytesPerLane>
    using global_store      = global_or_ds_store<BytesPerLane>;
    using global_store_b32  = global_store<4>;
    using global_store_b64  = global_store<8>;
    using global_store_b96  = global_store<12>;
    using global_store_b128 = global_store<16>;

    static constexpr uint16_t max_vmcnt   = 63;
    static constexpr uint16_t max_lgkmcnt = 15;
    static constexpr uint16_t max_expcnt  = 7;
    // Pack a waitcnt SIMM16 from separate vm/lgkm/exp counts.
    //
    // SIMM16[3:0] = vmcount low bits, [6:4] = export/mem-write count,
    // [11:8] = LGKMcnt (scalar-mem/GDS/LDS), [15:14] = vmcount high bits.
    inline __device__ static constexpr auto makecnt(uint16_t vmcnt   = max_vmcnt,
                                                    uint16_t lgkmcnt = max_lgkmcnt,
                                                    uint16_t expcnt  = max_expcnt) -> uint16_t
    {
        const uint16_t vmbits   = vmcnt & 0xF | (vmcnt & 0x30) << (14 - 4);
        const uint16_t lgkmbits = (lgkmcnt & 0xF) << 8;
        const uint16_t expbits  = (expcnt & 0x7) << 4;
        return vmbits | lgkmbits | expbits;
    }
    template <uint16_t Cnt>
    __device__ static void s_wait_vmcnt()
    {
        __builtin_amdgcn_s_waitcnt(makecnt(Cnt));
    }
    template <uint16_t Cnt>
    __device__ static void s_wait_lgkmcnt()
    {
        __builtin_amdgcn_s_waitcnt(makecnt(max_vmcnt, Cnt));
    }
    template <uint16_t Cnt>
    __device__ static void s_wait_expcnt()
    {
        __builtin_amdgcn_s_waitcnt(makecnt(max_vmcnt, max_lgkmcnt, Cnt));
    }
};

} // namespace bunnies

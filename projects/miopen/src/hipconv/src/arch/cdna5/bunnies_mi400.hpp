#pragma once

#include "bunnies.hpp"
#include "detail.h"

#include <bit>
#include <cstddef>

namespace bunnies
{

namespace detail
{
template <typename T>
concept has_num_items = requires(T t)
{
    {t.num_items};
};
} // namespace detail

struct arch_mi400
{
    static constexpr int wave_size         = 32;
    static constexpr std::size_t lds_bytes = 320 * 1024;
    using buffer_t                         = __amdgpu_buffer_rsrc_t;

    template <fpfmt Fmt, int Rows, int Cols, use Use>
    struct layout_config;
    template <>
    struct layout_config<fpfmt::e4m3, 16, 128, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 16 ^ x[1] / 16 * 32 ^ x[1] % 16};
        }
    };
    template <>
    struct layout_config<fpfmt::e4m3, 128, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 16 ^ x[1] / 16 * 32 ^ x[1] % 16, x[0] % 16};
        }
    };
    template <>
    struct layout_config<fpfmt::e4m3, 32, 128, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16 ^ x[1] / 64 * 16, x[0] / 16 * 16 ^ x[1] / 16 % 4 * 32 ^ x[1] % 16};
        }
    };
    template <>
    struct layout_config<fpfmt::e4m3, 128, 32, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 16 ^ x[1] / 16 % 4 * 32 ^ x[1] % 16, x[0] % 16 ^ x[1] / 64 * 16};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10) struct layout_config<Fmt, 16, 32, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 8 ^ x[1] / 8 * 16 ^ x[1] % 8};
        }
    };
    // 1:4 staging A is the 2:4 packed 16x32 layout above pulled back along item -> 2*item+1:
    // each staging item carries the value that matrix_cast places in the odd slot of its pair.
    // Its column is therefore the expanded K-group, so load_tile / store_tile address (row, group).
    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10) struct layout_config<Fmt, 16, 16, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 4 ^ x[1] / 4 * 8 ^ x[1] % 4};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10) struct layout_config<Fmt, 32, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 8 ^ x[1] / 8 * 16 ^ x[1] % 8, x[0] % 16};
        }
    };
    // Full B for swmmac_16x16x64 (32 items/lane), MI400 Shader Programming Guide 4.6.12.4:
    // k = 32*(item/16) + 16*(lane/16) + item%16; n = lane%16.
    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10) struct layout_config<Fmt, 64, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[1] / 16 * 32 ^ x[0] / 16 * 16 ^ x[1] % 16, x[0] % 16};
        }
    };
    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10 ||
             Fmt == fpfmt::e8m23) struct layout_config<Fmt, 16, 16, use::Acc>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 8 ^ x[1], x[0] % 16};
        }
    };
    template <>
    struct layout_config<fpfmt::e8m23, 32, 32, use::Acc>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 8 ^ x[1] / 8 % 2 * 16 ^ x[1] % 8 ^ x[1] / 32 * 16,
                    x[0] % 16 ^ x[1] / 16 % 2 * 16 ^ x[1] / 32 * 16};
        }
    };

    template <fpfmt Fmt, int Rows, int Cols, use Use>
    struct matrix
    {
        using arch                     = arch_mi400;
        using cfg                      = layout_config<Fmt, Rows, Cols, Use>;
        static constexpr fpfmt fmt     = Fmt;
        static constexpr int rows      = Rows;
        static constexpr int cols      = Cols;
        static constexpr use use_      = Use;
        static constexpr int num_items = [] {
            if constexpr(detail::has_num_items<cfg>)
            {
                return cfg::num_items;
            }
            return Rows * Cols / wave_size;
        }();

        using base_storage_t = base_storage_type_t<fmt>;
        using storage_t      = storage_type_t<fmt, num_items>;
        storage_t data;

        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return cfg::map(x);
        }
    };

    template <int Rows, int Cols, use Use>
    struct matrix<fpfmt::e8m10_e8m7x2split, Rows, Cols, Use>
    {
        using arch                     = arch_mi400;
        static constexpr fpfmt fmt     = fpfmt::e8m10_e8m7x2split;
        static constexpr int rows      = Rows;
        static constexpr int cols      = Cols;
        static constexpr use use_      = Use;
        static constexpr int num_items = Rows * Cols / wave_size;

        using storage_t = storage_type_t<fpfmt::e8m7, num_items>;
        storage_t big;
        storage_t small;
    };

    // A structured-sparse operand: compressed values plus the swmmac Src2 index. As on CDNA4, the
    // 1:4 staging form inherits a 16-column matrix and the hardware 2:4 form a 32-column matrix;
    // private inheritance shares matrix machinery without allowing a sparse operand to bind wmma.
    template <fpfmt Fmt, int Rows, int Cols, use Use, sparsity Sprs>
    struct sparse_matrix
        : private matrix<Fmt, Rows, Cols * live_per_group(Sprs) / sparsity_group_size(Sprs), Use>
    {
        static constexpr int live_per_grp    = live_per_group(Sprs);
        static constexpr int group_size      = sparsity_group_size(Sprs);
        static constexpr int compressed_cols = Cols * live_per_grp / group_size;
        using base                           = matrix<Fmt, Rows, compressed_cols, Use>;
        static constexpr int rows            = Rows;
        static constexpr int cols            = Cols;

        using base::data;
        using base::fmt;
        using base::num_items;
        using typename base::arch;
        using typename base::base_storage_t;

        uint32_t idx;

        // (lane,item) -> (row, expanded K-group) for the data slot. The 1:4 staging base already
        // names groups; the 2:4 packed A has two columns per group.
        __device__ static constexpr auto
        group_map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            const auto coord = base::map(x);
            if constexpr(Sprs == sparsity::n1of4)
                return coord;
            else
                return {coord[0], coord[1] / 2};
        }

        // The 1:4 staging tile exposes one logical column per group to load_tile / store_tile.
        __device__ static constexpr auto
        map(std::array<int, 2> const& x) -> std::array<int, 2> requires(Sprs == sparsity::n1of4)
        {
            return group_map(x);
        }

        // In-group position of compressed item `item`. The fields sit at a group_size /
        // live_per_grp bit stride: 2 for 2:4, sharing a nibble, 4 for 1:4, leaving one half empty.
        __device__ auto position(int item) const -> int
        {
            return (idx >> (group_size / live_per_grp * item)) & 3;
        }

        // Fill every compressed slot. Data follows the packed 16x32 A layout above. Index does not:
        // guide 4.6.12.4 assigns groups 0..7 to lanes 0..15 and groups 8..15 to lanes 16..31,
        // independently of which lane holds the group's packed A values, so generate its eight
        // lane-local nibbles separately.
        template <typename Pos, typename Value>
        __device__ void fill(Pos&& pos, Value&& value)
        {
            static_assert(Sprs == sparsity::n1of4, "fill places one live value per group");
            const int lane = lane_id();
            static_for<num_items>([&]<int item>() {
                const auto coord = group_map({lane, item});
                const int p      = pos(coord[0], coord[1]);
                data[item]       = value(coord[0], coord[1] * group_size + p);
            });
            idx = 0;
            static_for<num_items>([&]<int item>() {
                const int row = lane % 16;
                const int g   = 8 * (lane / 16) + item;
                idx |= (uint32_t)pos(row, g) << (4 * item);
            });
        }
    };

    // The tf32 split of a 2:4 operand. Splitting is per-element, so both halves keep the source's
    // pattern and its index: one `idx` serves all three MMAs. MMA-only, so no layout and no fill.
    template <int Rows, int Cols, use Use>
    struct sparse_matrix<fpfmt::e8m10_e8m7x2split, Rows, Cols, Use, sparsity::n2of4>
    {
        using arch                           = arch_mi400;
        static constexpr fpfmt fmt           = fpfmt::e8m10_e8m7x2split;
        static constexpr int live_per_grp    = live_per_group(sparsity::n2of4);
        static constexpr int group_size      = sparsity_group_size(sparsity::n2of4);
        static constexpr int compressed_cols = Cols * live_per_grp / group_size;
        static constexpr int rows            = Rows;
        static constexpr int cols            = Cols;
        static constexpr use use_            = Use;
        static constexpr int num_items       = Rows * compressed_cols / wave_size;

        using storage_t = storage_type_t<fpfmt::e8m7, num_items>;
        storage_t big;
        storage_t small;
        uint32_t idx;
    };

    // 1:4 -> 2:4: spread each live value into the odd slot of its 2:4 pair, lane-local because the
    // 1:4 layout is the 2:4 one pulled back along item -> 2*item + 1. gfx1250 keeps the live value
    // in the odd slot where CDNA4 uses the even one, so the index shifts up by one 2-bit field.
    template <fpfmt Fmt, int Rows, int Cols, use Use>
    inline __device__ static void
    matrix_cast(sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n2of4>& dest,
                sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n1of4> const& src)
    {
        using src_t  = sparse_matrix<Fmt, Rows, Cols, Use, sparsity::n1of4>;
        using elem_t = base_storage_type_t<Fmt>;
        static_for<src_t::num_items>([&]<int item>() {
            dest.data[2 * item]     = elem_t(0);
            dest.data[2 * item + 1] = src.data[item];
        });
        dest.idx = src.idx << 2;
    }

    // A 2:4 fp32 operand to its split form; the pattern is untouched, so the index carries over.
    template <int Rows, int Cols, use Use>
    inline __device__ static void
    matrix_cast(sparse_matrix<fpfmt::e8m10_e8m7x2split, Rows, Cols, Use, sparsity::n2of4>& dest,
                sparse_matrix<fpfmt::e8m10, Rows, Cols, Use, sparsity::n2of4> const& src)
    {
        dest.big   = packed_convert<bf16_t>(src.data);
        dest.small = packed_convert<bf16_t>(src.data - packed_convert<fp32_t>(dest.big));
        dest.idx   = src.idx;
    }

    // The tf32 form of the spread above, straight to the split the MMA takes so the intermediate
    // fp32 2:4 operand never occupies registers.
    template <int Rows, int Cols, use Use>
    inline __device__ static void
    matrix_cast(sparse_matrix<fpfmt::e8m10_e8m7x2split, Rows, Cols, Use, sparsity::n2of4>& dest,
                sparse_matrix<fpfmt::e8m10, Rows, Cols, Use, sparsity::n1of4> const& src)
    {
        using src_t = sparse_matrix<fpfmt::e8m10, Rows, Cols, Use, sparsity::n1of4>;
        static_for<src_t::num_items>([&]<int item>() {
            const fp32_t v           = src.data[item];
            const bf16_t big         = static_cast<bf16_t>(v);
            dest.big[2 * item]       = bf16_t(0);
            dest.big[2 * item + 1]   = big;
            dest.small[2 * item]     = bf16_t(0);
            dest.small[2 * item + 1] = static_cast<bf16_t>(v - static_cast<fp32_t>(big));
        });
        dest.idx = src.idx << 2;
    }

    // scales
    template <>
    struct layout_config<fpfmt::ue8m0, 16, 4, use::A>
    {
        static constexpr int num_items = 16 * 4 / (wave_size / 2);
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[1]};
        }
    };
    template <>
    struct layout_config<fpfmt::ue8m0, 4, 16, use::B>
    {
        static constexpr int num_items = 4 * 16 / (wave_size / 2);
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[1], x[0] % 16};
        }
    };
    template <>
    struct layout_config<fpfmt::ue8m0, 32, 4, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0], x[1]};
        }
    };
    template <>
    struct layout_config<fpfmt::ue8m0, 4, 32, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[1], x[0]};
        }
    };
    template <fpfmt Fmt, int Rows, int Cols, use Use>
    struct scale_matrix
    {
        using arch                     = arch_mi400;
        static constexpr fpfmt fmt     = Fmt;
        static constexpr int rows      = Rows;
        static constexpr int cols      = Cols;
        static constexpr use use_      = Use;
        static constexpr int num_items = Rows * Cols / wave_size;

        using base_storage_t = base_storage_type_t<fmt>;
        using storage_t      = storage_type_t<fmt, num_items>;
        storage_t data;

        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return layout_config<Fmt, Rows, Cols, Use>::map(x);
        }
    };

    // Every format except tf32 computes on its storage fragment, so a kernel that
    // splits `data` (loaded) from a compute operand can cast unconditionally and
    // only tf32 pays a real conversion; here the two coincide.
    template <fpfmt Fmt, int Rows, int Cols, use Use>
    inline __device__ static void matrix_cast(matrix<Fmt, Rows, Cols, Use>& dest,
                                              matrix<Fmt, Rows, Cols, Use> const& src)
    {
        dest.data = src.data;
    }

    template <fpfmt Fmt>
    requires(is_16bit<Fmt> || Fmt == fpfmt::e8m10) inline __device__
        static void matrix_cast(matrix<Fmt, 16, 16, use::Acc>& dest,
                                matrix<fpfmt::e8m23, 16, 16, use::Acc> const& src)
    {
        dest.data = packed_convert<base_storage_type_t<Fmt>>(src.data);
    }


    // Shape-agnostic, so 16x32 A, 32x16 B and swmmac's 64x16 B share one definition.
    template <int Rows, int Cols, use Use>
    inline __device__ static void
    matrix_cast(matrix<fpfmt::e8m10_e8m7x2split, Rows, Cols, Use>& dest,
                matrix<fpfmt::e8m10, Rows, Cols, Use> const& src)
    {
        dest.big   = packed_convert<bf16_t>(src.data);
        dest.small = packed_convert<bf16_t>(src.data - packed_convert<fp32_t>(dest.big));
    }

    template <uint32_t flags = 0>
    struct mma
    {
        static constexpr bool A_reuse = test(flags, wmma_flag::A_reuse);
        static constexpr bool B_reuse = test(flags, wmma_flag::B_reuse);

        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e5m10, 16, 32, use::A>& a,
                                    matrix<fpfmt::e5m10, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_wmma_f32_16x16x32_f16(
                false, a.data, false, b.data, 0, c.data, A_reuse, B_reuse);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e8m7, 16, 32, use::A>& a,
                                    matrix<fpfmt::e8m7, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                false, a.data, false, b.data, 0, c.data, A_reuse, B_reuse);
        }
        // bf16 in, f32 accumulate, bf16 output (mixed CD type). Folds the
        // f32->bf16 rounding into the MMA so an epilogue convert can be skipped;
        // round_to_bf16(a*b + c) is identical to converting the f32 result.
        __device__ static void wmma(matrix<fpfmt::e8m7, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e8m7, 16, 32, use::A>& a,
                                    matrix<fpfmt::e8m7, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(
                false, a.data, false, b.data, 0, c.data, A_reuse, B_reuse);
            // workaround for the hazard. TODO: remove this once the hazard is fixed.
            // Unlike the f32-output WMMAs, this one writes a D narrower than C instead of
            // accumulating in place, so C dies at the MMA and the scheduler fills the MMA's
            // shadow with VALU writes to C's registers. The hardware is still reading C's
            // tail there, and clang 23.0.0git does not model that hazard: the write lands on
            // the dword holding accumulator element 6, which then reads as zero. The barrier
            // keeps the shadow empty.
            __builtin_amdgcn_sched_barrier(0);
        }
        // f16 in, f32 accumulate, f16 output. No hardware mixed-CD f16 WMMA
        // exists, so this is the regular f32 WMMA plus an explicit f32->f16
        // convert. It exists purely to give the f16 path the same half-output
        // wmma signature as bf16's wmma_bf16f32, so callers need no per-dtype
        // branch (the convert that bf16 fuses into the MMA is just emitted
        // separately here).
        __device__ static void wmma(matrix<fpfmt::e5m10, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e5m10, 16, 32, use::A>& a,
                                    matrix<fpfmt::e5m10, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            matrix<fpfmt::e8m23, 16, 16, use::Acc> acc;
            acc.data = __builtin_amdgcn_wmma_f32_16x16x32_f16(
                false, a.data, false, b.data, 0, c.data, A_reuse, B_reuse);
            matrix_cast(d, acc);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e8m10_e8m7x2split, 16, 32, use::A>& a,
                                    matrix<fpfmt::e8m10_e8m7x2split, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                false, a.small, false, b.big, 0, c.data, 0, 0);
            d.data = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                false, a.big, false, b.big, 0, d.data, 0, 1);
            d.data = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                false, a.big, false, b.small, 0, d.data, 1, 0);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e8m10, 16, 32, use::A>& a,
                                    matrix<fpfmt::e8m10, 32, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            matrix<fpfmt::e8m10_e8m7x2split, 16, 32, use::A> a_split;
            matrix<fpfmt::e8m10_e8m7x2split, 32, 16, use::B> b_split;
            matrix_cast(a_split, a);
            matrix_cast(b_split, b);
            wmma(d, a_split, b_split, c);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    matrix<fpfmt::e4m3, 16, 128, use::A>& a,
                                    matrix<fpfmt::e4m3, 128, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                a.data, b.data, 0, c.data, A_reuse, B_reuse);
        }
        __device__ static void wmma_scale(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                          matrix<fpfmt::e4m3, 16, 128, use::A>& a,
                                          matrix<fpfmt::e4m3, 128, 16, use::B>& b,
                                          matrix<fpfmt::e8m23, 16, 16, use::Acc>& c,
                                          matrix<fpfmt::ue8m0, 16, 4, use::A>& a_scale,
                                          matrix<fpfmt::ue8m0, 4, 16, use::B>& b_scale)
        {
            d.data = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0,
                                                                      a.data,
                                                                      0,
                                                                      b.data,
                                                                      0,
                                                                      c.data,
                                                                      0,
                                                                      0,
                                                                      a_scale.data[0],
                                                                      0,
                                                                      0,
                                                                      b_scale.data[0],
                                                                      A_reuse,
                                                                      B_reuse);
        }
        __device__ static void wmma_scale(matrix<fpfmt::e8m23, 32, 32, use::Acc>& d,
                                          matrix<fpfmt::e4m3, 32, 128, use::A>& a,
                                          matrix<fpfmt::e4m3, 128, 32, use::B>& b,
                                          matrix<fpfmt::e8m23, 32, 32, use::Acc>& c,
                                          matrix<fpfmt::ue8m0, 32, 4, use::A>& a_scale,
                                          matrix<fpfmt::ue8m0, 4, 32, use::B>& b_scale)
        {
            static_unroll<2>([&](auto nb) {
                static_unroll<2>([&](auto mb) {
                    auto asub             = reinterpret_cast<uint32x16*>(&a.data) + mb;
                    auto bsub             = reinterpret_cast<uint32x16*>(&b.data) + nb;
                    auto csub             = reinterpret_cast<floatx8*>(&c.data) + mb + 2 * nb;
                    auto dsub             = reinterpret_cast<floatx8*>(&d.data) + mb + 2 * nb;
                    constexpr int opsel_b = nb;
                    constexpr int opsel_a = mb;
                    *dsub                 = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0,
                                                                             *asub,
                                                                             0,
                                                                             *bsub,
                                                                             0,
                                                                             *csub,
                                                                             opsel_a,
                                                                             0,
                                                                             a_scale.data[0],
                                                                             opsel_b,
                                                                             0,
                                                                             b_scale.data[0],
                                                                             A_reuse,
                                                                             B_reuse);
                });
            });
        }
    };

    // 2:4 structured-sparse WMMA (swmmac_f32_16x16x64). A is the compressed sparse
    // operand carrying its own index; B is the full 64x16 operand; D/C are 16x16 f32.
    template <uint32_t flags = 0>
    struct smma
    {
        static constexpr bool A_reuse = test(flags, wmma_flag::A_reuse);
        static constexpr bool B_reuse = test(flags, wmma_flag::B_reuse);

        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    sparse_matrix<fpfmt::e5m10, 16, 64, use::A, sparsity::n2of4>& a,
                                    matrix<fpfmt::e5m10, 64, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_swmmac_f32_16x16x64_f16(
                false, a.data, false, b.data, c.data, static_cast<int>(a.idx), A_reuse, B_reuse);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    sparse_matrix<fpfmt::e8m7, 16, 64, use::A, sparsity::n2of4>& a,
                                    matrix<fpfmt::e8m7, 64, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            d.data = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(
                false, a.data, false, b.data, c.data, static_cast<int>(a.idx), A_reuse, B_reuse);
        }
        __device__ static void
        wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
             sparse_matrix<fpfmt::e8m10_e8m7x2split, 16, 64, use::A, sparsity::n2of4>& a,
             matrix<fpfmt::e8m10_e8m7x2split, 64, 16, use::B>& b,
             matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            const int idx = static_cast<int>(a.idx);

            d.data = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(
                false, a.small, false, b.big, c.data, idx, 0, 0);
            d.data = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(
                false, a.big, false, b.big, d.data, idx, 0, 1);
            d.data = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(
                false, a.big, false, b.small, d.data, idx, 1, 0);
        }
        __device__ static void wmma(matrix<fpfmt::e8m23, 16, 16, use::Acc>& d,
                                    sparse_matrix<fpfmt::e8m10, 16, 64, use::A, sparsity::n2of4>& a,
                                    matrix<fpfmt::e8m10, 64, 16, use::B>& b,
                                    matrix<fpfmt::e8m23, 16, 16, use::Acc>& c)
        {
            sparse_matrix<fpfmt::e8m10_e8m7x2split, 16, 64, use::A, sparsity::n2of4> a_split;
            matrix<fpfmt::e8m10_e8m7x2split, 64, 16, use::B> b_split;
            matrix_cast(a_split, a);
            matrix_cast(b_split, b);
            wmma(d, a_split, b_split, c);
        }
    };

    template <int BytesPerLane>
    struct global_load_async_to_lds
    {
        __device__ static void load(void* global_ptr, void* lds_ptr)
        {
            if constexpr(BytesPerLane == 16)
            {
                using type = __attribute__((ext_vector_type(4))) int32_t;
                __builtin_amdgcn_global_load_async_to_lds_b128(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 8)
            {
                using type = __attribute__((ext_vector_type(2))) int32_t;
                __builtin_amdgcn_global_load_async_to_lds_b64(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 4)
            {
                using type = int32_t;
                __builtin_amdgcn_global_load_async_to_lds_b32(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 1)
            {
                using type = char;
                __builtin_amdgcn_global_load_async_to_lds_b8(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else
            {
                static_assert(BytesPerLane == 1 || BytesPerLane == 4 || BytesPerLane == 8 ||
                                  BytesPerLane == 16,
                              "BytesPerLane must be 1, 4, 8 or 16");
            }
        }
    };

    template <int BytesPerLane>
    struct global_store_async_from_lds
    {
        __device__ static void store(void* global_ptr, void* lds_ptr)
        {
            if constexpr(BytesPerLane == 16)
            {
                using type = __attribute__((ext_vector_type(4))) int32_t;
                __builtin_amdgcn_global_store_async_from_lds_b128(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 8)
            {
                using type = __attribute__((ext_vector_type(2))) int32_t;
                __builtin_amdgcn_global_store_async_from_lds_b64(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 4)
            {
                using type = int32_t;
                __builtin_amdgcn_global_store_async_from_lds_b32(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else if constexpr(BytesPerLane == 1)
            {
                using type = char;
                __builtin_amdgcn_global_store_async_from_lds_b8(
                    reinterpret_cast<type*>(global_ptr), reinterpret_cast<type*>(lds_ptr), 0, 0);
            }
            else
            {
                static_assert(BytesPerLane == 1 || BytesPerLane == 4 || BytesPerLane == 8 ||
                                  BytesPerLane == 16,
                              "BytesPerLane must be 1, 4, 8 or 16");
            }
        }
    };

    template <typename T>
    __device__ static auto make_buffer(T* global_ptr, int64_t global_size) -> buffer_t
    {
        return __builtin_amdgcn_make_buffer_rsrc(
            const_cast<std::remove_const_t<T>*>(global_ptr), 0, global_size * sizeof(T), 0);
    }

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
                static_assert(BytesPerLane == 1 || BytesPerLane == 2 || BytesPerLane == 4 ||
                                  BytesPerLane == 8 || BytesPerLane == 12 || BytesPerLane == 16,
                              "BytesPerLane must be 1, 2, 4, 8, 12, or 16.");
            }
        }
    };

    // Symmetric to buffer_store: a raw buffer (V#) load that dispatches on the
    // per-lane byte width to the matching raw_buffer_load_bXXX builtin and writes
    // the result into `dest`. Buffer addressing gives free hardware bounds
    // checking (OOB lanes read 0) vs a flat global_load.
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
                static_assert(BytesPerLane == 1 || BytesPerLane == 2 || BytesPerLane == 4 ||
                                  BytesPerLane == 8 || BytesPerLane == 12 || BytesPerLane == 16,
                              "BytesPerLane must be 1, 2, 4, 8, 12, or 16.");
            }
        }
    };

    // A plain pointer-dereference load: address-space-agnostic, so the same
    // instruction serves both LDS and global sources (the pointer's address space
    // decides which). Exposed under both `ds_load*` and `global_load*` aliases so
    // call sites read unambiguously regardless of where the data lives.
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

    // ds_load_tr16_b128 is a pure 128-bit transpose shuffle on 16-bit lanes; the
    // element semantics (fp16/bf16/short) are irrelevant to the instruction, so a
    // single dtype-agnostic _v8i16 load + bit-cast at the call site replaces the
    // per-dtype builtin dispatch (and sidesteps the __fp16 / _Float16 divergence).
    struct ds_load_tr16_b128
    {
        using type                         = __attribute__((ext_vector_type(8))) short;
        static constexpr int bits_per_load = 128;
        inline __device__ static auto
        map(int lane, int item, int bits_per_item) -> std::array<int, 2>
        {
            const auto num_items = bits_per_load / bits_per_item;
            const auto item0     = item / num_items * num_items;
            item                 = item % num_items;
            return {lane / 8 * 8 ^ item, lane % 8 ^ item0};
        }
        inline __device__ static void load(void* lds_ptr, void* dest)
        {
            *reinterpret_cast<type*>(dest) =
                __builtin_amdgcn_ds_load_tr16_b128_v8i16(reinterpret_cast<type*>(lds_ptr));
        }
    };

    // global_load_tr16_b128 is the global-memory sibling of ds_load_tr16_b128:
    // the same 128-bit / 16-lane transpose shuffle, but the source operand is
    // read straight from DRAM (address_space(1)) instead of LDS. Lane/item
    // mapping is identical, so it composes with the same RegTile::matrix::map
    // path. Used by dgrad to load the column-major W^T operand directly from
    // global, skipping the LDS staging + cooperative transpose.
    struct global_load_tr16_b128
    {
        using type                         = __attribute__((ext_vector_type(8))) short;
        static constexpr int bits_per_load = 128;
        inline __device__ static auto
        map(int lane, int item, int bits_per_item) -> std::array<int, 2>
        {
            const auto num_items = bits_per_load / bits_per_item;
            const auto item0     = item / num_items * num_items;
            item                 = item % num_items;
            return {lane / 8 * 8 ^ item, lane % 8 ^ item0};
        }
        inline __device__ static void load(void* gbl_ptr, void* dest)
        {
            using gbl_i16x8_t =
                __attribute__((ext_vector_type(8))) short __attribute__((address_space(1)));
            *reinterpret_cast<type*>(dest) =
                __builtin_amdgcn_global_load_tr16_b128_v8i16((gbl_i16x8_t*)gbl_ptr);
        }
    };

    template <int BytesPerLane>
    struct ds_store
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
            if constexpr(BytesPerLane <= 8)
            {
                // Insert s_nop here to prevent generation of ds_store_2addr_b32
                // that seems to be buggy in AM??
                asm volatile("s_nop 0");
            }
        }
    };
    using ds_store_b8   = ds_store<1>;
    using ds_store_b16  = ds_store<2>;
    using ds_store_b32  = ds_store<4>;
    using ds_store_b64  = ds_store<8>;
    using ds_store_b96  = ds_store<12>;
    using ds_store_b128 = ds_store<16>;

    template <int BytesPerLane>
    struct global_store
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
    using global_store_b32  = global_store<4>;
    using global_store_b64  = global_store<8>;
    using global_store_b96  = global_store<12>;
    using global_store_b128 = global_store<16>;

    union __attribute__((__packed__)) tdm_group0
    {
        struct
        {
            // s0
            struct
            {
                uint32_t count : 2             = 1; // 0: NULL tensor, 1: valid tensor
                uint32_t is_restore : 1        = 0; // Must be 0
                uint32_t is_store : 1          = 0; // Must be 0
                uint32_t nv : 1                = 0; // Must be 0
                uint32_t scope : 2             = 0; // Must be 0
                uint32_t th : 3                = 0; // Must be 0
                uint32_t user_null : 1         = 0; // Should be 0
                uint32_t reserved0 : 19        = 0; // Should be 0
                uint32_t gather_index_size : 1 = 0; // 0: 16-bit indicies, 1: 32-bit indices
                uint32_t gather_mode : 1       = 0; // 0: disabled, 1: enabled
            };
            // s1
            uint32_t lds_addr = 0; // LDS address
            // s2
            uint32_t global_addr_lo = 0; // lower 32 bit of global address
            // s3
            struct
            {
                uint32_t global_addr_hi : 25 = 0; // upper 32
                uint32_t reserved1 : 5       = 0; // Should be 0
                uint32_t type : 2            = 2; // Must be 2
            };
        };
        uint32x4 data;

        template <typename T>
        __device__ inline void set_lds_addr(T* ptr)
        {
            lds_addr = std::bit_cast<uintptr_t>(ptr);
        }

        __device__ inline auto get_global_addr()
        {
            return static_cast<uintptr_t>(global_addr_lo) |
                   (static_cast<uintptr_t>(global_addr_hi) << 32);
        }

        __device__ inline void set_global_addr(uintptr_t ptr_u)
        {
            global_addr_lo = ptr_u;
            global_addr_hi = ptr_u >> 32;
        }

        template <typename T>
        __device__ inline void set_global_addr(T* ptr)
        {
            set_global_addr(std::bit_cast<uintptr_t>(ptr));
        }
    };
    union __attribute__((__packed__)) tdm_group1
    {
        struct
        {
            // s0
            struct
            {
                uint32_t workgroup_mask : 16 = 0; // Workgroup mask used for multi-cast
                uint32_t data_size : 2 =
                    0; // data_size_in_bytes=2**data_size, i.e. 0: 1B, 1: 2B, 2: 4B, 3: 8B
                uint32_t atomic_barrier_enable : 1 =
                    0;                           // 0: disabled, 1: enabled - send atomic barrier
                uint32_t iterate_enable : 1 = 0; // 0: disabled, 1: tensor iteration enabled
                uint32_t pad_enable : 1     = 0; // 0: disabled, 1: padding enabled
                uint32_t early_timeout : 1  = 0; // 0: disabled, 1: multicast immediate timeout
                uint32_t pad_interval : 3   = 0; // pad_interval_in_dwords=2**(pad_interval+1), i.e.
                                                 // 0: 2 dword, 1: 4 dword, ...
                uint32_t pad_amount : 7 = 0; // pad_amount_in_dwords=pad_amount+1, i.e. 0: 1 dword,
                                             // 1: 2 dword, 2: 3 dword, ...
            };
            // s1
            struct
            {
                uint32_t atomic_barrier_address : 16 =
                    0; // Bits 3:18 of LDS barrier address (64-bit aligned)
                uint32_t tensor_dim0_lo : 16 = 0; // Tensor shape[0], lower 16 bits
            };
            // s2
            struct
            {
                uint32_t tensor_dim0_hi : 16 = 0; // Tensor shape[1], upper 16 bits
                uint32_t tensor_dim1_lo : 16 = 0; // Tensor shape[1], lower 16 bits
            };
            // s3
            struct
            {
                uint32_t tensor_dim1_hi : 16 = 0; // Tensor shape[1], upper 16 bits
                uint32_t tile_dim0 : 16      = 0; // Tile shape[0]
            };
            // s4
            struct
            {
                uint32_t tile_dim1 : 16 = 0; // Tile shape[1]
                uint32_t tile_dim2 : 16 = 0; // Tile shape[2]
            };
            // s5
            uint32_t tensor_stride1_lo = 0; // Stride of first mode, lower 32 bits
            // s6
            struct
            {
                uint32_t tensor_stride1_hi : 16 = 0; // Stride of first mode, upper 16 bits
                uint32_t tensor_stride2_lo : 16 = 0; // Stride of second mode, lower 16 bits
            };
            // s7
            uint32_t tensor_stride2_hi = 0; // Stride of second mode, upper 32 bits
        };
        uint32x8 data;

        __device__ inline auto get_tensor_dim0() const
        {
            return tensor_dim0_lo | (tensor_dim0_hi << 16);
        }
        __device__ inline auto get_tensor_dim1() const
        {
            return tensor_dim1_lo | (tensor_dim1_hi << 16);
        }

        __device__ inline void set_tensor_dim0(uint32_t dim0)
        {
            tensor_dim0_lo = dim0;
            tensor_dim0_hi = dim0 >> 16;
        }

        __device__ inline void set_tensor_dim1(uint32_t dim1)
        {
            tensor_dim1_lo = dim1;
            tensor_dim1_hi = dim1 >> 16;
        }

        __device__ inline auto get_tensor_stride1() const
        {
            return tensor_stride1_lo | (static_cast<uint64_t>(tensor_stride1_hi) << 32);
        }
        __device__ inline auto get_tensor_stride2() const
        {
            return tensor_stride2_lo | (static_cast<uint64_t>(tensor_stride2_hi) << 16);
        }

        __device__ inline void set_tensor_stride1(uint64_t stride)
        {
            tensor_stride1_lo = stride;
            tensor_stride1_hi = stride >> 32;
        }

        __device__ inline void set_tensor_stride2(uint64_t stride)
        {
            tensor_stride2_lo = stride;
            tensor_stride2_hi = stride >> 16;
        }
    };

    __device__ inline static void tdm_mode0_step(tdm_group0& d0, tdm_group1& d1, uint32_t by)
    {
        uintptr_t addr = d0.get_global_addr();
        addr += by << d1.data_size;
        d0.set_global_addr(addr);
        d1.set_tensor_dim0(d1.get_tensor_dim0() - by);
    }

    template <uint32_t DataSize, uint32_t TileDim0, uint32_t TileStride0>
    __device__ inline static void configure_padding(tdm_group1& d1)
    {
        constexpr uint32_t tile_dim0_dword = TileDim0 * DataSize / 4;
        static_assert(tile_dim0_dword * 4 == TileDim0 * DataSize,
                      "TileDim0 times DataSize must be divisible by 4");
        static_assert(tile_dim0_dword >= 2, "Pad interval must be at least 2 DWORDS");
        static_assert(tile_dim0_dword <= 256, "Pad interval must be at most 256 DWORDS");
        static_assert(is_power_of_two(tile_dim0_dword), "Pad interval must be a power of two");

        constexpr uint32_t tile_stride0_dword = TileStride0 * DataSize / 4;
        static_assert(tile_stride0_dword * 4 == TileStride0 * DataSize,
                      "TileStride0 times DataSize must be divisible by 4");
        static_assert(tile_stride0_dword >= tile_dim0_dword,
                      "TileStride0 must be greater than or equal to TileDim0");

        constexpr uint32_t pad_amount = tile_stride0_dword - tile_dim0_dword;
        static_assert(pad_amount <= 128, "Pad amount must be at most 128 DWORDS");

        if(pad_amount > 0)
        {
            d1.pad_enable   = 1;
            d1.pad_interval = ilog2(tile_dim0_dword) - 1;
            d1.pad_amount   = pad_amount - 1;
        }
        else
        {
            d1.pad_enable = 0;
        }
    }

    union __attribute__((__packed__)) tdm_group2
    {
        struct
        {
            // s0
            uint32_t tensor_dim2 = 0; // Tensor shape[2]
                                      // s1
            uint32_t tensor_dim3 = 0; // Tensor shape[3] or LDS addr increment
            // s2
            uint32_t tensor_stride3_lo = 0; // Stride of third mode, lower 32 bits, or global addr
                                            // increment s3
            struct
            {
                uint32_t tensor_stride3_hi : 16 =
                    0; // Stride of third mode, upper 16 bits, or global addr increment
                uint32_t tile_dim3 : 16 = 0; // Tile shape[3] or iterate count
            };
        };
        uint32x4 data;

        __device__ inline auto get_tensor_dim2() const { return tensor_dim2; }
        __device__ inline auto get_tensor_dim3() const { return tensor_dim3; }
        __device__ inline void set_tensor_dim2(uint32_t dim2) { tensor_dim2 = dim2; }
        __device__ inline void set_tensor_dim3(uint32_t dim3) { tensor_dim3 = dim3; }

        __device__ inline auto get_tensor_stride3() const
        {
            return tensor_stride3_lo | (static_cast<uint64_t>(tensor_stride3_hi) << 32);
        }
        __device__ inline void set_tensor_stride3(uint64_t stride)
        {
            tensor_stride3_lo = stride;
            tensor_stride3_hi = stride >> 32;
        }
    };

    union __attribute__((__packed__)) tdm_group3
    {
        struct
        {
            // s0
            uint32_t tensor_stride4_lo = 0; // Stride of fourth mode, lower 32 bits
            // s1
            struct
            {
                uint32_t tensor_stride4_hi : 16 = 0; // Stride of fourth mode, upper 16 bits
                uint32_t tensor_dim4_lo : 16    = 0; // Tensor shape[4], lower 16 bits
            };
            // s2
            struct
            {
                uint32_t tensor_dim4_hi : 16 = 0; // Tensor shape[4], upper 16 bits
                uint32_t tile_dim4 : 16      = 0; // Tile shape[4]
            };
            // s3
            uint32_t reserved0 = 0;
        };
        uint32x4 data;

        __device__ inline auto get_tensor_dim4() const
        {
            return tensor_dim4_lo | (tensor_dim4_hi << 16);
        }
        __device__ inline void set_tensor_dim4(uint32_t dim4)
        {
            tensor_dim4_lo = dim4;
            tensor_dim4_hi = dim4 >> 16;
        }

        __device__ inline auto get_tensor_stride4() const
        {
            return tensor_stride4_lo | (static_cast<uint64_t>(tensor_stride4_hi) << 32);
        }
        __device__ inline void set_tensor_stride4(uint64_t stride)
        {
            tensor_stride4_lo = stride;
            tensor_stride4_hi = stride >> 32;
        }
    };

    union __attribute__((__packed__)) tdm_group4
    {
        uint32x8 data = 0;
    };

    enum class scope : int
    {
        WGP = 0,
        SE  = 1,
        DEV = 2,
        SYS = 3
    };
    enum class th : int
    {
        RT    = 0,
        NT    = 1,
        HT    = 2,
        LU    = 3,
        NT_RT = 4,
        RT_NT = 5,
        NT_HT = 6,
    };
    template <scope S, th T, bool Speculative>
    __device__ constexpr static auto make_prefetch_flags() -> int
    {
        static_assert(T != th::NT, "NT is unsupported for prefetch");
        static_assert(T != th::LU, "LU is unsupported for prefetch");
        static_assert(!Speculative || S != scope::WGP,
                      "prefetch to WGP scope is always non-speculative");
        static_assert(Speculative || (T != th::NT_RT && T != th::RT_NT && T != th::NT_HT),
                      "NT_RT, RT_NT, NT_HT are always speculative");
        th t = T;
        if(!Speculative)
        {
            // Cf. section 4.2.1 in Shader Programming Guide
            switch(T)
            {
            case th::RT:
                t = th::NT;
                break;
            case th::HT:
                t = th::LU;
                break;
            default:
                break;
            }
        }
        return (static_cast<int>(S) << 3) | static_cast<int>(t);
    }

    template <int Flags>
    struct prefetch
    {
        static constexpr int cacheline_size       = 256;
        static constexpr int cachelines_per_round = wave_size;
        template <int IOffset>
        inline __device__ static void fetch(uint8_t const* ptr, int64_t v_offset)
        {
            __builtin_amdgcn_global_prefetch(ptr + v_offset + IOffset, Flags);
        }
    };

    // Use the abomination below if you want to enforce GVS addressing
    //
    // Note: v_offset is int32_t and the caller must ensure that (wave_size-1) * stride does not
    // overflow
    template <int Flags>
    struct prefetch_gvs
    {
        static constexpr int cacheline_size       = 256;
        static constexpr int cachelines_per_round = wave_size;
        template <int IOffset>
        inline __device__ static void fetch(uint8_t const* ptr, int32_t v_offset)
        {
#define PF_CASE(SCOPE, TH)                                                                        \
    else if constexpr((Flags & 0x7) == static_cast<int>(th::TH) &&                                \
                      (Flags >> 3) == static_cast<int>(scope::SCOPE))                             \
    {                                                                                             \
        asm volatile("global_prefetch_b8 %0, %1 offset:%2 scope:SCOPE_" #SCOPE " th:TH_LOAD_" #TH \
                     :                                                                            \
                     : "v"(v_offset), "s"(ptr), "i"(IOffset));                                    \
    }
#define PF_CASE_ALL_SCOPES(TH) \
    PF_CASE(WGP, TH)           \
    PF_CASE(SE, TH)            \
    PF_CASE(DEV, TH)           \
    PF_CASE(SYS, TH)

            if constexpr(false) {}
            PF_CASE_ALL_SCOPES(RT)
            PF_CASE_ALL_SCOPES(NT)
            PF_CASE_ALL_SCOPES(HT)
            PF_CASE_ALL_SCOPES(LU)
            PF_CASE_ALL_SCOPES(NT_RT)
            PF_CASE_ALL_SCOPES(RT_NT)
            PF_CASE_ALL_SCOPES(NT_HT)
            else
            {
                static_assert(Flags == 0, "Unsupported prefetch flags");
            }

#undef PF_CASE_ALL_SCOPES
#undef PF_CASE
        }
    };
};

// Non-returning, device-scope *cascading* fp32 global atomic add (gfx1250).
// TH_ATOMIC_CASCADE_RT defers the device-scope realization: a cache below DEV
// scope (WGP$/L2$) initializes a line to the first add's operand on a miss and
// accumulates subsequent adds locally; the accumulated partial is pushed up via
// an add-on-eviction, and the device-scope writeback at kernel end resolves it
// into the target. This turns a cross-workgroup reduction into cache-local
// accumulation, shedding most of the global atomic-RMW traffic the plain
// atomicAdd epilogue pays. The destination must still be pre-zeroed (the final
// writeback adds the accumulated value onto memory). No clang builtin exposes
// the cascade TH bit, so emit the instruction directly.
__device__ __forceinline__ void cascade_atomic_add_f32(float* addr, float val)
{
    asm volatile("global_atomic_add_f32 %0, %1, off th:TH_ATOMIC_CASCADE_RT scope:SCOPE_DEV"
                 :
                 : "v"(addr), "v"(val)
                 : "memory");
}

} // namespace bunnies

extern "C" __device__ void llvm_amdgcn_s_wait_dscnt(uint16_t count) asm("llvm.amdgcn.s.wait.dscnt");

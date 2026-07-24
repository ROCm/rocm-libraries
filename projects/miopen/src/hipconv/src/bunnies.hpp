#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace bunnies
{

using int16x4   = __attribute__((ext_vector_type(4))) int16_t;
using int32x2   = __attribute__((ext_vector_type(2))) int32_t;
using uint32x2  = __attribute__((ext_vector_type(2))) uint32_t;
using uint32x3  = __attribute__((ext_vector_type(3))) uint32_t;
using uint32x4  = __attribute__((ext_vector_type(4))) uint32_t;
using uint32x8  = __attribute__((ext_vector_type(8))) uint32_t;
using uint32x16 = __attribute__((ext_vector_type(16))) uint32_t;
using floatx4   = __attribute__((ext_vector_type(4))) float;
using floatx8   = __attribute__((ext_vector_type(8))) float;
using fp16x2    = __attribute__((ext_vector_type(2))) _Float16;
using bf16x2    = __attribute__((ext_vector_type(2))) __bf16;
using fp16x4    = __attribute__((ext_vector_type(4))) _Float16;
using bf16x4    = __attribute__((ext_vector_type(4))) __bf16;
using fp16x8    = __attribute__((ext_vector_type(8))) _Float16;
using bf16x8    = __attribute__((ext_vector_type(8))) __bf16;
using fp16x16   = __attribute__((ext_vector_type(16))) _Float16;
using bf16x16   = __attribute__((ext_vector_type(16))) __bf16;

template <int BytesPerLane>
using packed_type = std::conditional_t<
    BytesPerLane == 16,
    uint32x4,
    std::conditional_t<
        BytesPerLane == 12,
        uint32x3,
        std::conditional_t<
            BytesPerLane == 8,
            uint32x2,
            std::conditional_t<BytesPerLane == 4,
                               uint32_t,
                               std::conditional_t<BytesPerLane == 2, uint16_t, uint8_t>>>>>;

enum class use
{
    A,
    B,
    Acc
};

enum class wmma_flag : uint32_t
{
    A_reuse = 1,
    B_reuse = 2
};
constexpr auto test(uint32_t flags, wmma_flag flag) -> bool
{
    return flags & static_cast<int>(flag);
}

enum class fpfmt : int
{
    e11m52 = 0x400, // fp64
    e8m23  = 0x200, // fp32
    e5m10  = 0x100, // fp16
    e8m7   = 0x101, // bf16
    e4m3   = 0x80,  // fp8
    e5m2   = 0x81,  // bf8
    e2m3   = 0x60,  // fp6
    e3m2   = 0x61,  // bf6
    e2m1   = 0x40,  // fp4
    // scale layouts (unsigned)
    ue8m0 = 0x1080,
};
constexpr auto bits_per_item(fpfmt f)
{
    return (static_cast<int>(f) >> 4) & 0xFF;
}
constexpr auto is_scale_fmt(fpfmt f)
{
    return (static_cast<int>(f) & 0x1000) != 0;
}

// clang-format off
template <fpfmt Fmt, int Bits> constexpr bool is_wide = bits_per_item(Fmt) == Bits;
template <fpfmt Fmt, int Bits> constexpr bool is_wide_notscale = is_wide<Fmt,Bits> && !is_scale_fmt(Fmt);
template <fpfmt Fmt, int Bits> constexpr bool is_wide_scale = is_wide<Fmt,Bits> && is_scale_fmt(Fmt);
template <fpfmt Fmt> constexpr bool is_64bit = is_wide_notscale<Fmt, 64>;
template <fpfmt Fmt> constexpr bool is_32bit = is_wide_notscale<Fmt, 32>;
template <fpfmt Fmt> constexpr bool is_16bit = is_wide_notscale<Fmt, 16>;
template <fpfmt Fmt> constexpr bool is_8bit = is_wide_notscale<Fmt, 8>;
template <fpfmt Fmt> constexpr bool is_6bit = is_wide_notscale<Fmt, 6>;
template <fpfmt Fmt> constexpr bool is_4bit = is_wide_notscale<Fmt, 4>;
template <fpfmt Fmt> constexpr bool is_8bit_scale = is_wide_scale<Fmt, 8>;
// clang-format on

// clang-format off
template <fpfmt Fmt> struct base_storage_type { using type = uint32_t; };
template <> struct base_storage_type<fpfmt::e11m52> { using type = double; };
template <> struct base_storage_type<fpfmt::e8m23> { using type = float; };
template <> struct base_storage_type<fpfmt::e5m10> { using type = _Float16; };
template <> struct base_storage_type<fpfmt::e8m7> { using type = __bf16; };
template <> struct base_storage_type<fpfmt::ue8m0> { using type = int; };
// clang-format on
template <fpfmt Fmt>
using base_storage_type_t = typename base_storage_type<Fmt>::type;
template <fpfmt Fmt, int NumItems>
constexpr int storage_vector_length =
    NumItems * bits_per_item(Fmt) / (8 * sizeof(base_storage_type_t<Fmt>));
template <fpfmt Fmt, int NumItems>
using storage_type_t =
    __attribute__((ext_vector_type(storage_vector_length<Fmt, NumItems>))) base_storage_type_t<Fmt>;

template <std::integral T>
constexpr auto is_power_of_two(T x) -> bool
{
    return x > 0 && ((x & (x - 1)) == 0);
}

template <std::integral T>
constexpr auto ilog2(T x) -> T
{
    T l = 0;
    while(x > 1)
    {
        ++l;
        x >>= 1;
    }
    return l;
}

template <typename F, int... Is>
__device__ constexpr void static_unroll(std::integer_sequence<int, Is...> trips, F&& f)
{
    (f(std::integral_constant<int, Is>()), ...);
}

template <int TripCount, typename F>
__device__ constexpr void static_unroll(F&& f)
{
    static_unroll(std::make_integer_sequence<int, TripCount>{}, std::forward<F>(f));
}

///////////////////////////
////// Architecture ///////
///////////////////////////

template <typename T>
concept device_arch = requires(T t)
{
    {T::wave_size};
};

///////////////////////////
////// Register tile //////
///////////////////////////

template <typename T>
concept reg_tile_concept = requires(T & t, int mb, int nb)
{
    typename T::matrix;
    {
        T::row_blocks
    } -> std::same_as<const int&>;
    {
        T::col_blocks
    } -> std::same_as<const int&>;
    {
        t.block(mb, nb)
    } -> std::same_as<typename T::matrix&>;
};

template <typename Matrix, int RowBlocks, int ColBlocks, int Stride, bool Owner = false>
struct reg_tile_base
{
    using matrix = Matrix;
    using container_t =
        std::conditional_t<Owner, std::array<Matrix, RowBlocks * ColBlocks>, Matrix*>;
    static constexpr int row_blocks = RowBlocks;
    static constexpr int col_blocks = ColBlocks;
    static constexpr int stride     = Stride;

    container_t blocks;

    inline __device__ auto block(int mb, int nb) -> matrix& { return blocks[mb * stride + nb]; }
    inline __device__ auto block(int mb, int nb) const -> matrix const&
    {
        return blocks[mb * stride + nb];
    }

    template <int SubRowBlocks, int SubColBlocks>
    inline __device__ auto sub(int mb0,
                               int nb0) -> reg_tile_base<Matrix, SubRowBlocks, SubColBlocks, Stride>
    {
        static_assert(SubRowBlocks <= RowBlocks);
        static_assert(SubColBlocks <= ColBlocks);
        return {&block(mb0, nb0)};
    }
};

template <typename Matrix, int RowBlocks, int ColBlocks>
struct reg_tile : reg_tile_base<Matrix, RowBlocks, ColBlocks, ColBlocks, true>
{
};

template <typename Dest, typename Src>
inline __device__ void tile_cast(Dest& dest, Src const& src)
{
    using arch = typename Dest::matrix::arch;
    static_assert(std::is_same_v<arch, typename Src::matrix::arch>);
    static_assert(Dest::row_blocks == Src::row_blocks);
    static_assert(Dest::col_blocks == Src::col_blocks);
    for(int mb = 0; mb < Dest::row_blocks; ++mb)
    {
        for(int nb = 0; nb < Dest::col_blocks; ++nb)
        {
            arch::matrix_cast(dest.block(mb, nb), src.block(mb, nb));
        }
    }
}

///////////////////////////
///////// Memref //////////
///////////////////////////

template <typename IdxT = int>
struct slice
{
    IdxT offset = 0, size = 0;
};

namespace detail
{
template <typename IdxT>
__device__ auto offset(IdxT i)
{
    return i;
}
template <typename IdxT>
__device__ auto offset(slice<IdxT> i)
{
    return i.offset;
}
template <typename IdxT>
__device__ auto size(IdxT i)
{
    return 0;
}
template <typename IdxT>
__device__ auto size(slice<IdxT> i)
{
    return i.size;
}
} // namespace detail

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
struct tensor_view
{
    static constexpr int dim = Dim;
    using tuple_t            = std::array<IdxT, Dim>;

    OffsetT offset;
    std::array<IdxT, Dim> shape, stride;

    __device__ auto delta(std::array<IdxT, Dim> const& idx) const -> IdxT
    {
        IdxT p = 0;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            p += idx[i] * stride[i];
        }
        return p;
    }
    template <std::integral... I>
    __device__ auto delta(I... idx) const -> IdxT
    {
        static_assert(sizeof...(I) == Dim);

        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return delta(offsets);
    }
    __device__ auto operator()(std::array<IdxT, Dim> const& idx) const -> OffsetT
    {
        return offset + delta(idx);
    }
    template <std::integral... I>
    __device__ auto operator()(I... idx) const -> OffsetT
    {
        return offset + delta(std::forward<I>(idx)...);
    }

    // Checks whether a multi-index is within bounds; does not check whether index is negative
    __device__ auto in_bounds(std::array<IdxT, Dim> const& idx) const -> bool
    {
        bool ok = true;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            ok = ok && idx[i] < shape[i];
        }
        return ok;
    }
    template <std::integral... I>
    __device__ auto in_bounds(I... idx) const -> bool
    {
        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return in_bounds(offsets);
    }
    // Checks whether a multi-index is within bounds; checks that indices are non-negative
    __device__ auto in_bounds_maybe_negative(std::array<IdxT, Dim> const& idx) const -> bool
    {
        bool ok = true;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            ok = ok && idx[i] >= 0 && idx[i] < shape[i];
        }
        return ok;
    }
    template <std::integral... I>
    __device__ auto in_bounds_maybe_negative(I... idx) const -> bool
    {
        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return in_bounds_maybe_negative(offsets);
    }

    template <typename... I>
    __device__ auto subview(I&&... idx_or_slice) const
    {
        static_assert(sizeof...(I) == Dim);
        static_assert(((std::is_same_v<std::decay_t<I>, IdxT> ||
                        std::is_same_v<std::decay_t<I>, slice<IdxT>>) &&
                       ...));

        constexpr int SubDim = (static_cast<int>(std::is_same_v<I, slice<IdxT>>) + ...);

        std::array<IdxT, Dim> offsets  = {detail::offset(idx_or_slice)...};
        std::array<bool, Dim> is_slice = {std::is_same_v<I, slice<IdxT>>...};

        OffsetT suboffset = offset;
        std::array<IdxT, SubDim> subshape, substride;
        int j = 0;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            suboffset += offsets[i] * stride[i];
            if(is_slice[i])
            {
                subshape[j]  = shape[i];
                substride[j] = stride[i];
                ++j;
            }
        }
        return tensor_view<SubDim, IdxT, OffsetT>(suboffset, subshape, substride);
    }
};

template <int Dim, typename T, typename IdxT = int>
using memref = tensor_view<Dim, IdxT, T*>;

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto
make_view(OffsetT offset, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return tensor_view<Dim, IdxT, OffsetT>{offset, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_col_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[0] = 1;
    for(int mode = 0; mode < Dim - 1; ++mode)
    {
        stride[mode + 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_row_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[Dim - 1] = 1;
    for(int mode = Dim - 1; mode > 0; --mode)
    {
        stride[mode - 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto
make_memref(T* ptr, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return make_view<Dim, IdxT, T*>(ptr, shape, stride);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_col_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_col_major<Dim, IdxT, T*>(shape);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_row_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_row_major<Dim, IdxT, T*>(shape);
}

///////////////////////////
///////// Actions /////////
///////////////////////////
//
__device__ __forceinline__ auto lane_id() -> int
{
    return threadIdx.x % warpSize;
}

__device__ __forceinline__ auto wave_id() -> int
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / warpSize);
}

template <int num_rounds, int num_waves, typename F>
__device__ void wave_distribute(int wave, F&& f)
{
    constexpr int num_rounds_per_wave = num_rounds / num_waves;
    constexpr int remainder_rounds    = num_rounds - num_rounds_per_wave * num_waves;
#pragma unroll
    for(int i = 0; i < num_rounds_per_wave; ++i)
    {
        f(i + num_rounds_per_wave * wave);
    }
    if constexpr(remainder_rounds > 0)
    {
        if(wave < remainder_rounds)
        {
            f(num_rounds_per_wave * num_waves + wave);
        }
    }
}

struct buffer_load_to_lds_config
{
    int rows;
    int cols;
    int bytes_per_lane;
    int num_waves;

    template <device_arch Arch, typename T>
    __device__ constexpr auto num_rounds() const
    {
        return 1 + (rows * cols * sizeof(T) - 1) / (Arch::wave_size * bytes_per_lane);
    }
    template <device_arch Arch, typename T>
    __device__ constexpr auto min_loads_per_wave() const
    {
        return num_rounds<Arch, T>() / num_waves;
    }
    template <device_arch Arch, typename T>
    __device__ constexpr auto max_loads_per_wave() const
    {
        return 1 + (num_rounds<Arch, T>() - 1) / num_waves;
    }
};
template <device_arch Arch, buffer_load_to_lds_config cfg, typename T, typename SwizzleInv>
__device__ __forceinline__ void buffer_load_to_lds(int wave,
                                                   typename Arch::buffer_t global_buffer,
                                                   tensor_view<2> const& global_view,
                                                   std::array<int, 2> global_offset,
                                                   SwizzleInv&& swizzle_inv,
                                                   T* lds_ptr,
                                                   int lds_offset)
{
    using load_inst            = typename Arch::template buffer_load_lds<cfg.bytes_per_lane>;
    const auto lane            = lane_id();
    constexpr int num_rounds   = cfg.num_rounds<Arch, T>();
    constexpr int lane_stride  = cfg.bytes_per_lane / sizeof(T);
    constexpr int round_stride = Arch::wave_size * lane_stride;
    const auto s_offset        = global_view(global_offset[0], global_offset[1]) * sizeof(T);
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        void* lds_dest      = lds_ptr + lds_offset + round * round_stride;
        const auto [mm, kk] = swizzle_inv(round * round_stride + lane * lane_stride);
        const auto v_offset = global_view.delta(mm, kk) * sizeof(T);
        load_inst::load(global_buffer, lds_dest, v_offset, s_offset);
    });
}
template <device_arch Arch,
          buffer_load_to_lds_config cfg,
          typename VSOffsetMap,
          typename T,
          typename SwizzleInv>
__device__ __forceinline__ void buffer_load_to_lds(int wave,
                                                   typename Arch::buffer_t global_buffer,
                                                   VSOffsetMap&& vsoffset,
                                                   SwizzleInv&& swizzle_inv,
                                                   T* lds_ptr,
                                                   int lds_offset)
{
    using load_inst          = typename Arch::template buffer_load_lds<cfg.bytes_per_lane>;
    const auto lane          = lane_id();
    constexpr int num_rounds = cfg.num_rounds<Arch, T>();
    const int lane_stride    = cfg.bytes_per_lane / sizeof(T);
    const int round_stride   = Arch::wave_size * lane_stride;
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        void* lds_dest                = lds_ptr + lds_offset + round * round_stride;
        const auto [mm, kk]           = swizzle_inv(round * round_stride + lane * lane_stride);
        const auto [voffset, soffset] = vsoffset(mm, kk);
        load_inst::load(global_buffer, lds_dest, voffset, soffset);
    });
}

template <int BytesPerLane, reg_tile_concept RegTile, typename VOffsetMap>
__device__ __forceinline__ void buffer_store(typename RegTile::matrix::arch::buffer_t global_buffer,
                                             RegTile& rt,
                                             VOffsetMap&& voffset)
{
    using arch                        = typename RegTile::matrix::arch;
    using store_inst                  = typename arch::template buffer_store<BytesPerLane>;
    using type                        = packed_type<BytesPerLane>;
    constexpr int bpi                 = bits_per_item(RegTile::matrix::fmt);
    constexpr int num_rounds          = RegTile::matrix::num_items * bpi / (8 * BytesPerLane);
    constexpr int num_items_per_round = RegTile::matrix::num_items / num_rounds;
    const int lane                    = lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto item0   = rnd * num_items_per_round;
                const auto coord   = RegTile::matrix::map({lane, item0});
                const int v_offset = voffset(mb, nb, coord[0], coord[1]);
                store_inst::store(
                    global_buffer, reinterpret_cast<type*>(&rt.block(mb, nb)) + rnd, v_offset, 0);
            }
        }
    }
}

struct async_config
{
    int rows;
    int cols;
    int bytes_per_lane;
    int num_waves;

    template <device_arch Arch, typename T>
    __device__ constexpr auto num_rounds() const
    {
        return rows * cols * sizeof(T) / (Arch::wave_size * bytes_per_lane);
    }

    template <device_arch Arch, typename T>
    __device__ constexpr auto index(int round, int lane) const -> std::array<int, 2>
    {
        const int items_per_lane  = bytes_per_lane / sizeof(T);
        const int items_per_round = Arch::wave_size * items_per_lane;
        const int lane_offset     = items_per_lane * lane;
        int r, c;
        // The general formula is
        // offset = round * items_per_round + lane_offset;
        // r      = offset / cols;
        // c      = offset % cols;
        //
        // if cols >= items_per_round and cols % items_per_round == 0 then
        // using lane_offset < items_per_round we have
        // r = round * items_per_round / cols + lane_offset / cols
        //   = round / (cols / items_per_round)
        // c = round * items_per_round % cols + lane_offset % cols
        //   = round % (cols / items_per_round) + lane_offset
        if(cols >= items_per_round && cols % items_per_round == 0)
        {
            const int rounds_per_col = cols / items_per_round;
            r                        = round / rounds_per_col;
            c                        = round % rounds_per_col * items_per_round + lane_offset;
        }
        // if items_per_round >= cols and items_per_round % cols == 0 then
        // r = round * items_per_round / cols + lane_offset / cols
        //   = round * (items_per_round / cols) + lane_offset / cols
        // c = round * items_per_round % cols + lane_offset % cols
        //   = lane_offset % cols
        else if(items_per_round >= cols && items_per_round % cols == 0)
        {
            const int cols_per_round = items_per_round / cols;
            r                        = round * cols_per_round + lane_offset / cols;
            c                        = lane_offset % cols;
        }
        else
        {
            const int offset = round * items_per_round + lane_offset;
            r                = offset / cols;
            c                = offset % cols;
        }
        return {r, c};
    }
};
template <device_arch Arch,
          async_config cfg,
          typename VSOffsetMap,
          typename T,
          typename LDSLayoutMap>
__device__ __forceinline__ void buffer_store_from_lds(int wave,
                                                      typename Arch::buffer_t global_buffer,
                                                      VSOffsetMap&& vsoffset,
                                                      T* lds_ptr,
                                                      LDSLayoutMap&& lds_layout)
{
    static_assert(cfg.bytes_per_lane >= sizeof(T) && cfg.bytes_per_lane % sizeof(T) == 0);
    using store_inst         = Arch::template buffer_store<cfg.bytes_per_lane>;
    using type               = packed_type<cfg.bytes_per_lane>;
    constexpr int num_rounds = cfg.num_rounds<Arch, T>();
    const auto lane          = lane_id();
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        const auto [r, c]             = cfg.index<Arch, T>(round, lane);
        type* data                    = reinterpret_cast<type*>(lds_ptr + lds_layout(r, c));
        const auto [voffset, soffset] = vsoffset(r, c);
        store_inst::store(global_buffer, data, voffset, soffset);
    });
}

template <device_arch Arch, async_config cfg, typename T, typename VOffsetMap, typename Swizzle>
__device__ __forceinline__ void global_load_async_to_lds(int wave,
                                                         T* global_ptr,
                                                         VOffsetMap&& voffset,
                                                         T* lds_ptr,
                                                         Swizzle&& swizzle)
{
    static_assert(cfg.bytes_per_lane >= sizeof(T) && cfg.bytes_per_lane % sizeof(T) == 0);
    using load_inst      = typename Arch::template global_load_async_to_lds<cfg.bytes_per_lane>;
    const auto lane      = lane_id();
    const int num_rounds = cfg.num_rounds<Arch, T>();
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        const auto [r, c] = cfg.index<Arch, T>(round, lane);
        load_inst::load(global_ptr + voffset(r, c), lds_ptr + swizzle(r, c));
    });
}

template <device_arch Arch, async_config cfg, typename T, typename VOffsetMap, typename Swizzle>
__device__ __forceinline__ void global_store_async_from_lds(int wave,
                                                            T* global_ptr,
                                                            VOffsetMap&& voffset,
                                                            T* lds_ptr,
                                                            Swizzle&& swizzle)
{
    static_assert(cfg.bytes_per_lane >= sizeof(T) && cfg.bytes_per_lane % sizeof(T) == 0);
    using store_inst     = typename Arch::template global_store_async_from_lds<cfg.bytes_per_lane>;
    const auto lane      = lane_id();
    const int num_rounds = cfg.num_rounds<Arch, T>();
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        const auto [r, c] = cfg.index<Arch, T>(round, lane);
        store_inst::store(global_ptr + voffset(r, c), lds_ptr + swizzle(r, c));
    });
}
template <device_arch Arch,
          async_config cfg,
          typename T,
          typename InBoundsMap,
          typename VOffsetMap,
          typename Swizzle>
__device__ __forceinline__ void global_store_async_from_lds_checked(int wave,
                                                                    T* global_ptr,
                                                                    InBoundsMap&& in_bounds,
                                                                    VOffsetMap&& voffset,
                                                                    T* lds_ptr,
                                                                    Swizzle&& swizzle)
{
    static_assert(cfg.bytes_per_lane >= sizeof(T) && cfg.bytes_per_lane % sizeof(T) == 0);
    using store_inst     = typename Arch::template global_store_async_from_lds<cfg.bytes_per_lane>;
    const auto lane      = lane_id();
    const int num_rounds = cfg.num_rounds<Arch, T>();
    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        const auto [r, c] = cfg.index<Arch, T>(round, lane);
        if(in_bounds(r, c))
        {
            store_inst::store(global_ptr + voffset(r, c), lds_ptr + swizzle(r, c));
        }
    });
}

template <int BytesPerLane, typename T, reg_tile_concept RegTile, typename VOffsetMap>
__device__ __forceinline__ void global_store(T* global_ptr, RegTile& rt, VOffsetMap&& voffset)
{
    using arch                        = typename RegTile::matrix::arch;
    using type                        = packed_type<BytesPerLane>;
    constexpr int bpi                 = bits_per_item(RegTile::matrix::fmt);
    constexpr int num_rounds          = RegTile::matrix::num_items * bpi / (8 * BytesPerLane);
    constexpr int num_items_per_round = RegTile::matrix::num_items / num_rounds;
    const int lane                    = lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto item0   = rnd * num_items_per_round;
                const auto coord   = RegTile::matrix::map({lane, item0});
                const int v_offset = voffset(mb, nb, coord[0], coord[1]);
                *reinterpret_cast<type*>(global_ptr + v_offset) =
                    *(reinterpret_cast<type*>(&rt.block(mb, nb)) + rnd);
            }
        }
    }
}

// Drives a tiled register load: for each block/round it maps the lane's element
// to a (row,col) coord, asks `map` for the source offset, and issues LoadInst.
// Address-space-agnostic (the LoadInst + `base` pointer decide LDS vs global),
// so despite historical usage it is not LDS-specific.
template <typename LoadInst, reg_tile_concept RegTile, typename MemT, typename Map>
__device__ void load_tile(RegTile& rt, MemT* base, Map&& map)
{
    using ld_t                        = typename LoadInst::type;
    constexpr int bpi                 = bits_per_item(RegTile::matrix::fmt);
    constexpr int num_items_per_round = LoadInst::bits_per_load / bpi;
    constexpr int num_rounds          = RegTile::matrix::num_items / num_items_per_round;
    const int lane                    = lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto laneitem = LoadInst::map(lane, rnd * num_items_per_round, bpi);
                const auto coord    = RegTile::matrix::map(laneitem);
                const int offset    = map(mb, nb, coord[0], coord[1]);
                LoadInst::load(base + offset,
                               reinterpret_cast<ld_t*>(&rt.block(mb, nb).data) + rnd);
            }
        }
    }
}

// Like load_tile, but only the rounds whose representative element is non-zero
// (per `nzmap`) are loaded; the rest keep their pre-zeroed register contents.
// This drives structured-sparse (e.g. block-diagonal grouped) operands where a
// lane covers only a contiguous run of a larger tile. Address-space-agnostic:
// `ptr_base` may point to LDS or global, so it serves both `ds_load*` and
// `global_load*` LoadInsts.
template <typename LoadInst,
          reg_tile_concept RegTile,
          typename MemT,
          typename Map,
          typename NonZeroMap>
__device__ void load_sparse(RegTile& rt, MemT* ptr_base, Map&& map, NonZeroMap&& nzmap)
{
    using ld_t                        = typename LoadInst::type;
    constexpr int bpi                 = bits_per_item(RegTile::matrix::fmt);
    constexpr int num_items_per_round = LoadInst::bits_per_load / bpi;
    constexpr int num_rounds          = RegTile::matrix::num_items / num_items_per_round;
    const int lane                    = lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto laneitem = LoadInst::map(lane, rnd * num_items_per_round, bpi);
                const auto coord    = RegTile::matrix::map(laneitem);
                if(nzmap(mb, nb, coord[0], coord[1]))
                {
                    const int offset = map(mb, nb, coord[0], coord[1]);
                    LoadInst::load(ptr_base + offset,
                                   reinterpret_cast<ld_t*>(&rt.block(mb, nb).data) + rnd);
                }
            }
        }
    }
}

template <typename StoreInst, reg_tile_concept RegTile, typename MemT, typename Map>
__device__ void store_tile(RegTile& rt, MemT* base, Map&& map)
{
    using st_t                        = typename StoreInst::type;
    constexpr int bpi                 = bits_per_item(RegTile::matrix::fmt);
    constexpr int num_items_per_round = StoreInst::bits_per_store / bpi;
    constexpr int num_rounds          = RegTile::matrix::num_items / num_items_per_round;
    const int lane                    = lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto laneitem = StoreInst::map(lane, rnd * num_items_per_round, bpi);
                const auto coord    = RegTile::matrix::map(laneitem);
                const int offset    = map(mb, nb, coord[0], coord[1]);
                StoreInst::store(base + offset,
                                 reinterpret_cast<st_t*>(&rt.block(mb, nb).data) + rnd);
            }
        }
    }
}

// Prefetch MxN matrix stored in row-major format, i.e. address calculation is m * stride + n
struct prefetch_config
{
    int rows;
    int cols;
    int num_waves;
};
template <typename PrefetchInst, prefetch_config cfg, typename T>
__device__ static void prefetch_matrix(int wave, T const* base, int64_t stride)
{
    constexpr int cacheline_size       = PrefetchInst::cacheline_size;
    constexpr int cl_per_round         = PrefetchInst::cachelines_per_round;
    constexpr int cl_per_col           = 1 + (cfg.cols * sizeof(T) - 1) / cacheline_size;
    constexpr int num_rounds           = 1 + (cfg.rows - 1) / cl_per_round;
    constexpr bool has_remainder_round = num_rounds * cl_per_round != cfg.rows;

    const auto lane = lane_id();

    wave_distribute<num_rounds, cfg.num_waves>(wave, [&](int round) {
        const int m = round * cl_per_round;
        if(!has_remainder_round || m + lane < cfg.rows)
        {
            uint8_t const* row_ptr = reinterpret_cast<uint8_t const*>(base + stride * m);
            static_unroll<cl_per_col>([&](auto n) {
                constexpr int i_offset = cacheline_size * n;
                PrefetchInst::template fetch<i_offset>(row_ptr, sizeof(T) * stride * lane);
            });
        }
    });
}

namespace detail
{
template <typename D, typename A, typename B, typename C>
__device__ void mma_assert()
{
    static_assert(std::is_same_v<typename A::matrix::arch, typename B::matrix::arch>);
    static_assert(std::is_same_v<typename A::matrix::arch, typename C::matrix::arch>);
    static_assert(std::is_same_v<typename A::matrix::arch, typename D::matrix::arch>);
    static_assert(D::matrix::use_ == use::Acc);
    static_assert(A::matrix::use_ == use::A);
    static_assert(B::matrix::use_ == use::B);
    static_assert(C::matrix::use_ == use::Acc);
    static_assert(C::matrix::rows == D::matrix::rows && C::row_blocks == D::row_blocks);
    static_assert(C::matrix::cols == D::matrix::cols && C::col_blocks == D::col_blocks);
    static_assert(C::matrix::rows == A::matrix::rows && C::row_blocks == A::row_blocks);
    static_assert(C::matrix::cols == B::matrix::cols && C::col_blocks == B::col_blocks);
    static_assert(A::matrix::cols == B::matrix::rows && A::col_blocks == B::row_blocks);
}
} // namespace detail

template <typename D, typename A, typename B, typename C>
__device__ void mma(D& d, A& a, B& b, C& c)
{
    detail::mma_assert<D, A, B, C>();

    using arch                 = A::matrix::arch;
    constexpr uint32_t b_reuse = static_cast<uint32_t>(wmma_flag::B_reuse);

#pragma unroll
    for(int nb = 0; nb < C::col_blocks; ++nb)
    {
#pragma unroll
        for(int kb = 0; kb < A::col_blocks; ++kb)
        {
            arch::template mma<>::wmma(
                d.block(0, nb), a.block(0, kb), b.block(kb, nb), c.block(0, nb));
#pragma unroll
            for(int mb = 1; mb < C::row_blocks; ++mb)
            {
                arch::template mma<b_reuse>::wmma(
                    d.block(mb, nb), a.block(mb, kb), b.block(kb, nb), c.block(mb, nb));
            }
        }
    }
}

template <typename C, typename A, typename B>
__device__ void mma(C& c, A& a, B& b)
{
    detail::mma_assert<C, A, B, C>();

    using arch                 = A::matrix::arch;
    constexpr uint32_t b_reuse = static_cast<uint32_t>(wmma_flag::B_reuse);

    auto zero = typename C::matrix{};
#pragma unroll
    for(int nb = 0; nb < C::col_blocks; ++nb)
    {
        arch::template mma<>::wmma(c.block(0, nb), a.block(0, 0), b.block(0, nb), zero);
#pragma unroll
        for(int mb = 1; mb < C::row_blocks; ++mb)
        {
            arch::template mma<b_reuse>::wmma(
                c.block(mb, nb), a.block(mb, 0), b.block(0, nb), zero);
        }
#pragma unroll
        for(int kb = 1; kb < A::col_blocks; ++kb)
        {
            arch::template mma<>::wmma(
                c.block(0, nb), a.block(0, kb), b.block(kb, nb), c.block(0, nb));
#pragma unroll
            for(int mb = 1; mb < C::row_blocks; ++mb)
            {
                arch::template mma<b_reuse>::wmma(
                    c.block(mb, nb), a.block(mb, kb), b.block(kb, nb), c.block(mb, nb));
            }
        }
    }
}

template <typename D, typename A, typename B, typename C, typename AScale, typename BScale>
__device__ void mma_scale(D& d, A& a, B& b, C& c, AScale& ascale, BScale& bscale)
{
    detail::mma_assert<D, A, B, C>();

    using arch = A::matrix::arch;

#pragma unroll
    for(int nb = 0; nb < C::col_blocks; ++nb)
    {
#pragma unroll
        for(int kb = 0; kb < A::col_blocks; ++kb)
        {
            arch::template mma<>::wmma_scale(d.block(0, nb),
                                             a.block(0, kb),
                                             b.block(kb, nb),
                                             c.block(0, nb),
                                             ascale.block(0, kb),
                                             bscale.block(kb, nb));
#pragma unroll
            for(int mb = 1; mb < C::row_blocks; ++mb)
            {
                constexpr uint32_t flags = static_cast<uint32_t>(wmma_flag::A_reuse);
                arch::template mma<flags>::wmma_scale(d.block(mb, nb),
                                                      a.block(mb, kb),
                                                      b.block(kb, nb),
                                                      c.block(mb, nb),
                                                      ascale.block(mb, kb),
                                                      bscale.block(kb, nb));
            }
        }
    }
}

} // namespace bunnies

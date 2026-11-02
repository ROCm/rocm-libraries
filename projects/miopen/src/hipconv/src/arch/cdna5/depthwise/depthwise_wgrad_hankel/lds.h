#pragma once

// The workgroup's LDS: how a row step's staging is laid out, and the reads that
// go at it. The layout is a compile-time function of the config.

#include "config_table.h"
#include "detail.h"
#include "mathutil.h"
#include "types.h"

#include <cstdint>

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;

// Sizes of the buffers a block stages a row step in. An input_stage_* or delta_*
// row is the channel-contiguous form a TDM lands.
struct StageGeometry
{
    int input_cols;
    int input_stage_slot_bytes;
    int delta_rows;
    int delta_pool_rows;
    int delta_pf;
    int delta_row_bytes;
    int block_bytes;
};

constexpr int DELTA_PAD = 0;

constexpr int IN_PAD = 8;

constexpr int TR_GROUP = 8;

constexpr int TDM_LDS_ALIGN = 128;

constexpr int align_up(int n, int a)
{
    return (n + a - 1) / a * a;
}

// The transposing read through the builtin, which is bitwise on 16-bit elements,
// so one definition serves every fragment type.
template <typename Frag>
__device__ inline Frag ds_load_tr16(unsigned lds_addr)
{
    using lds_i16x8 = __attribute__((ext_vector_type(8))) short __attribute__((address_space(3)));
    using i16x8     = __attribute__((ext_vector_type(8))) short;
    const i16x8 raw = __builtin_amdgcn_ds_load_tr16_b128_v8i16((lds_i16x8*)(__SIZE_TYPE__)lds_addr);
    return __builtin_bit_cast(Frag, raw);
}

using dword4 = __attribute__((ext_vector_type(4))) unsigned;

// A store named in the LDS address space rather than through a generic pointer,
// so the width asked for is the width the ds_store carries.
template <typename T>
__device__ inline void ds_store(unsigned lds_addr, T v)
{
    using lds_ptr                                       = T __attribute__((address_space(3)))*;
    *reinterpret_cast<lds_ptr>((__SIZE_TYPE__)lds_addr) = v;
}

// One store of exactly N consecutive floats, N being the taps of a channel a
// lane's half of the accumulator holds. Compile-time, so the width is one store.
template <int N>
__device__ inline void store_n(float* dst, const float (&v)[N])
{
    using f32x2 = __attribute__((ext_vector_type(2))) float;
    using f32x3 = __attribute__((ext_vector_type(3))) float;
    using f32x4 = __attribute__((ext_vector_type(4))) float;
    static_assert(N >= 1 && N <= 8, "a channel's taps in one half of the accumulator");
    if constexpr(N >= 4)
    {
        const f32x4 w{v[0], v[1], v[2], v[3]};
        __builtin_memcpy(dst, &w, sizeof(w));
        if constexpr(N > 4)
        {
            float rest[N - 4];
            static_for<N - 4>([&]<int I>() { rest[I] = v[4 + I]; });
            store_n(dst + 4, rest);
        }
    }
    else if constexpr(N == 3)
    {
        const f32x3 w{v[0], v[1], v[2]};
        __builtin_memcpy(dst, &w, sizeof(w));
    }
    else if constexpr(N == 2)
    {
        const f32x2 w{v[0], v[1]};
        __builtin_memcpy(dst, &w, sizeof(w));
    }
    else
        dst[0] = v[0];
}

// Every buffer is sized by the tile the engine lands rather than by the slice a
// wave covers. Wave derives the same column counts; see its OWN and NHALO.
constexpr StageGeometry stage_geometry(const Config& cfg)
{
    constexpr int elem_bytes = 2;
    const int wg_ch          = cfg.staged_channels();
    const int qw             = cfg.q_per_wave();
    StageGeometry g{};
    const int qw_half    = qw / 2;
    const int own        = qw_half * cfg.stride / TR_GROUP;
    const int nout       = qw_half / 2;
    const bool sel_whole = cfg.stride == 1 && cfg.kw <= 3;
    const int coarse_max = sel_whole ? 0 : (cfg.kw - 1) / 2;
    const int e_n        = cfg.stride * (nout - 1) + coarse_max + 2;
    const int nhalo      = (e_n - 4 * own + 3) / 4;
    const int reach      = (2 * own + nhalo) * TR_GROUP;
    const int span       = cfg.stride * (qw - 1) + cfg.kw;
    g.input_cols         = align_up(span, TR_GROUP) >= reach ? span : (span > reach ? span : reach);
    const int input_row  = wg_ch + (cfg.a_from_tr16() ? IN_PAD : 0);
    g.input_stage_slot_bytes =
        align_up(align_up(g.input_cols, TR_GROUP) * input_row * elem_bytes, TDM_LDS_ALIGN) +
        TDM_LDS_ALIGN;
    g.delta_rows      = cfg.kh;
    g.delta_pf        = 3;
    g.delta_pool_rows = cfg.b_rotates()
                            ? (g.delta_rows > g.delta_pf + 1 ? g.delta_rows : g.delta_pf + 1)
                            : g.delta_rows + g.delta_pf;
    g.delta_row_bytes = align_up((qw + DELTA_PAD) * wg_ch * elem_bytes, TDM_LDS_ALIGN);
    g.block_bytes =
        cfg.prefetch_depth * g.input_stage_slot_bytes + g.delta_pool_rows * g.delta_row_bytes;
    return g;
}

} // namespace hipconv::cdna5::depthwise_wgrad_hankel

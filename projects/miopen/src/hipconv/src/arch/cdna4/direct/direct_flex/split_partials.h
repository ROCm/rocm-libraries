#pragma once

// Split-K: the piece count, the fp32 partials and the reduce kernel.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "layer_pars.h"
#include "bunnies_cdna4.hpp"
#include "magic_division.h"
#include "packed_ops.h"
#include "types.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace hipconv::cdna4::direct_flex
{

// A split's fixed cost in steps of the tile, measured (Split-K).
constexpr int split_cost_steps(const Config& cfg)
{
    return cfg.k_tile() == 256 ? 14 : 12;
}

inline int env_int(const char* name, int fallback = 0)
{
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// HIPCONV_DIRECT_FLEX_SPLIT=<n> pins at most n pieces per tile; 1 disables the split.
inline int split_pin()
{
    static const int pin = env_int("HIPCONV_DIRECT_FLEX_SPLIT");
    return pin;
}

// The largest divisor of chunks at most split_pin(), or 0 without a pin.
inline int pinned_pieces(int chunks)
{
    if(split_pin() <= 0)
        return 0;
    int pieces = std::min(split_pin(), chunks);
    while(chunks % pieces != 0)
        --pieces;
    return pieces;
}

// Steps of one piece of a split tile: its reduction share and the split's fixed cost (Split-K).
inline int split_steps(const Config& cfg, int taps, int chunks, int pieces)
{
    return taps * chunks / pieces + split_cost_steps(cfg);
}

// Split-K pieces per tile: the even division of the chunks the step model ranks fastest.
inline int split_pieces(const Config& cfg, const ConvParams& par, int cu_count)
{
    // A csplit tile splits each chunk within the workgroup and never across workgroups.
    if(cfg.csplit != 1)
        return 1;
    const int chunks = (par.c + C_CHUNK - 1) / C_CHUNK;
    const int tiles  = tile_count(cfg, par);
    if(const int pin = pinned_pieces(chunks); pin > 0)
        return pin;
    const int taps = filter_taps(par);
    int best       = 1;
    int best_steps = taps * chunks;
    for(int pieces = 2; pieces <= chunks && tiles * pieces <= cu_count; ++pieces)
    {
        const int steps = split_steps(cfg, taps, chunks, pieces);
        if(chunks % pieces == 0 && steps < best_steps)
        {
            best       = pieces;
            best_steps = steps;
        }
    }
    return best;
}

// A split's partials: one fp32 copy of the output per piece, padded to whole tiles.
inline size_t partials_bytes(const Config& cfg, const ConvParams& par, int pieces)
{
    const int x_tiles = static_cast<int>((out_rows(par) + cfg.x_tile() - 1) / cfg.x_tile());
    const int k_tiles = (par.k + cfg.k_tile() - 1) / cfg.k_tile();
    return static_cast<size_t>(pieces) * x_tiles * cfg.x_tile() * k_tiles * cfg.k_tile() *
           sizeof(float);
}

constexpr int REDUCE_THREADS = 256;

struct ReducePhase
{
    int rows, q, p;
    int out0;
    int copy0, copies, copy_stride; // in floats
    MagicDiv q_div, p_div;
    int e = 1; // a conv3d's output planes
    MagicDiv e_div{};
};
struct ReducePars
{
    ReducePhase phases[MAX_PHASES];
    int n_phases;
    int k, k_pad;
    int out_hw, out_row, out_sw;
    int out_plane = 0;
};

// Split-K reduce: sums partials in piece order, so the result is deterministic.
//
// A phased dgrad's tapless phase has no copies and is written as zeros.
// CONTIG: one phase whose rows are the output's pixels in order.
template <DataType DT, bool DEPTH, bool CONTIG>
__global__
__launch_bounds__(REDUCE_THREADS) void direct_flex_reduce(const float* __restrict__ partials,
                                                          ToType<DT>* __restrict__ out,
                                                          ReducePars rp)
{
    using dtype     = ToType<DT>;
    using fp32x4    = bunnies::floatx4;
    using dtypex4   = packed_vec_t<dtype, 4>;
    const int quads = (rp.k + 3) / 4;
    int64_t total   = 0;
    for(int f = 0; f < (CONTIG ? 1 : rp.n_phases); ++f)
        total += static_cast<int64_t>(rp.phases[f].rows) * quads;
    const int stride = gridDim.x * REDUCE_THREADS;
    for(int64_t i = blockIdx.x * REDUCE_THREADS + threadIdx.x; i < total; i += stride)
    {
        int f       = 0;
        int64_t idx = i;
        if constexpr(!CONTIG)
            while(idx >= static_cast<int64_t>(rp.phases[f].rows) * quads)
                idx -= static_cast<int64_t>(rp.phases[f++].rows) * quads;
        const ReducePhase& g = rp.phases[f];
        const int x          = static_cast<int>(idx / quads);
        const int kk         = static_cast<int>(idx - static_cast<int64_t>(x) * quads) * 4;
        const float* p       = partials + g.copy0 + static_cast<int64_t>(x) * rp.k_pad + kk;
        using load           = bunnies::arch_cdna4::global_load_b128;
        fp32x4 sum           = {0, 0, 0, 0};
        if constexpr(CONTIG)
            load::load(p, &sum);
        for(int j = CONTIG ? 1 : 0; j < g.copies; ++j)
        {
            fp32x4 v;
            load::load(p + static_cast<int64_t>(j) * g.copy_stride, &v);
            sum += v;
        }
        const int row = static_cast<int>(g.q_div.div(static_cast<uint32_t>(x)));
        const int col = x - row * g.q;
        int64_t pixel;
        if constexpr(CONTIG)
            pixel = x;
        else if constexpr(DEPTH)
        {
            const int plane = static_cast<int>(g.p_div.div(static_cast<uint32_t>(row)));
            const int img   = static_cast<int>(g.e_div.div(static_cast<uint32_t>(plane)));
            pixel           = static_cast<int64_t>(img) * rp.out_hw + g.out0 +
                    (plane - img * g.e) * rp.out_plane + (row - plane * g.p) * rp.out_row +
                    col * rp.out_sw;
        }
        else
        {
            const int img = static_cast<int>(g.p_div.div(static_cast<uint32_t>(row)));
            pixel         = static_cast<int64_t>(img) * rp.out_hw + g.out0 +
                    (row - img * g.p) * rp.out_row + col * rp.out_sw;
        }
        dtype* dst = out + pixel * rp.k + kk;
        if(rp.k % 4 == 0)
            *reinterpret_cast<dtypex4*>(dst) = packed_convert<dtype>(sum);
        else
            for(int e = 0; e < 4 && kk + e < rp.k; ++e)
                dst[e] = static_cast<dtype>(sum[e]);
    }
}

// Instantiated once, in direct_flex_reduce.cpp, so the shards that launch a split share one copy.
#define HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DT, DEPTH, CONTIG)              \
    EXTERN template __global__ void direct_flex_reduce<DT, DEPTH, CONTIG>( \
        const float* __restrict__, ToType<DT>* __restrict__, ReducePars);
#define HIPCONV_DIRECT_FLEX_REDUCES(EXTERN)                          \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::fp16, false, false) \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::fp16, true, false)  \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::fp16, false, true)  \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::bf16, false, false) \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::bf16, true, false)  \
    HIPCONV_DIRECT_FLEX_REDUCE(EXTERN, DataType::bf16, false, true)
HIPCONV_DIRECT_FLEX_REDUCES(extern)

// Reduce parameters of an unphased split: one fp32 copy of the output per piece, back to back.
inline ReducePars contig_reduce_pars(const Config& cfg, const ConvParams& par, int pieces)
{
    ReducePars rp{};
    rp.k           = par.k;
    rp.k_pad       = (par.k + cfg.k_tile() - 1) / cfg.k_tile() * cfg.k_tile();
    rp.n_phases    = 1;
    ReducePhase& g = rp.phases[0];
    g.rows         = static_cast<int>(out_rows(par));
    g.copies       = pieces;
    g.copy_stride  = static_cast<int>(partials_bytes(cfg, par, 1) / sizeof(float));
    return rp;
}

} // namespace hipconv::cdna4::direct_flex

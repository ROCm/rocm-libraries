#pragma once

// Stores of a tile's accumulators: the output, or a split piece's fp32 partials.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "tile_order.h"
#include "traits.h"
#include "bunnies.hpp"
#include "detail.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

template <class T>
class OutputWriter
{
    using Args   = KernelArgs<T::PHASED>;
    using dtype  = typename T::dtype;
    using arch   = bunnies::arch_cdna4;
    using rt_acc = typename T::rt_acc;
    using rt_out = typename T::rt_out;

    static constexpr Config cfg     = T::cfg;
    static constexpr int BLOCK_X16  = T::BLOCK_X16;
    static constexpr int BLOCK_K16  = T::BLOCK_K16;
    static constexpr int HALF       = T::HALF;
    static constexpr int HALF_K     = T::HALF_K;
    static constexpr int elem_bytes = T::elem_bytes;

public:
    __device__
    OutputWriter(const Args& args, dtype* out, const int& xi, const int& ki, rt_acc (&acc)[2][2])
        : args_(args)
        , out_(out)
        , xi_(xi)
        , ki_(ki)
        , acc_(acc)
    {
    }

    // Stores a tile, permuting lanes so four write a row's 64 bytes (Epilogue).
    __device__ __forceinline__ void store_tile(Tile t) const
    {
        typename Args::Pars& kp = args_.cold();
        const int lane          = bunnies::volatile_lane_id();
        // Pinned: offsets derived from xi_ and ki_ were hoisted out of the main loop and spilled.
        const int xi = bunnies::pin_sgpr(xi_), ki = bunnies::pin_sgpr(ki_);
        // After the exchange, a lane holds piece `piece` of row `row` of a 16-row block.
        const int row   = lane / 4;
        const int piece = lane % 4;
        int rows        = kp.rows;
        int64_t pix0    = t.x0;
        int pix_row[2][BLOCK_X16];
        if constexpr(T::PHASED)
        {
            const Geom g       = args_.geom();
            rows               = g.rows;
            const bool strided = kp.strided != 0;
            Origin o{};
            if(strided)
            {
                o    = origin_of(g, t.x0);
                pix0 = o.pix0;
            }
#pragma unroll
            for(int i = 0; i < 2; ++i)
#pragma unroll
                for(int xb = 0; xb < BLOCK_X16; ++xb)
                {
                    const int dm   = i * HALF + xi * cfg.wave_x() + xb * MFMA_M + row;
                    pix_row[i][xb] = strided ? pixel_after(g, o, t.x0 + dm) : dm;
                }
        }
        const int64_t r_base = pix0 * kp.k;
        const int out_elems  = span(kp.out_elems - r_base);
        const auto out_rsrc  = arch::make_buffer(out_ + r_base, out_elems);
        const int out_oob    = out_elems * elem_bytes;
        auto pixel           = [&](int i, int xb, int dm) {
            if constexpr(T::PHASED)
                return pix_row[i][xb];
            return dm;
        };
        auto voffset = [&](int dm, int pix, int kk) {
            return t.x0 + dm < rows && kk < kp.k ? (pix * kp.k + kk) * elem_bytes : out_oob;
        };
        const int src_pair   = (((piece & 1) << 1 | piece >> 1) * 16 + row) * 4;
        const int src_single = (piece * 16 + row) * 4;
        auto permute         = [](int src, uint32_t v) {
            return static_cast<uint32_t>(__builtin_amdgcn_ds_bpermute(src, static_cast<int>(v)));
        };
        const auto coord = T::mat_out::map({lane, 0});
        static_for<2>([&]<int I>() {
            static_for<2>([&]<int J>() {
                rt_out narrowed;
                bunnies::tile_cast(narrowed, acc_[I][J]);
                const int k_wave = t.k0 + J * HALF_K + ki * cfg.wave_k();
#pragma unroll
                for(int xb = 0; xb < BLOCK_X16; ++xb)
                {
                    const int x_block = I * HALF + xi * cfg.wave_x() + xb * MFMA_M;
                    if constexpr(!T::NARROW_K)
                    {
                        const int dm  = x_block + row;
                        const int pix = pixel(I, xb, dm);
#pragma unroll
                        for(int kb = 0; kb + 1 < BLOCK_K16; kb += 2)
                        {
                            const auto a =
                                __builtin_bit_cast(bunnies::uint32x2, narrowed.block(kb, xb).data);
                            const auto b = __builtin_bit_cast(bunnies::uint32x2,
                                                              narrowed.block(kb + 1, xb).data);
                            const auto sx =
                                __builtin_amdgcn_permlane16_swap(a[0], b[0], false, false);
                            const auto sy =
                                __builtin_amdgcn_permlane16_swap(a[1], b[1], false, false);
                            bunnies::uint32x4 v = {permute(src_pair, sx[0]),
                                                   permute(src_pair, sy[0]),
                                                   permute(src_pair, sx[1]),
                                                   permute(src_pair, sy[1])};
                            arch::buffer_store<16>::store(
                                out_rsrc,
                                &v,
                                voffset(dm, pix, k_wave + kb * MFMA_N + piece * 8),
                                0);
                        }
                        if constexpr(BLOCK_K16 % 2 != 0)
                        {
                            constexpr int kb = BLOCK_K16 - 1;
                            const auto a =
                                __builtin_bit_cast(bunnies::uint32x2, narrowed.block(kb, xb).data);
                            bunnies::uint32x2 v = {permute(src_single, a[0]),
                                                   permute(src_single, a[1])};
                            arch::buffer_store<8>::store(
                                out_rsrc,
                                &v,
                                voffset(dm, pix, k_wave + kb * MFMA_N + piece * 4),
                                0);
                        }
                    }
                    else
                    {
                        const int dm = x_block + coord[1];
#pragma unroll
                        for(int kb = 0; kb < BLOCK_K16; ++kb)
                        {
                            auto& blk    = narrowed.block(kb, xb);
                            const int kk = k_wave + kb * MFMA_N + coord[0];
#pragma unroll
                            for(int e = 0; e < 4; ++e)
                            {
                                dtype value = blk.data[e];
                                arch::buffer_store<2>::store(
                                    out_rsrc, &value, voffset(dm, dm, kk + e), 0);
                            }
                        }
                    }
                }
            });
        });
    }

    // Stores a split piece's fp32 accumulators to its padded copy (Split-K).
    // piece_of(kp) gives the split-piece index; computed by the caller, it added scalar code.
    template <class PieceOf>
    __device__ __forceinline__ void store_partial(Tile t, PieceOf&& piece_of) const
    {
        typename Args::Pars& kp = args_.cold();
        const int lane          = bunnies::volatile_lane_id();
        const int k_pad         = kp.tiles_k * cfg.k_tile();
        const int piece_i       = piece_of(kp);
        int64_t first;
        if constexpr(T::PHASED)
            first = args_.seg().copy0 + (static_cast<int64_t>(piece_i) *
                                             kp.phases[args_.seg().phase].x_tiles * cfg.x_tile() +
                                         t.x0) *
                                            k_pad;
        else
            first = (static_cast<int64_t>(piece_i) * kp.x_tiles * cfg.x_tile() + t.x0) * k_pad;
        const auto rsrc   = arch::make_buffer(kp.partials + first + t.k0,
                                            static_cast<int64_t>(cfg.x_tile()) * k_pad);
        const int dst_row = lane / 4;
        const int piece   = lane % 4;
        const int src     = (piece * 16 + dst_row) * 4;
        static_for<2>([&]<int I>() {
            static_for<2>([&]<int J>() {
#pragma unroll
                for(int xb = 0; xb < BLOCK_X16; ++xb)
#pragma unroll
                    for(int kb = 0; kb < BLOCK_K16; ++kb)
                    {
                        const int dm = I * HALF + xi_ * cfg.wave_x() + xb * MFMA_M + dst_row;
                        const int dk = J * HALF_K + ki_ * cfg.wave_k() + kb * MFMA_N + piece * 4;
                        const auto a =
                            __builtin_bit_cast(bunnies::uint32x4, acc_[I][J].block(kb, xb).data);
                        bunnies::uint32x4 v;
#pragma unroll
                        for(int e = 0; e < 4; ++e)
                            v[e] = static_cast<uint32_t>(
                                __builtin_amdgcn_ds_bpermute(src, static_cast<int>(a[e])));
                        arch::buffer_store<16>::store(rsrc, &v, (dm * k_pad + dk) * 4, 0);
                    }
            });
        });
    }

    // Clamps n >= 0 to INT32_MAX / elem_bytes.
    //
    // 32-bit halves, the high one pinned: a 64-bit compare ran in VALU against a constant held in
    // two VGPRs through the main loop.
    static __device__ int span(int64_t n)
    {
        constexpr uint32_t LIM = INT32_MAX / elem_bytes;
        const auto u           = static_cast<uint64_t>(n);
        const uint32_t hi      = bunnies::pin_sgpr(static_cast<uint32_t>(u >> 32));
        const auto lo          = static_cast<uint32_t>(u);
        return static_cast<int>(hi != 0 || lo > LIM ? LIM : lo);
    }

private:
    // A strided phase's tile origin and pixel stepping (Epilogue mapping).
    struct Origin
    {
        int n0, i0, j0;
        int64_t pix0;
        int e0 = 0;
    };

    __device__ __forceinline__ Origin origin_of(const Geom& g, int x0) const
    {
        typename Args::Pars& kp = args_.cold();
        if constexpr(T::DEPTH)
        {
            const int f = args_.seg().phase;
            const int e = kp.dep.pe[f];
            const int row0 =
                static_cast<int>(detail::magic_div(*g.q_div, static_cast<uint32_t>(x0)));
            const int j0 = x0 - row0 * g.q;
            const int plane0 =
                static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(row0)));
            const int i0 = row0 - plane0 * g.p;
            const int n0 = static_cast<int>(
                detail::magic_div(kp.dep.pe_div[f], static_cast<uint32_t>(plane0)));
            const int e0 = plane0 - n0 * e;
            return Origin{n0,
                          i0,
                          j0,
                          static_cast<int64_t>(n0) * kp.out_hw + g.out0 + e0 * kp.dep.out_plane +
                              i0 * kp.out_row + j0 * kp.out_sw,
                          e0};
        }
        const int row0 = static_cast<int>(detail::magic_div(*g.q_div, static_cast<uint32_t>(x0)));
        const int j0   = x0 - row0 * g.q;
        const int n0   = static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(row0)));
        const int i0   = row0 - n0 * g.p;
        return Origin{n0,
                      i0,
                      j0,
                      static_cast<int64_t>(n0) * kp.out_hw + g.out0 + i0 * kp.out_row +
                          j0 * kp.out_sw};
    }

    __device__ __forceinline__ int pixel_after(const Geom& g, const Origin& o, int x) const
    {
        typename Args::Pars& kp = args_.cold();
        if constexpr(T::DEPTH)
        {
            const int f = args_.seg().phase;
            const int r = static_cast<int>(detail::magic_div(*g.q_div, static_cast<uint32_t>(x)));
            const int j = x - r * g.q;
            const int plane =
                static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(r)));
            const int i = r - plane * g.p;
            const int n =
                static_cast<int>(detail::magic_div(kp.dep.pe_div[f], static_cast<uint32_t>(plane)));
            const int e = plane - n * kp.dep.pe[f];
            return (n - o.n0) * kp.out_hw + (e - o.e0) * kp.dep.out_plane +
                   (i - o.i0) * kp.out_row + (j - o.j0) * kp.out_sw;
        }
        const int r = static_cast<int>(detail::magic_div(*g.q_div, static_cast<uint32_t>(x)));
        const int j = x - r * g.q;
        const int n = static_cast<int>(detail::magic_div(*g.p_div, static_cast<uint32_t>(r)));
        const int i = r - n * g.p;
        return (n - o.n0) * kp.out_hw + (i - o.i0) * kp.out_row + (j - o.j0) * kp.out_sw;
    }

    const Args& args_;
    dtype* __restrict__ out_;
    const int& xi_;
    const int& ki_;
    rt_acc (&acc_)[2][2];
};

} // namespace hipconv::cdna4::direct_flex

#pragma once

#include "config.hpp"
#include "config_matcher.hpp"
#include "bunnies.hpp"
#include "bunnies_mi400.hpp"
#include "tensor_view.hpp"
#include "conv_kernel.h"
#include "direct_conv_kernel.h"
#include "mathutil.h"
#include "launch_params.h"
#include "types.h"
#include "hipconv/conv_params.hpp"
#include "hip_util.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace hipconv::cdna5::direct_wgrad
{

using arch = bunnies::arch_mi400;

// dw[g][k][r][s][c]
//      = sum_{n,p,q} dy[n][p][q][g][k]
//                    * input[n][p*sh + r*dh - pad_h][q*sw + s*dw - pad_w][g][c]
//
// Arithmetic intensity
// 2*G*K*R*S*C*N*P*Q / (2*N*P*Q*G*K + 2*N*(P+R)*(Q+S)*G*C + 4*G*K*R*S*C)
//      = K*R*S*C*N*P*Q / (N*P*Q*K + N*(P+R)*(Q+S)*C + 2*K*R*S*C)
//      = K*R*S*C / (K + (1+R/P)*(1+S/Q)*C + 2*K*R*S*C/(N*P*Q))
// P >> R, Q >> S
//      = K*R*S*C / (K + C + 2*K*R*S*C/(N*P*Q))
// => max R,S; balance K and C
//
// Blocking:
// per-wave: R(kh) + S(kw) + K(2) + C(2)
// per-wg:   R(kh) + S(kw) + K(4) + C(8)
//    --or-- R(kh) + S(kw) + K(8) + C(4)
//
// Summation bounds:
//
// dw[g][k][r][s][c]
//     = sum_{n=0}^{N-1} sum_{p=0}^{P-1} sum_{q=0}^{Q-1}
//           dy[n][p][q][g][k] * input[n][p + r - pad_h][q + s - pad_w][g][c]
//
// Out-of-bounds accesses are defined to be zero and do not contribute to the sum:
//
// p+r-pad_h >= 0 and p >= 0, therefore p >= max(0, pad_h-(kh-1))
// p+r-pad_h < H  and p < P,  therefore p < min(H+pad_h, P)
//
// dw[g][k][r][s][c]
//     = sum_{p=max(0, pad_h-(kh-1))}^{min(H+pad_h-1,P-1)}
//       sum_{q=max(0, pad_w-(kw-1))}^{min(W+pad_w-1,Q-1)}
//       sum_{n=0}^{N-1} dy[n][p][q][g][k] * input[n][p + r - pad_h][q + s - pad_w][g][c]
//

// Global modes
BUNNIES_DEFINE_MODE(N, int64_t)
BUNNIES_DEFINE_MODE(G, int64_t)
BUNNIES_DEFINE_MODE(C, int64_t)
BUNNIES_DEFINE_MODE(K, int64_t)
BUNNIES_DEFINE_MODE(R, int64_t)
BUNNIES_DEFINE_MODE(S, int64_t)
BUNNIES_DEFINE_MODE(H, int64_t)
BUNNIES_DEFINE_MODE(W, int64_t)
BUNNIES_DEFINE_MODE(P, int64_t)
BUNNIES_DEFINE_MODE(Q, int64_t)
// LDS modes
BUNNIES_DEFINE_MODE(Kblock, int)
BUNNIES_DEFINE_MODE(Cblock, int)
BUNNIES_DEFINE_MODE(Pblock, int)
BUNNIES_DEFINE_MODE(Qblock, int)
BUNNIES_DEFINE_MODE(Hring, int)
BUNNIES_DEFINE_MODE(Wpad, int)
BUNNIES_DEFINE_MODE(Rblock, int)
BUNNIES_DEFINE_MODE(Sblock, int)

enum class wgrad_wave_type
{
    in_tdm,
    dy_tdm,
    regular
};

inline __device__ auto wave_type(int wave_id_c, int wave_id_k) -> wgrad_wave_type
{
    if(wave_id_k == 0 && wave_id_c == 0)
        return wgrad_wave_type::in_tdm;
    if(wave_id_k == 0 && wave_id_c == 1)
        return wgrad_wave_type::dy_tdm;
    return wgrad_wave_type::regular;
}

template <Config cfg, hipconv::DataType DT>
__device__ void conv2d_direct_wgrad_cdna5_nhwc_impl(const ToType<DT>* __restrict__ in,
                                                    const ToType<DT>* __restrict__ dy,
                                                    float* __restrict__ dw,
                                                    int batch_size,
                                                    int groups,
                                                    int c_per_group,
                                                    int k_per_group,
                                                    int hi,
                                                    int wi,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dilation_y,
                                                    int dilation_x,
                                                    int py,
                                                    int px)
{
    using T                         = ToType<DT>;
    namespace bn                    = bunnies;
    constexpr auto tiles            = cfg.tiles();
    constexpr auto reg_stride_c     = cfg.reg_stride_c();
    constexpr auto reg_stride_k     = cfg.reg_stride_k();
    constexpr auto reg_tiles_c      = cfg.reg_tiles_c();
    constexpr auto reg_tiles_k      = cfg.reg_tiles_k();
    constexpr auto block_size_q     = cfg.wmma_size_q;
    constexpr auto block_size_q_pad = block_size_q + (cfg.kw - 1);
    constexpr auto half_fmt = DT == hipconv::DataType::bf16 ? bn::fpfmt::e8m7 : bn::fpfmt::e5m10;
    constexpr auto prefill_factor = divup(cfg.chunk_size_p + (cfg.kh - 1), cfg.chunk_size_p);

    static_assert(cfg.ring_size_p % cfg.chunk_size_p == 0,
                  "tile_size_h must be divisible by chunk_size_p");
    static_assert((1 + prefill_factor) * cfg.chunk_size_p <= cfg.ring_size_p,
                  "Beginning and end of p ring buffer must not overlap");
    static_assert(cfg.chunk_size_p >= cfg.kh - 1, "Chunk size must be larger than or equal kh - 1");

    using mat_a = arch::matrix<half_fmt, cfg.wmma_size_c, cfg.wmma_size_q, bn::use::A>;
    using mat_b = arch::matrix<half_fmt, cfg.wmma_size_q, cfg.wmma_size_k, bn::use::B>;
    using mat_c = arch::matrix<bn::fpfmt::e8m23, cfg.wmma_size_c, cfg.wmma_size_k, bn::use::Acc>;
    using rt_a  = bn::reg_tile<mat_a, cfg.kh * cfg.kw * reg_tiles_c, 1>;
    using rt_b  = bn::reg_tile<mat_b, 1, reg_tiles_k>;
    using rt_c  = bn::reg_tile<mat_c, cfg.kh * cfg.kw * reg_tiles_c, reg_tiles_k>;

    const int lane      = bn::lane_id();
    const int wave_id   = bn::wave_id();
    const int wave_id_c = __builtin_amdgcn_readfirstlane(wave_id % cfg.tiles_c);
    const int wave_id_k = __builtin_amdgcn_readfirstlane(wave_id / cfg.tiles_c);
    const int wave_rank = __builtin_amdgcn_readfirstlane(wave_id / 4);
    const auto blocks_k = divup(k_per_group, cfg.tile_size_k);
    const int tile_g    = blockIdx.y / blocks_k;
    const int tile_k    = blockIdx.y % blocks_k * cfg.tile_size_k;
    const int tile_c    = blockIdx.x * cfg.tile_size_c;

    // TODO: Generalize N-tiling to NHW tiling.
    const int base_tile_size_n = 1 + (batch_size - 1) / gridDim.z;
    const int tile_n           = blockIdx.z * base_tile_size_n;

    if(tile_n >= batch_size)
        return;
    const int tile_size_n = std::min(base_tile_size_n, batch_size - tile_n);

    rt_c c_acc = {};

    constexpr int num_buf              = 2;
    constexpr int k_pad_amount         = 8; // in dwords
    constexpr int k_pad_amount_item    = k_pad_amount * 4 / sizeof(T);
    constexpr int c_pad_amount         = 8; // in dwords
    constexpr int c_pad_amount_item    = c_pad_amount * 4 / sizeof(T);
    constexpr int c_dw_pad_amount      = 4; // in dwords
    constexpr int c_dw_pad_amount_item = c_dw_pad_amount * 4 / sizeof(float);

    constexpr auto dy_lds_shape =
        Pblock(cfg.ring_size_p) + Qblock(block_size_q) + Kblock(cfg.tile_size_k);
    constexpr auto dy_lds_stride = [&] {
        constexpr auto q_stride = Kblock(dy_lds_shape).get() + k_pad_amount_item;
        constexpr auto p_stride = Qblock(dy_lds_shape).get() * q_stride;
        return Pblock(p_stride) + Qblock(q_stride) + Kblock(1);
    }();
    constexpr int dy_lds_size = Pblock(dy_lds_shape).get() * Pblock(dy_lds_stride).get();

    constexpr auto in_lds_shape =
        Hring(cfg.ring_size_p) + Wpad(block_size_q_pad) + Cblock(cfg.tile_size_c);
    constexpr auto in_lds_stride = [&] {
        constexpr auto w_stride = Cblock(in_lds_shape).get() + c_pad_amount_item;
        constexpr auto h_stride = Wpad(in_lds_shape).get() * w_stride;
        return Hring(h_stride) + Wpad(w_stride) + Cblock(1);
    }();
    constexpr int in_lds_size = Hring(in_lds_shape).get() * Hring(in_lds_stride).get() +
                                // We need space for extra w because we may write out of bounds
                                Wpad(in_lds_shape).get() * Wpad(in_lds_stride).get();

    constexpr auto dw_lds_shape =
        Kblock(cfg.tile_size_k) + Rblock(cfg.kh) + Sblock(cfg.kw) + Cblock(cfg.tile_size_c);
    constexpr auto dw_lds_stride = [&] {
        constexpr auto s_stride = Cblock(dw_lds_shape).get() + c_dw_pad_amount_item;
        constexpr auto r_stride = Sblock(dw_lds_shape).get() * s_stride;
        constexpr auto k_stride = Rblock(dw_lds_shape).get() * r_stride;
        return Kblock(k_stride) + Rblock(r_stride) + Sblock(s_stride) + Cblock(1);
    }();
    constexpr int dw_lds_size =
        Kblock(dw_lds_shape).get() * Kblock(dw_lds_stride).get() * sizeof(float) / sizeof(T);

    __shared__ T lds[std::max(dy_lds_size + in_lds_size, dw_lds_size)];
    T* dy_lds = &lds[0];
    T* in_lds = &lds[dy_lds_size];

    const int qbegin = std::max(0, px - (cfg.kw - 1));
    const int qend   = std::min(wi + px, wo);
    const int qcount = qend - qbegin;
    const int pbegin = std::max(0, py - (cfg.kh - 1));
    const int pend   = std::min(hi + py, ho);
    const int pcount = pend - pbegin;

    // w0 = qbegin - px = max(0, px - (kw - 1)) - px
    // px > (kw - 1):
    //     w0 = -(kw-1)
    // px <= (kw - 1):
    //     0 >= w0 = -px >= -(kw - 1)
    //
    // => 0 >= w0 >= -(kw - 1)
    // => 0 <= w0_adj = -w0 <= kw - 1
    const int w0_adj = px - qbegin;

    // 0 <= h0_adj <= kh - 1
    const int h0_adj = py - pbegin;

    const auto dy_view =
        bn::make_named_memref(dy,
                              N(batch_size) + P(ho) + Q(wo) + G(groups) + K(k_per_group),
                              bn::mode_order<K, G, Q, P, N>{})
            .moved_by(N(tile_n) + G(tile_g) + K(tile_k) + P(pbegin) + Q(qbegin));
    const auto in_view =
        bn::make_named_memref(in,
                              N(batch_size) + H(hi) + W(wi) + G(groups) + C(c_per_group),
                              bn::mode_order<C, G, W, H, N>{})
            .moved_by(N(tile_n) + G(tile_g) + C(tile_c));

    const auto dy_lds_view      = bn::named_view(dy_lds, dy_lds_shape, dy_lds_stride);
    const auto dy_lds_view_part = dy_lds_view.moved_by(Kblock(wave_id_k * reg_stride_k));
    const auto in_lds_view      = bn::named_view(in_lds, in_lds_shape, in_lds_stride);
    const auto in_lds_view_part = in_lds_view.moved_by(Cblock(wave_id_c * reg_stride_c));

    auto const init_tdm = [&](arch::tdm_group0& d0, arch::tdm_group1& d1, arch::tdm_group2& d2) {
        d1.data_size = bn::ilog2(sizeof(T));
        if(wave_type(wave_id_c, wave_id_k) == wgrad_wave_type::in_tdm)
        {
            const auto& in_shape  = in_view.shape();
            const auto& in_stride = in_view.stride();

            d1.tile_dim0 = Cblock(in_lds_shape).get();
            arch::configure_padding<sizeof(T),
                                    Cblock(in_lds_shape).get(),
                                    Wpad(in_lds_stride).get()>(d1);
            d1.set_tensor_dim0(C(in_shape).get());

            d1.tile_dim1 = Wpad(in_lds_shape).get();
            d1.set_tensor_stride1(W(in_stride).get());

            d1.tile_dim2 = cfg.chunk_size_p;
            d1.set_tensor_stride2(H(in_stride).get());
        }
        else if(wave_type(wave_id_c, wave_id_k) == wgrad_wave_type::dy_tdm)
        {
            const auto& dy_shape  = dy_view.shape();
            const auto& dy_stride = dy_view.stride();

            d1.tile_dim0 = Kblock(dy_lds_shape).get();
            arch::configure_padding<sizeof(T),
                                    Kblock(dy_lds_shape).get(),
                                    Qblock(dy_lds_stride).get()>(d1);
            d1.set_tensor_dim0(K(dy_shape).get());

            d1.tile_dim1 = Qblock(dy_lds_shape).get();
            d1.set_tensor_stride1(Q(dy_stride).get());

            d1.tile_dim2 = cfg.chunk_size_p;
            d1.set_tensor_stride2(P(dy_stride).get());
        }
        else
        {
            d0.count = 0;
        }
    };

    auto d0 = arch::tdm_group0{};
    auto d1 = arch::tdm_group1{};
    auto d2 = arch::tdm_group2{};
    auto d3 = arch::tdm_group3{};
    auto d4 = arch::tdm_group4{};
    init_tdm(d0, d1, d2);

    if(wave_type(wave_id_c, wave_id_k) == wgrad_wave_type::in_tdm)
    {
        // Use TDM to zero LDS
        d0.set_lds_addr(in_lds_view.offset());
        d0.set_global_addr(in_view.offset());
        d1.tile_dim2 = Hring(in_lds_shape).get();
        d1.set_tensor_dim1(0);
        d2.set_tensor_dim2(H(in_view.shape()).get());
        __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
    }

    auto const prefill_in = [&](auto in_view,
                                int in_ring0,
                                int w0_adj,
                                int h0_adj,
                                arch::tdm_group0& d0,
                                arch::tdm_group1& d1,
                                arch::tdm_group2& d2) {
        if(h0_adj > 0)
        {
            // Use TDM to zero LDS
            d0.set_lds_addr(in_lds_view(Hring(in_ring0 % cfg.ring_size_p)));
            d0.set_global_addr(in_view.offset());
            d1.tile_dim2 = h0_adj;
            d1.set_tensor_dim1(0);
            d2.set_tensor_dim2(1);
            __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
        }

        const int64_t max_tensor_dim1 = Wpad(in_lds_shape).get() - w0_adj;
        d1.set_tensor_dim1(std::min(W(in_view.shape()).get(), max_tensor_dim1));

        if(cfg.kh - 1 < cfg.chunk_size_p || cfg.chunk_size_p > h0_adj)
        {
            d0.set_lds_addr(
                in_lds_view(Hring((in_ring0 + h0_adj) % cfg.ring_size_p) + Wpad(w0_adj)));
            d0.set_global_addr(in_view.offset());
            d1.tile_dim2 = cfg.chunk_size_p - h0_adj;
            d2.set_tensor_dim2(H(in_view.shape()).get());
            __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
        }
#pragma unroll
        for(int f = 1; f < prefill_factor; ++f)
        {
            auto const p0 = f * cfg.chunk_size_p;
            d0.set_lds_addr(in_lds_view(Hring((in_ring0 + p0) % cfg.ring_size_p) + Wpad(w0_adj)));

            const auto next_view = in_view.moved_by(H(p0 - h0_adj));
            d0.set_global_addr(next_view.offset());
            d1.tile_dim2 = cfg.chunk_size_p;
            d2.set_tensor_dim2(std::max(int64_t{0}, H(next_view.shape()).get()));
            __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
        }
    };
    auto const prefill_dy = [&](auto dy_view,
                                int dy_ring0,
                                arch::tdm_group0& d0,
                                arch::tdm_group1& d1,
                                arch::tdm_group2& d2) {
        d1.set_tensor_dim1(Q(dy_view.shape()).get());
#pragma unroll
        for(int f = 0; f < prefill_factor; ++f)
        {
            auto const p0 = f * cfg.chunk_size_p;
            d0.set_lds_addr(dy_lds_view(Pblock((dy_ring0 + p0) % cfg.ring_size_p)));

            const auto next_view = dy_view.moved_by(P(p0));
            d0.set_global_addr(next_view.offset());
            d2.set_tensor_dim2(std::max(int64_t{0}, P(next_view.shape()).get()));
            __builtin_amdgcn_tensor_load_to_lds(d0.data, d1.data, d2.data, d3.data, d4.data, 0);
        }
    };

    // Input LDS layout for bn::lds_load
    auto const in_lds_layout = [&](int p, bool may_wrap) {
        return [&, p, may_wrap](int r_s_cb, int, int c, int q) {
            const int cb = r_s_cb % reg_tiles_c;
            const int s  = r_s_cb / reg_tiles_c % cfg.kw;
            const int r  = r_s_cb / (reg_tiles_c * cfg.kw);
            c += cb * cfg.wmma_size_c;
            int p_r = p + r;
            if(may_wrap)
                p_r %= cfg.ring_size_p;
            return in_lds_view_part.delta(Hring(p_r) + Wpad(q + s) + Cblock(c));
        };
    };
    // dY LDS layout for bn::lds_load
    auto const dy_lds_layout = [&](int p) {
        return [&, p](int, int kb, int q, int k) {
            k += kb * cfg.wmma_size_k;
            return dy_lds_view_part.delta(Pblock(p) + Qblock(q) + Kblock(k));
        };
    };

    rt_a a;
    rt_b b;

    const int in_prange =
        std::max(prefill_factor, divup(pcount + (cfg.kh - 1), cfg.chunk_size_p)) * cfg.chunk_size_p;
    int in_ring0 = 0;

    const int dy_prange =
        std::max(prefill_factor, divup(pcount, cfg.chunk_size_p)) * cfg.chunk_size_p;
    int dy_ring0 = 0;

    auto const inner_conv = [&]<wgrad_wave_type WaveType>() {
        __builtin_amdgcn_s_wait_tensorcnt(0);
        if constexpr(WaveType == wgrad_wave_type::in_tdm)
            prefill_in(in_view, in_ring0, w0_adj, h0_adj, d0, d1, d2);
        if constexpr(WaveType == wgrad_wave_type::dy_tdm)
            prefill_dy(dy_view, dy_ring0, d0, d1, d2);
        __builtin_amdgcn_s_wait_tensorcnt(0);
        __syncthreads();

        // Split barrier for 8-wave schedule
        if(wave_rank == 1)
        {
            __builtin_amdgcn_s_barrier();
        }
        __builtin_amdgcn_sched_barrier(0);

        auto const w0_w0_adj = [&px](int q_abs) -> std::pair<int, int> {
            int w0           = q_abs - px;
            const int w0_adj = std::max(0, -w0);
            w0 += w0_adj;
            return {w0, w0_adj};
        };


#pragma nounroll
        for(int qchunk = 0; qchunk < qcount; qchunk += block_size_q)
        {
            const auto [w0, w0_adj] = w0_w0_adj(qbegin + qchunk);
            auto in_view_q          = in_view.moved_by(W(w0));
            auto dy_view_q          = dy_view.moved_by(Q(qchunk));
#pragma nounroll
            for(int n = 0; n < tile_size_n; ++n)
            {
                auto in_view_q_n = in_view_q.moved_by(N(n));
                auto dy_view_q_n = dy_view_q.moved_by(N(n));

#pragma nounroll
                for(int pchunk = 0; pchunk < pcount; pchunk += cfg.chunk_size_p)
                {
                    if constexpr(WaveType == wgrad_wave_type::in_tdm)
                    {
                        const int pnext = pchunk + prefill_factor * cfg.chunk_size_p - h0_adj;
                        const int pmax  = make_divisible(pcount, cfg.chunk_size_p) + (cfg.kh - 1);
                        if(pnext < pmax)
                        {
                            auto const p0 = (h0_adj + in_ring0 + pnext) % cfg.ring_size_p;
                            d0.set_lds_addr(in_lds_view(Hring(p0) + Wpad(w0_adj)));
                            d1.tile_dim2   = cfg.chunk_size_p;
                            auto next_view = in_view_q_n.moved_by(H(pnext));
                            if(H(next_view.shape()).get() >= 0)
                            {
                                d0.set_global_addr(next_view.offset());
                                d2.set_tensor_dim2(H(next_view.shape()).get());
                            }
                            else
                            {
                                // We use TDM to fill LDS with zeros
                                d2.set_tensor_dim2(0);
                            }
                            __builtin_amdgcn_tensor_load_to_lds(
                                d0.data, d1.data, d2.data, d3.data, d4.data, 0);
                        }
                        // Start pre-fill in last iteration
                        if(pchunk + cfg.chunk_size_p >= pcount)
                        {
                            const auto in_ring0_next = (in_ring0 + in_prange) % cfg.ring_size_p;
                            if(n + 1 < tile_size_n)
                            {
                                const auto next_view = in_view_q_n.moved_by(N(1));
                                prefill_in(next_view, in_ring0_next, w0_adj, h0_adj, d0, d1, d2);
                            }
                            else if(qchunk + block_size_q < qcount)
                            {
                                const auto [w0, w0_adj] = w0_w0_adj(qbegin + qchunk + block_size_q);
                                const auto next_view    = in_view.moved_by(W(w0));
                                prefill_in(next_view, in_ring0_next, w0_adj, h0_adj, d0, d1, d2);
                            }
                        }
                    }
                    else if constexpr(WaveType == wgrad_wave_type::dy_tdm)
                    {
                        const int pnext = pchunk + prefill_factor * cfg.chunk_size_p;
                        if(pnext < pcount)
                        {
                            d0.set_lds_addr(
                                dy_lds_view(Pblock((dy_ring0 + pnext) % cfg.ring_size_p)));
                            auto next_view = dy_view_q_n.moved_by(P(pnext));
                            d0.set_global_addr(next_view.offset());
                            d2.set_tensor_dim2(P(next_view.shape()).get());
                            __builtin_amdgcn_tensor_load_to_lds(
                                d0.data, d1.data, d2.data, d3.data, d4.data, 0);
                        }
                        // Start pre-fill in last iteration
                        if(pchunk + cfg.chunk_size_p >= pcount)
                        {
                            const auto dy_ring0_next = (dy_ring0 + dy_prange) % cfg.ring_size_p;
                            if(n + 1 < tile_size_n)
                            {
                                prefill_dy(dy_view_q_n.moved_by(N(1)), dy_ring0_next, d0, d1, d2);
                            }
                            else if(qchunk + block_size_q < qcount)
                            {
                                const auto next_view = dy_view_q.moved_by(Q(block_size_q));
                                prefill_dy(next_view, dy_ring0_next, d0, d1, d2);
                            }
                        }
                    }

                    const auto in_ring_offset = (in_ring0 + pchunk) % cfg.ring_size_p;
                    const auto dy_ring_offset = (dy_ring0 + pchunk) % cfg.ring_size_p;

#pragma unroll
                    for(int p0 = 0; p0 < cfg.chunk_size_p; ++p0)
                    {
                        load_tile<arch::ds_load_tr16_b128>(
                            b, dy_lds_view_part.offset(), dy_lds_layout(dy_ring_offset + p0));
                        load_tile<arch::ds_load_tr16_b128>(
                            a,
                            in_lds_view_part.offset(),
                            in_lds_layout(in_ring_offset + p0,
                                          p0 + (cfg.kh - 1) >= cfg.chunk_size_p));
                        __builtin_amdgcn_s_barrier();
                        __builtin_amdgcn_sched_barrier(0);

                        __builtin_amdgcn_s_setprio(1);
                        mma<bn::walk_order::gray, bn::reuse_priority::a>(c_acc, a, b, c_acc);
                        __builtin_amdgcn_s_setprio(0);
                        if(p0 == cfg.chunk_size_p - 1)
                            __builtin_amdgcn_s_wait_tensorcnt(0);
                        __builtin_amdgcn_s_barrier();
                        __builtin_amdgcn_sched_barrier(0);
                    }
                }
                in_ring0 = (in_ring0 + in_prange) % cfg.ring_size_p;
                dy_ring0 = (dy_ring0 + dy_prange) % cfg.ring_size_p;
            }
        }
        if(wave_rank == 0)
        {
            __builtin_amdgcn_s_barrier();
        }
        __builtin_amdgcn_sched_barrier(0);
    };

    switch(wave_type(wave_id_c, wave_id_k))
    {
    case wgrad_wave_type::in_tdm:
        inner_conv.template operator()<wgrad_wave_type::in_tdm>();
        break;
    case wgrad_wave_type::dy_tdm:
        inner_conv.template operator()<wgrad_wave_type::dy_tdm>();
        break;
    default:
        inner_conv.template operator()<wgrad_wave_type::regular>();
        break;
    }

    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();

    float* dw_lds          = reinterpret_cast<float*>(lds);
    const auto dw_lds_view = bn::named_view(dw_lds, dw_lds_shape, dw_lds_stride);
    const auto dw_lds_view_part =
        dw_lds_view.moved_by(Kblock(wave_id_k * reg_stride_k) + Cblock(wave_id_c * reg_stride_c));
    auto const dw_lds_layout = [&](int r_s_cb, int kb, int c, int k) {
        int cb = r_s_cb % reg_tiles_c;
        int s  = r_s_cb / reg_tiles_c % cfg.kw;
        int r  = r_s_cb / (reg_tiles_c * cfg.kw);
        k += kb * cfg.wmma_size_k;
        c += cb * cfg.wmma_size_c;
        return dw_lds_view_part.delta(Kblock(k) + Rblock(r) + Sblock(s) + Cblock(c));
    };
    store_tile<arch::ds_store_b128>(c_acc, dw_lds_view_part.offset(), dw_lds_layout);
    __syncthreads();

    auto dw_view =
        bn::make_named_memref(dw,
                              G(groups) + K(k_per_group) + R(cfg.kh) + S(cfg.kw) + C(c_per_group),
                              bn::mode_order<C, S, R, K, G>{})
            .moved_by(G(tile_g) + K(tile_k) + C(tile_c));

    auto const atomic_add = [&] {
        const auto& dw_shape = dw_view.shape();
        const int k0         = wave_id_k * reg_stride_k;
        const int k1         = std::min(k0 + reg_stride_k, static_cast<int>(K(dw_shape).get()));

        const auto [c0, c1, rs0, num_rs, rs_remainder] = [&]() -> std::array<int, 5> {
            const int c_max = std::min(cfg.tile_size_c, static_cast<int>(C(dw_shape).get()));
            constexpr int total_num_rs = cfg.kh * cfg.kw;

            // Distribute rs iterations to c tiles if tile_size_c is too small for distribution.
            if constexpr(cfg.tile_size_c == arch::wave_size)
            {
                constexpr int rs_chunk_size = total_num_rs / cfg.tiles_c;
                constexpr int rem           = total_num_rs - rs_chunk_size * cfg.tiles_c;
                return {0,
                        c_max,
                        wave_id_c * rs_chunk_size + (wave_id_c < rem ? wave_id_c : rem),
                        rs_chunk_size,
                        wave_id_c < rem ? 1 : 0};
            }
            const int c0 = wave_id_c * reg_stride_c;
            const int c1 = std::min(c0 + reg_stride_c, c_max);
            return {c0, c1, 0, total_num_rs};
        }();

        for(int k = k0; k < k1; ++k)
        {
            for(int c = c0 + lane; c < c1; c += arch::wave_size)
            {
#pragma unroll
                for(int rs = 0; rs < num_rs; ++rs)
                {
                    const int r = (rs0 + rs) / cfg.kw;
                    const int s = (rs0 + rs) % cfg.kw;
                    bn::cascade_atomic_add_f32(
                        dw_view(K(k) + R(r) + S(s) + C(c)),
                        *dw_lds_view(Kblock(k) + Rblock(r) + Sblock(s) + Cblock(c)));
                }
                if(rs_remainder)
                {
                    const int r = (rs0 + num_rs) / cfg.kw;
                    const int s = (rs0 + num_rs) % cfg.kw;
                    bn::cascade_atomic_add_f32(
                        dw_view(K(k) + R(r) + S(s) + C(c)),
                        *dw_lds_view(Kblock(k) + Rblock(r) + Sblock(s) + Cblock(c)));
                }
            }
        }
    };
    atomic_add();
}

template <Config cfg, hipconv::DataType DT>
__launch_bounds__(cfg.tiles() * arch::wave_size, 2) __global__
    void conv2d_direct_wgrad_nhwc_cdna5(const ToType<DT>* __restrict__ in,
                                        const ToType<DT>* __restrict__ dY,
                                        float* __restrict__ dW,
                                        int batch_size,
                                        int groups,
                                        int c_per_group,
                                        int k_per_group,
                                        int hi,
                                        int wi,
                                        int ho,
                                        int wo,
                                        int sy,
                                        int sx,
                                        int dy,
                                        int dx,
                                        int py,
                                        int px)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_tensor_load_to_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_tensorcnt) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_load_tr16_b128_v8i16))
    {
        conv2d_direct_wgrad_cdna5_nhwc_impl<cfg, DT>(in,
                                                     dY,
                                                     dW,
                                                     batch_size,
                                                     groups,
                                                     c_per_group,
                                                     k_per_group,
                                                     hi,
                                                     wi,
                                                     ho,
                                                     wo,
                                                     sy,
                                                     sx,
                                                     dy,
                                                     dx,
                                                     py,
                                                     px);
    }
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const hipconv::ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* /*workspace*/,
                 hipStream_t stream)
{
    HIP_CHECK(hipMemsetAsync(out, 0, ConvSize(par).weight_grad_bytes(), stream));

    auto typed_launch = [&]<hipconv::DataType DT>() {
        using dtype = ToType<DT>;
        conv2d_direct_wgrad_nhwc_cdna5<cfg, DT>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const dtype*>(in),
                static_cast<const dtype*>(wei),
                static_cast<float*>(out),
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                par.h,
                par.w,
                par.p,
                par.q,
                par.stride_h,
                par.stride_w,
                par.dilation_h,
                par.dilation_w,
                par.pad_h,
                par.pad_w);
    };

    if(par.input_type == hipconv::DataType::bf16)
        typed_launch.template operator()<hipconv::DataType::bf16>();
    else
        typed_launch.template operator()<hipconv::DataType::fp16>();
}

class Direct_WgradConvKernel : public DirectConvKernel
{
public:
    constexpr Direct_WgradConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DirectConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "direct_wgrad"; }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    bool is_applicable(const hipconv::ConvParams& par) const override
    {
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.weight_type != par.input_type || par.output_grad_type() != par.input_type)
            return false;
        if(par.weight_grad_type != DataType::fp32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        // reject depthwise
        if(par.k == par.c && par.channels_per_group() == 1)
            return false;
        return true;
    }

    bool is_valid_config(const hipconv::ConvParams& par) const override
    {
        if(par.kh != cfg_.kh)
            return false;
        if(par.kw != cfg_.kw)
            return false;
        return true;
    }

    LaunchParams get_launch_params(const hipconv::ConvParams& par) const override
    {
        constexpr auto NUM_WGP = 256; // hardcoded for now...

        const auto blocks_c = divup(par.channels_per_group(), cfg_.tile_size_c);
        const auto blocks_k = divup(par.filters_per_group(), cfg_.tile_size_k);
        const auto blocks_g = par.groups;
        const auto blocks_split =
            std::min(par.n, std::max(1, NUM_WGP / (blocks_c * blocks_k * blocks_g)));

        LaunchParams launch;
        launch.grid       = dim3(blocks_c, blocks_k * blocks_g, blocks_split);
        launch.block_size = dim3(cfg_.tiles() * arch::wave_size, 1, 1);
        return launch;
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna5::direct_wgrad

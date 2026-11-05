#pragma once

// Tiles with the reduction split in the workgroup: X(128) × K(128) four ways, (C/2, C(64)/2), and
// X(128) × K(256) two ways, (C(64)/2).
// Layout and ordering: docs/algorithms/details/direct/direct-flex-csplit.md. Parenthesized section
// names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "lds_layout.h"
#include "offset_table.h"
#include "output_writer.h"
#include "tile_order.h"
#include "traits.h"
#include "bunnies_cdna4.hpp"
#include "detail.h"
#include "packed_ops.h"

#include <hip/hip_runtime.h>
#include <climits>
#include <cstdint>

namespace hipconv::cdna4::direct_flex
{

template <class T>
__device__ void direct_flex_csplit_impl(LayerPars pars,
                                        const typename T::dtype* __restrict__ in,
                                        const typename T::dtype* __restrict__ wei,
                                        typename T::dtype* __restrict__ out)
{
    using dtype          = typename T::dtype;
    using arch           = bunnies::arch_cdna4;
    using rt_in          = typename T::rt_in;
    using rt_wgt         = typename T::rt_wgt;
    using rt_acc         = typename T::rt_acc;
    constexpr Config cfg = T::cfg;
    constexpr int N      = T::N;
    constexpr int HALF   = T::HALF;
    constexpr int HALF_K = T::HALF_K;
    constexpr int ELEM   = T::elem_bytes;
    constexpr bool DGRAD = T::DGRAD;
    // Staging differs by split.
    //
    // Four ways: each group stages its own chunks in two stages. Two ways: both groups read every
    // chunk from one of three stages and each stages half of it (alternate passes of each
    // half-tile), group 0 a step ahead of the reads and group 1 two.
    constexpr bool SHARED  = cfg.csplit == 2;
    constexpr int STAGES   = SHARED ? 3 : 2;
    constexpr int G_THR    = cfg.threads() / 2; // a wave group's threads
    constexpr int GRANS    = T::granules;
    constexpr int ROWS     = G_THR / GRANS; // rows a group stages per pass
    constexpr int IN_RNDS  = HALF / ROWS / (SHARED ? 2 : 1);
    constexpr int W_RNDS   = HALF_K / ROWS / (SHARED ? 2 : 1);
    constexpr int IN_EL    = HALF * C_CHUNK;
    constexpr int W_EL     = HALF_K * C_CHUNK;
    constexpr int CB       = C_CHUNK * ELEM;       // a chunk's bytes along a fibre
    constexpr int CB_STEP  = SHARED ? CB : 2 * CB; // between a group's chunks
    constexpr int K_STEPS  = Config::k_steps();
    constexpr bool ODD     = T::CHUNKS == Chunks::Odd; // an odd count of a group's chunks
    constexpr int STAGE_EL = SHARED ? 2 * (IN_EL + W_EL) : 2 * 2 * (IN_EL + W_EL);
    static_assert(IN_RNDS >= 1 && W_RNDS >= 1 && N % 2 == 1 && (!SHARED || N % 3 == 0));
    static_assert(SHARED || (HALF_K == HALF && IN_RNDS == 2));

    // One object, so the reduction at the tile's end reuses all of it.
    __shared__ dtype stage_mem[STAGES * STAGE_EL];
    __shared__ typename OffsetTable<T>::Tab in_tab;
    constexpr int TAB_ROW = OffsetTable<T>::COLS * 4;

    KernelArgs<false> args;
    const TileOrder<T> order(args);
    OffsetTable<T> table(args, in_tab);
    const int tid   = threadIdx.x;
    const int wave  = bunnies::wave_id();
    const int group = wave / 4;                         // four ways: chunk parity; two: C(32) half
    const int slice = SHARED ? group : wave % 4 / 2;    // C(32) half of the chunk
    const int xi    = SHARED ? wave % 4 / 2 : wave % 2; // X(32) blocks of each X half
    const int ki    = SHARED ? wave % 2 : 0;            // K(64) blocks of each K half
    auto in_group = [&](int i) { return static_cast<unsigned>(bunnies::pin_sgpr(wave)) / 4 == i; };

    // Half-tile h of stage s: inputs 0 and 1, weights 2 and 3.
    auto half_buf = [&](int s, int h) -> dtype* {
        if constexpr(SHARED)
            return stage_mem + s * STAGE_EL + (h < 2 ? h * IN_EL : 2 * IN_EL + (h - 2) * W_EL);
        else
            return stage_mem + ((s * 4 + h) * 2 + group) * IN_EL;
    };

    // A group's staging thread keeps one slot column and two rows ROWS apart (Operand staging).
    const int gt         = tid % G_THR;
    const int row0       = gt / GRANS;
    const int gran       = detail::granule_for_slot(row0, gt % GRANS);
    const int gran_off   = gran * GRANULE * ELEM;
    const int wave_slot0 = __builtin_amdgcn_readfirstlane(wave % 4 * WAVE_SIZE);
    // LDS slot of each staging round, hoisted out of the loop: the memory phases are issue-bound.
    auto pass_in = [&](int ii) { return SHARED ? group : ii; };
    auto pass_w  = [&](int ii) { return SHARED ? group + 2 * ii : ii; };
    int in_slot[IN_RNDS], w_slot[W_RNDS];
#pragma unroll
    for(int ii = 0; ii < IN_RNDS; ++ii)
        in_slot[ii] = bunnies::pin_sgpr((wave_slot0 + pass_in(ii) * G_THR) * GRANULE);
#pragma unroll
    for(int ii = 0; ii < W_RNDS; ++ii)
        w_slot[ii] = bunnies::pin_sgpr((wave_slot0 + pass_w(ii) * G_THR) * GRANULE);

    const int filter_elems = pars.kh * pars.kw * (DGRAD ? pars.k : pars.c);
    int wgt_voff[2][W_RNDS];
#pragma unroll
    for(int h = 0; h < 2; ++h)
#pragma unroll
        for(int ii = 0; ii < W_RNDS; ++ii)
        {
            if constexpr(DGRAD)
            {
                // Dgrad weights stage C rows of K columns (Weights in KRSC).
                constexpr int SLOTS = HALF_K / GRANULE;
                const int row       = gt / SLOTS + pass_w(ii) * (G_THR / SLOTS);
                const int slot      = gt % SLOTS;
                const int col       = detail::dgrad_group(slot / 2, row, HALF_K / MFMA_N) * MFMA_N +
                                (slot % 2) * GRANULE;
                wgt_voff[h][ii] = (row * filter_elems + h * HALF_K + col) * ELEM;
            }
            else
            {
                const int row   = row0 + pass_w(ii) * ROWS;
                wgt_voff[h][ii] = ((h * HALF_K + row) * filter_elems + gran * GRANULE) * ELEM;
            }
        }

    auto wgt_rsrc_of = [&](int k0) {
        if constexpr(DGRAD)
            return arch::make_buffer(wei + k0, static_cast<int64_t>(pars.wei_elems) - k0);
        return arch::make_buffer(wei + static_cast<int64_t>(k0) * filter_elems,
                                 static_cast<int64_t>(pars.k - k0) * filter_elems);
    };
    // Async: the compiler inserts no vmcnt wait before reads of stage_mem; the waits it inserted
    // for buffer_load_lds doubled the kernel's time.
    auto dma = [](arch::buffer_t rsrc, dtype* lds, int voff, int soff) {
        arch::buffer_load_async_lds<16>::load(rsrc, lds, voff, soff);
    };

    const OperandReader<T> reader(xi, ki);
    rt_in in_reg[2][K_STEPS];
    rt_wgt wgt_reg[2][K_STEPS];
    rt_acc acc[2][2];

    arch::buffer_t in_rsrc = arch::make_buffer(in, 0), wgt_rsrc = wgt_rsrc_of(0);
    int tab_buf = 0;
    // A thread's table entries for a position: input half h, round ii.
    //
    // ent_next holds the entries of the step a group copies next, read a step early so its copies
    // issue first in their memory phase.
    int ent[2][IN_RNDS], ent_next[2][IN_RNDS];
    auto read_entries = [&](int q, int(&ent)[2][IN_RNDS]) {
        using lds_int    = __attribute__((address_space(3))) int;
        const char* base = reinterpret_cast<const char*>(&in_tab[0][0][0]) + tab_buf + q * TAB_ROW;
#pragma unroll
        for(int h = 0; h < 2; ++h)
#pragma unroll
            for(int ii = 0; ii < IN_RNDS; ++ii)
                ent[h][ii] =
                    *(volatile lds_int*)(base + (h * HALF + row0 + pass_in(ii) * ROWS) * 4);
    };
    auto load_wgt = [&](int s, int h, int wsoff) {
#pragma unroll
        for(int ii = 0; ii < W_RNDS; ++ii)
            dma(wgt_rsrc, half_buf(s, 2 + h) + w_slot[ii], wgt_voff[h][ii], wsoff);
    };
    auto load_in = [&](int s, int h, int cb) {
#pragma unroll
        for(int ii = 0; ii < IN_RNDS; ++ii)
            dma(in_rsrc, half_buf(s, h) + in_slot[ii], ent[h][ii] + gran_off, cb);
    };
    auto wsoff_of = [&](int q, int cb) {
        return pars.seq[q].wgt_off + (DGRAD ? cb * filter_elems : cb);
    };

    // One step on stage S, its copies interleaved with its reads.
    //
    // The compiler cannot tell the copies' LDS writes from the reads, so it keeps this order and
    // the reads issue while the copies queue. With ahead, step q's table entries are read for the
    // next step's copies.
    auto run_step = [&]<int S>(auto&& copy, bool ahead, int q) {
        copy.template operator()<0>();
        if(ahead)
            read_entries(q, ent_next);
        reader.read_in(reinterpret_cast<const char*>(half_buf(S, 0)), in_reg[0]);
        copy.template operator()<1>();
        reader.read_in(reinterpret_cast<const char*>(half_buf(S, 1)), in_reg[1]);
        copy.template operator()<2>();
        reader.read_wgt(reinterpret_cast<const char*>(half_buf(S, 2)), wgt_reg[0]);
        copy.template operator()<3>();
        reader.read_wgt(reinterpret_cast<const char*>(half_buf(S, 3)), wgt_reg[1]);
        arch::s_wait_lgkmcnt<0>();
        if(ahead)
        {
#pragma unroll
            for(int h = 0; h < 2; ++h)
#pragma unroll
                for(int ii = 0; ii < IN_RNDS; ++ii)
                    ent[h][ii] = ent_next[h][ii];
        }
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        static_for<2>([&]<int I>() {
            static_for<2>(
                [&]<int J>() { bunnies::mma(acc[I][J], wgt_reg[J][0], in_reg[I][0], acc[I][J]); });
        });
        bunnies::pin(acc[0][0], acc[0][1]);
        bunnies::pin(acc[1][0], acc[1][1]);
        __builtin_amdgcn_s_setprio(0);
        // A wave's copies retire here, before any wave's memory phase reads them.
        arch::s_wait_vmcnt<0>();
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    // Step P of a sweep on stage S, the group's chunk at cb bytes; LAST: the tile's last sweep.
    auto step = [&]<int P, int S, bool LAST>(int cb) {
        constexpr bool COPY  = !(LAST && P + 1 >= N); // step P + 1 lies in the tile
        constexpr bool AHEAD = !(LAST && P + 2 >= N); // and step P + 2
        constexpr int Q1     = (P + 1) % N;
        constexpr int Q2     = (P + 2) % N;
        const int cb1        = P + 1 < N ? cb : cb + CB_STEP;
        const int wsoff      = wsoff_of(Q1, cb1);
        auto copy            = [&]<int H>() {
            if constexpr(COPY)
            {
                if constexpr(H < 2)
                    load_wgt(S ^ 1, H, wsoff);
                else
                    load_in(S ^ 1, H - 2, cb1);
            }
        };
        run_step.template operator()<S>(copy, AHEAD, Q2);
    };
    // Split two ways: step P reads stage P % 3; group g copies its half of step P + 1 + g.
    auto step3 = [&]<int P, bool LAST>(int cb) {
        constexpr int S = P % 3;
        const bool g1   = in_group(1);
        // C0, C1: the copies of group 0 and group 1 lie in the tile; A0, A1: the entries each reads
        // ahead.
        constexpr bool C0  = !(LAST && P + 1 >= N);
        constexpr bool C1  = !(LAST && P + 2 >= N);
        constexpr bool A0  = !(LAST && P + 2 >= N);
        constexpr bool A1  = !(LAST && P + 3 >= N);
        const bool copy_on = C0 && C1 ? true : (g1 ? C1 : C0);
        const bool ahead   = A0 && A1 ? true : (g1 ? A1 : A0);
        const int cb_d     = g1 ? (P + 2 < N ? cb : cb + CB) : (P + 1 < N ? cb : cb + CB);
        const int wsoff    = g1 ? wsoff_of((P + 2) % N, cb_d) : wsoff_of((P + 1) % N, cb_d);
        const int st       = g1 ? (P + 2) % 3 : (P + 1) % 3;
        auto copy          = [&]<int H>() {
            if(copy_on)
            {
                if constexpr(H < 2)
                    load_wgt(st, H, wsoff);
                else
                    load_in(st, H - 2, cb_d);
            }
        };
        run_step.template operator()<S>(copy, ahead, g1 ? (P + 3) % N : (P + 2) % N);
    };
    auto sweep = [&]<int S0, bool LAST>(int cb) {
        if constexpr(SHARED)
            static_for<N>([&]<int P>() { step3.template operator()<P, LAST>(cb); });
        else
            static_for<N>([&]<int P>() { step.template operator()<P, (S0 + P) % 2, LAST>(cb); });
    };

    using f4 = bunnies::floatx4;
    // The tile's output buffer size, clamped by OutputWriter::span.
    auto out_span = [&](const Tile& t) {
        return OutputWriter<T>::span(pars.out_elems - static_cast<int64_t>(t.x0) * pars.k);
    };
    // Stores 8 reduced channels of one pixel; an out-of-range pixel or channel gets the offset past
    // the buffer, which the range check drops.
    auto store8 = [&](const Tile& t, int out_elems, const f4& lo, const f4& hi, int pix, int kk) {
        const auto out_rsrc =
            arch::make_buffer(out + static_cast<int64_t>(t.x0) * pars.k, out_elems);
        using d4 = packed_vec_t<dtype, 4>;
        struct
        {
            d4 a, b;
        } v{packed_convert<dtype>(lo), packed_convert<dtype>(hi)};
        const int voff =
            t.x0 + pix < pars.rows && kk < pars.k ? (pix * pars.k + kk) * ELEM : out_elems * ELEM;
        arch::buffer_store<16>::store(out_rsrc, &v, voff, 0);
    };

    // Split two ways: waves w and w + 4 combine their partial sums through stage_mem and store the
    // tile, one (X half, K half) block at a time.
    auto reduce_pair_store = [&](const Tile& t) {
        auto* xm               = reinterpret_cast<f4*>(stage_mem);
        const int lane         = bunnies::volatile_lane_id();
        constexpr int ROW_U    = cfg.wave_k() / 4; // a row's 16-byte units: K(64) fp32
        constexpr int REGION_U = cfg.wave_x() * ROW_U;
        const int pair         = wave % 4;
        auto unit              = [&](int region, int row, int u) {
            return region * REGION_U + row * ROW_U + (u ^ (row & (ROW_U - 1)));
        };
        const int out_elems = out_span(t);
        static_for<2>([&]<int I>() {
            static_for<2>([&]<int J>() {
                // lane is volatile_lane_id: with lane_id, LICM hoisted the addresses out of the
                // X-tile loop and the dgrad kernels spilled them.
                bunnies::store_tile<arch::ds_store_b128>(
                    acc[I][J], xm, [&](int kb, int xb, int k, int x) {
                    return unit(pair * 2 + group, xb * MFMA_M + x, (kb * MFMA_N + k) / 4);
                }, lane);
                arch::s_wait_lgkmcnt<0>();
                __builtin_amdgcn_s_barrier();
                // Each wave of the pair stores 16 of the block's 32 rows, 8 pixels per instruction.
                static_for<cfg.wave_x() / 16>([&]<int NS>() {
                    const int row = group * (cfg.wave_x() / 2) + NS * 8 + lane / 8;
                    const int u   = (lane % 8) * 2; // 8 channels: two units
                    const f4 lo   = xm[unit(pair * 2, row, u)] + xm[unit(pair * 2 + 1, row, u)];
                    const f4 hi =
                        xm[unit(pair * 2, row, u + 1)] + xm[unit(pair * 2 + 1, row, u + 1)];
                    store8(t,
                           out_elems,
                           lo,
                           hi,
                           I * HALF + xi * cfg.wave_x() + row,
                           t.k0 + J * HALF_K + ki * cfg.wave_k() + (lane % 8) * 8);
                });
                arch::s_wait_lgkmcnt<0>();
                __builtin_amdgcn_s_barrier();
            });
        });
    };

    // Sums the partial sums of the four (group, slice) pairs through stage_mem, stores the tile.
    auto reduce_store = [&](const Tile& t) {
        auto* xm               = reinterpret_cast<f4*>(stage_mem);
        const int lane         = bunnies::volatile_lane_id();
        constexpr int ROW_U    = HALF_K * 2 / 4; // a row's 16-byte units: K(128) fp32
        constexpr int REGION_U = 32 * ROW_U;     // X(32) rows
        const int q            = 2 * group + slice;
        auto unit              = [&](int region, int row, int u) {
            return region * REGION_U + row * ROW_U + (u ^ (row & 15));
        };
        const int out_elems = out_span(t);
        static_for<2>([&]<int I>() {
            static_for<2>([&]<int J>() {
                bunnies::store_tile<arch::ds_store_b128>(
                    acc[I][J], xm, [&](int kb, int xb, int k, int x) {
                    return unit(xi * 4 + q, xb * MFMA_M + x, (J * HALF_K + kb * MFMA_N + k) / 4);
                });
            });
            arch::s_wait_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();
            static_for<2>([&]<int NS>() {
                const int row = q * 8 + NS * 4 + lane / 16; // within the X(32) band
                const int u   = (lane % 16) * 2;            // 8 channels: two units
                f4 lo{}, hi{};
                static_for<4>([&]<int QQ>() {
                    lo += xm[unit(xi * 4 + QQ, row, u)];
                    hi += xm[unit(xi * 4 + QQ, row, u + 1)];
                });
                store8(t, out_elems, lo, hi, I * HALF + xi * 32 + row, t.k0 + (lane % 16) * 8);
            });
            arch::s_wait_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();
        });
    };

    // A group's chunks: all of them split two ways, alternate ones split four ways.
    const int g_chunks = SHARED ? pars.chunks : pars.chunks / 2;
    for(int ti = order.static_tile(0); ti >= 0; ti = order.static_next(ti))
    {
        const Tile t    = order.decode(ti);
        const Base base = OffsetTable<T>::origin(args, t);
        in_rsrc         = arch::make_buffer(in + base.x_base, INT32_MAX / ELEM);
        wgt_rsrc        = wgt_rsrc_of(t.k0);
        // Group 0's 256 threads cover the table's X(128) pixels and both position parities.
        if(in_group(0))
            table.build_next([&](int) { return t; }, &base);
        else
            table.skip_next();
        table.next_tile();
        for(auto& row : acc)
            for(auto& a : row)
                a = rt_acc{};
        arch::s_wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
        // Step 0: position 0 of the group's first chunk; step 1's entries.
        const int cb0 = SHARED ? 0 : group * CB;
        read_entries(0, ent);
        load_wgt(0, 0, wsoff_of(0, cb0));
        load_wgt(0, 1, wsoff_of(0, cb0));
        load_in(0, 0, cb0);
        load_in(0, 1, cb0);
        if constexpr(SHARED)
        {
            // Group 1's half of step 1, which it would have copied in step -1.
            if(in_group(1))
            {
                read_entries(1, ent);
                load_wgt(1, 0, wsoff_of(1, cb0));
                load_wgt(1, 1, wsoff_of(1, cb0));
                load_in(1, 0, cb0);
                load_in(1, 1, cb0);
            }
            read_entries(in_group(1) ? 2 : 1, ent);
        }
        else
            read_entries(1, ent);
        arch::s_wait_vmcnt<0>();
        arch::s_wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
        // Wave group 1 runs a phase behind group 0 (Ping-pong schedule).
        if(in_group(1))
            __builtin_amdgcn_s_barrier();

        int cb = cb0;
        for(int j = ODD ? 1 : 2; j < g_chunks; j += 2)
        {
            sweep.template operator()<0, false>(cb);
            cb += CB_STEP;
            sweep.template operator()<1, false>(cb);
            cb += CB_STEP;
        }
        if constexpr(ODD)
            sweep.template operator()<0, true>(cb);
        else
        {
            sweep.template operator()<0, false>(cb);
            cb += CB_STEP;
            sweep.template operator()<1, true>(cb);
        }
        // Group 0 waits out group 1's last compute phase.
        if(in_group(0))
            __builtin_amdgcn_s_barrier();
        if constexpr(SHARED)
            reduce_pair_store(t);
        else
            reduce_store(t);
        tab_buf ^= TAB_ROWS * TAB_ROW;
    }
}

} // namespace hipconv::cdna4::direct_flex

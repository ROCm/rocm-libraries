#pragma once

// Dense convolution as GEMM tiles over the unfolded output positions.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "csplit_kernel.h"
#include "config.h"
#include "dgrad_plan.h"
#include "dynamic_order.h"
#include "kernel_args.h"
#include "layer_pars.h"
#include "lds_layout.h"
#include "load_stream.h"
#include "offset_table.h"
#include "output_writer.h"
#include "split_partials.h"
#include "tile_order.h"
#include "traits.h"
#include "bunnies_cdna4.hpp"
#include "config_desc.h"
#include "config_table.h"
#include "direct_conv_kernel.h"
#include "detail.h"
#include "hip_util.h"

#include "launch_params.h"
#include "magic_division.h"
#include "persistent_grid.h"
#include "types.h"
#include "hipconv/conv_params.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hipconv::cdna4::direct_flex
{

// Sweep flags: a position's last chunk, the next sweep's, and the filter tile's end (Schedule).
struct SweepKind
{
    bool last;
    bool next_last;
    bool end;
};

// One variant's kernel; Traits documents the template parameters.
template <Config cfg,
          DataType DT,
          bool DGRAD,
          bool NARROW_K,
          bool C_TAIL,
          Chunks CHUNKS,
          int N,
          bool ONE_FTILE,
          bool SPLIT,
          bool DEPTH>
__device__ __forceinline__ void direct_flex_impl(const Hot hot,
                                                 LayerPars pars,
                                                 const ToType<DT>* __restrict__ in,
                                                 const ToType<DT>* __restrict__ wei,
                                                 ToType<DT>* __restrict__ out)
{
    using T = Traits<cfg, DT, DGRAD, NARROW_K, C_TAIL, CHUNKS, N, ONE_FTILE, SPLIT, DEPTH>;
    constexpr bool PHASED    = T::PHASED;
    constexpr bool ONE_STEP  = T::ONE_STEP;
    constexpr bool ONE_CHUNK = T::ONE_CHUNK;
    constexpr int RING       = T::RING;
    using dtype              = typename T::dtype;
    using arch               = bunnies::arch_cdna4;
    using dtl                = arch::buffer_load_lds<16>;
    constexpr int BLOCK_X16  = T::BLOCK_X16;
    constexpr int BLOCK_K16  = T::BLOCK_K16;
    constexpr int HALF       = T::HALF;
    constexpr int HALF_K     = T::HALF_K;
    using mat_out            = typename T::mat_out;
    using rt_wgt             = typename T::rt_wgt;
    using rt_in              = typename T::rt_in;
    using rt_acc             = typename T::rt_acc;
    using rt_out             = typename T::rt_out;
    constexpr int granules   = T::granules;
    constexpr int elem_bytes = T::elem_bytes;

    // One array per half-tile and stage, so the waitcnt pass tells reads from in-flight DMA.
    __shared__ dtype a00[HALF * C_CHUNK];
    __shared__ dtype a01[HALF * C_CHUNK];
    __shared__ dtype a10[HALF * C_CHUNK];
    __shared__ dtype a11[HALF * C_CHUNK];
    __shared__ dtype b00[HALF * C_CHUNK];
    __shared__ dtype b01[HALF * C_CHUNK];
    __shared__ dtype b10[HALF * C_CHUNK];
    __shared__ dtype b11[HALF * C_CHUNK];

    auto in_half = [&]<int H>(int stage) -> dtype* {
        if constexpr(H == 0)
            return stage ? a10 : a00;
        else
            return stage ? a11 : a01;
    };
    auto wgt_half = [&]<int H>(int stage) -> dtype* {
        if constexpr(H == 0)
            return stage ? b10 : b00;
        else
            return stage ? b11 : b01;
    };

    const auto front = KernelArgs<PHASED>::fetch_front();

    // Persistent grid with XCD-contiguous runs of the tile order (Grid).
    // readfirstlane keeps the bases scalar: in VGPRs, the descriptors waterfall every access.
    constexpr int XCDS = persistent::NUM_XCD;
    KernelArgs<PHASED> args;
    using kernarg_pars = typename KernelArgs<PHASED>::Pars;
    using kernarg_seg  = typename KernelArgs<PHASED>::Seg;
    const TileOrder<T> order(args);

    // The current tile and the next RING - 1, as tile indices or mailbox entries; -1 past the last.
    int ring[RING];
    order.static_tiles(ring, &hot);
    // The run's end and stride in SGPRs: static_next(t) rederives them at every X tile.
    const auto run    = order.own_run(&hot);
    const int run_end = run.first + run.len;
    const int run_wgs = run.wgs;
    if(ring[0] < 0)
        return;

    // Tile `ahead` past the current; past the last, the last again, so loads keep valid addresses.
    auto tile_ahead = [&](int ahead) {
        if constexpr(ONE_STEP)
        {
            int t = ring[0];
#pragma unroll
            for(int i = 1; i < RING; ++i)
                t = i <= ahead && ring[i] >= 0 ? ring[i] : t;
            return order.decode(t & ~STOLEN);
        }
        int t = ahead == 0 ? ring[0] : ahead == 1 ? ring[1] : ring[2];
        t     = t < 0 ? ring[1] : t;
        return order.decode((t < 0 ? ring[0] : t) & ~STOLEN);
    };
    // Whether tile `ahead` past the current exists; only conv3d on positions inner reads it, so it
    // has no one-step form.
    auto has_tile = [&](int ahead) {
        return (ahead == 0 ? ring[0] : ahead == 1 ? ring[1] : ring[2]) >= 0;
    };

    int piece    = 0;
    int wei_skip = 0;
    if constexpr(SPLIT)
    {
        int per = pars.split_tiles;
        if constexpr(PHASED)
            per = order.split_tiles();
        piece = __builtin_amdgcn_readfirstlane(ring[0] / per);
        ring[0] -= piece * per;
        in += piece * pars.chunks * C_CHUNK;
        if constexpr(DGRAD)
            wei_skip = piece * pars.chunks * C_CHUNK * pars.kh * pars.kw * pars.k;
        else
            wei += piece * pars.chunks * C_CHUNK;
        wei += wei_skip;
    }

    const DynamicOrder<T> dyn(args, order);
    // Read once: each dyn.enabled() call reloads the kernel argument and waits for it.
    // INT_MIN when enabled; ANDed with ~tile index, its sign bit is the claim condition.
    const int dyn_bit = dyn.enabled() ? INT_MIN : 0;
    const int tid     = threadIdx.x;

    // Streams keep tile bases and form descriptors per load: two bases take half the SGPRs.
    auto in_base_of = [&](const Tile& t) { return in + OffsetTable<T>::origin(args, t).x_base; };
    auto in_rsrc    = [&](const dtype* base) {
        return arch::make_buffer(base, INT32_MAX / elem_bytes);
    };
    // tile_ahead(0), decoded from the preloaded arguments so that it runs before wait_front.
    const Tile tile0 = order.decode(ring[0] & ~STOLEN, &hot);
    // tile_cur and tile_nxt keep the crossing's decode for the stores, with positions inner.
    // Several filter tiles decode again: holding the tiles added K(256) spills.
    Tile tile_cur = tile0, tile_nxt = tile0;
    auto stored = [&] {
        if constexpr(ONE_FTILE && N > 0)
            return tile_cur;
        return tile_ahead(0);
    };
    // Weight descriptor from the tile's first channel: lane offsets stay fixed (Weight fibres).
    auto wgt_rsrc_of = [&](int k0) {
        kernarg_pars& kp = args.cold();
        if constexpr(DGRAD)
            return arch::make_buffer(wei + k0, static_cast<int64_t>(kp.wei_elems) - wei_skip - k0);
        const int filter_elems = kp.kh * kp.kw * kp.c;
        return arch::make_buffer(wei + static_cast<int64_t>(k0) * filter_elems,
                                 static_cast<int64_t>(kp.k - k0) * filter_elems);
    };
    KernelArgs<PHASED>::wait_front(front);
    arch::buffer_t wgt_rsrc = wgt_rsrc_of(tile0.k0);
    // Computed once for both input halves and the first table: cold() blocks CSE of origin() calls.
    const Base base0     = OffsetTable<T>::origin(args, tile0, &hot);
    const dtype* in_b[2] = {in + base0.x_base, in + base0.x_base};
    // A separate in_b[1]: one base shared by both halves pushed K256 kernels past 256 VGPRs.
    in_b[1] = bunnies::pin_sgpr(in_b[1]);
    // Past any descriptor's NUM_RECORDS.
    const int in_oob  = NULL_OFFSET;
    const int wgt_oob = NULL_OFFSET;

    // A thread keeps one slot column and walks rows a workgroup apart; round i is half i / 2.
    constexpr int rows_per_pass = cfg.threads() / granules;
    static_assert(2 * rows_per_pass == HALF, "a half-tile stages in two rounds");
    const int stage_slot_g = tid % granules;
    const int stage_row0   = tid / granules;
    const int stage_g      = detail::granule_for_slot(stage_row0, stage_slot_g);
    const int wave_slot0   = __builtin_amdgcn_readfirstlane(tid - (tid % WAVE_SIZE));

    // Input offset tables, double-buffered: a fibre offset per position and pixel (Input fibres).
    __shared__ typename OffsetTable<T>::Tab in_tab;
    OffsetTable<T> table(args, in_tab);
    // Added to each table entry in 32 bits: a 64-bit add over both held a second VGPR in the loop.
    const int gran_off = stage_g * GRANULE * elem_bytes;

    constexpr bool SKIP = DEPTH && N > 0;

    // Weight staging: a piece is one output channel's fibre (Weight fibres, Weights in KRSC).
    const int filter_elems = pars.kh * pars.kw * (DGRAD ? pars.k : pars.c);
    // A chunk's step along the weight rows: C_CHUNK channels of a fibre, or in dgrad C_CHUNK rows.
    const int wgt_chunk_bytes  = DGRAD ? C_CHUNK * filter_elems * elem_bytes : C_CHUNK * elem_bytes;
    constexpr int DG_ROW_SLOTS = HALF / GRANULE;
    auto dg_row = [&](int ii) { return tid / DG_ROW_SLOTS + ii * (cfg.threads() / DG_ROW_SLOTS); };
    int wgt_voff[cfg.wgt_rounds()];
#pragma unroll
    for(int i = 0; i < cfg.wgt_rounds(); ++i)
    {
        if constexpr(DGRAD)
        {
            const int row  = dg_row(i % 2);
            const int slot = tid % DG_ROW_SLOTS;
            const int col  = detail::dgrad_group(slot / 2, row) * MFMA_N + (slot % 2) * GRANULE;
            wgt_voff[i] =
                col < HALF_K ? (row * filter_elems + (i / 2) * HALF_K + col) * elem_bytes : wgt_oob;
        }
        else
        {
            const int row = stage_row0 + (i % 2) * rows_per_pass;
            wgt_voff[i] =
                row < HALF_K
                    ? (((i / 2) * HALF_K + row) * filter_elems + stage_g * GRANULE) * elem_bytes
                    : wgt_oob;
        }
    }

    // Wave (xi, ki) owns X rows xi * wave_x + [0, wave_x) and K rows ki * wave_k + [0, wave_k).
    // Waves 0-3 and 4-7, one of each per SIMD, form the ping-pong's wave groups.
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE_SIZE);
    const int xi   = wave / cfg.waves_k;
    const int ki   = wave % cfg.waves_k;
    // Compare the wave index at each use.
    // A hoisted comparison holds a 64-bit lane mask; four such masks spilled.
    auto in_group = [&](int i) { return static_cast<unsigned>(bunnies::pin_sgpr(wave)) / 4 == i; };
    auto wave0    = [&] { return bunnies::pin_sgpr(wave) == 0; };

    const StreamWalk<T> walk(args, pars);
    const int taps = walk.taps();
    // The tile a crossing stream targets: the next, or with one-step tiles the one after.
    int cross_ahead = 1;
    auto crossing   = [&] {
        if constexpr(ONE_STEP)
            return tile_ahead(cross_ahead);
        return tile_ahead(1);
    };

    // A lane whose granule lies past C reads nothing in a position's last chunk.
    const bool tail_ok = (SPLIT && piece + 1 < pars.pieces) || stage_g * GRANULE < pars.c_last;
    auto tail          = [&](int last_chunk, int voffset, int oob) {
        if constexpr(C_TAIL)
            return last_chunk && !tail_ok ? oob : voffset;
        else
            return voffset;
    };
    // Dgrad weight rows past C lie past the descriptor's end, so the range check zeroes them.
    auto wgt_tail = [&](int last_chunk, int voffset) {
        if constexpr(DGRAD)
            return voffset;
        else
            return tail(last_chunk, voffset, wgt_oob);
    };

    Stream in_st[2]  = {walk.start(), walk.start()};
    Stream wgt_st[2] = {walk.start(), walk.start()};

    // Half 1's table entries sit TAB_HALF past half 0's, as stage_row0 < rows_per_pass.
    constexpr int TAB_HALF = 2 * rows_per_pass * 4;
    const int tab_lane     = OffsetTable<T>::col(stage_row0) * 4;
    int tab_off[2];
    // Input streams start a position one memory phase before its first load (Ping-pong schedule).
    auto locate = [&]<int H>() {
        const Stream& st = in_st[H];
        const int flip   = walk.next_flip(st);
        tab_off[H]       = ((st.next.in_row ^ flip) & 0xffff) + tab_lane;
        // Keeps the offset in this phase: the compiler would sink it to the read.
        tab_off[H] = bunnies::pin_vgpr(tab_off[H]);
    };
    bunnies::int32x2 tab_next;
    auto load_offsets = [&]<int H>() {
        // LDS-typed from in_tab, so no vmcnt wait; volatile, so read here (Ping-pong schedule).
        using lds_int32x2 = __attribute__((address_space(3))) bunnies::int32x2;
        const char* p = reinterpret_cast<const char*>(&in_tab[0][0][0]) + tab_off[H] + H * TAB_HALF;
        tab_next      = *(volatile lds_int32x2*)p;
    };
    auto start_in = [&]<int H>() {
        Stream& st = in_st[H];
        bool crossed;
        st.in_row = walk.advance(st, crossed).in_row;
        if(crossed)
            in_b[H] = in_base_of(crossing());
        st.voff[0] = tab_next[0] + gran_off;
        st.voff[1] = tab_next[1] + gran_off;
        // Keeps the offsets in this phase: the compiler would sink them into the input phase.
        asm volatile("" : "+v"(st.voff[0]), "+v"(st.voff[1]));
    };

    // Loads input half H into stage `stage`, zero-filling pixels outside the image.
    // NEW marks a position's first load, whose offsets start_in formed.
    auto fetch_in = [&]<int H, bool NEW>(int stage, int last_chunk) {
        Stream& st = in_st[H];
        if constexpr(NEW)
        {
            st.soffset = 0;
        }
        else
        {
            st.soffset += C_CHUNK * elem_bytes;
        }
#pragma unroll
        for(int ii = 0; ii < 2; ++ii)
        {
            const int slot0 = wave_slot0 + ii * cfg.threads();
            dtl::load(in_rsrc(in_b[H]),
                      &in_half.template operator()<H>(stage)[slot0 * GRANULE],
                      tail(last_chunk, st.voff[ii], in_oob),
                      st.soffset);
        }
    };

    auto fetch_wgt = [&]<int H, bool NEW>(int stage, int last_chunk) {
        Stream& st = wgt_st[H];
        if constexpr(NEW)
        {
            bool crossed;
            st.soffset = walk.advance(st, crossed).wgt_off;
            if(H == 0 && crossed)
                wgt_rsrc = wgt_rsrc_of(crossing().k0);
        }
        else
        {
            st.soffset += wgt_chunk_bytes;
        }
#pragma unroll
        for(int ii = 0; ii < 2; ++ii)
        {
            const int slot0 = wave_slot0 + ii * cfg.threads();
            dtl::load(wgt_rsrc,
                      &wgt_half.template operator()<H>(stage)[slot0 * GRANULE],
                      wgt_tail(last_chunk, wgt_voff[2 * H + ii]),
                      st.soffset);
        }
    };

    // Operands for one block, one register tile per C(32) half of the step.
    rt_in in_reg[Config::k_steps()];
    rt_wgt wgt_0_reg[Config::k_steps()];
    rt_wgt wgt_1_reg[Config::k_steps()];
    rt_acc acc[2][2];
    auto zero_acc = [&]() {
        for(auto& row : acc)
            for(auto& a : row)
                a = rt_acc{};
    };
    // With ZERO_FIRST an X tile's first step zeroes the accumulators through SrcC
    // (docs/algorithms/details/direct/direct-flex-registers.md).
    constexpr bool ZERO_FIRST = ONE_FTILE && !(C_TAIL && CHUNKS == Chunks::Even);
    if constexpr(!ZERO_FIRST)
        zero_acc();

    const OperandReader<T> reader(xi, ki);
    auto read_in = [&]<int H>(int stage) {
        reader.read_in(reinterpret_cast<const char*>(in_half.template operator()<H>(stage)),
                       in_reg);
    };
    auto read_wgt = [&]<int H>(int stage, rt_wgt(&dst)[Config::k_steps()]) {
        reader.read_wgt(reinterpret_cast<const char*>(wgt_half.template operator()<H>(stage)), dst);
    };

    auto compute = [&]<int I, int J, bool ZERO>(rt_wgt(&b)[Config::k_steps()]) {
        static_for<Config::k_steps()>([&]<int CS>() {
            if constexpr(ZERO && CS == 0)
                bunnies::mma(acc[I][J], b[CS], in_reg[CS]);
            else
                bunnies::mma(acc[I][J], b[CS], in_reg[CS], acc[I][J]);
        });
    };

    // Waits for the next memory phase's loads and this phase's LDS reads (Ping-pong schedule).
    auto end_memory = [&]() {
        arch::s_wait_vmcnt<8>();
        arch::s_wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // Keeps a memory phase's LDS reads ahead of its global loads, which the compiler would swap.
    auto end_reads = [] { __builtin_amdgcn_sched_barrier(0); };
    // A compute phase runs the two blocks of input half I, one per weight half.
    // TILE_END: an X tile's last compute phase; group 1 skips its barrier and runs on into the
    // epilogue phase.
    auto run_compute = [&]<int I, bool ZERO = false, bool TILE_END = false>() {
        __builtin_amdgcn_s_setprio(1);
        // Keeps the accumulators live across X tiles; defined anew per tile, they spill in K256.
        if constexpr(ZERO)
            bunnies::pin(acc[I][0], acc[I][1]);
        compute.template operator()<I, 0, ZERO>(wgt_0_reg);
        compute.template operator()<I, 1, ZERO>(wgt_1_reg);
        // Without this the compiler moves a position transition's MFMAs into the next memory phase.
        bunnies::pin(acc[I][0], acc[I][1]);
        __builtin_amdgcn_s_setprio(0);
        if constexpr(TILE_END)
        {
            if(in_group(0))
                __builtin_amdgcn_s_barrier();
        }
        else
            __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    // One iteration on stage s0 (Ping-pong schedule); a literal s0 avoids a vmcnt(0) per iteration.
    auto step = [&]<bool NEW1, bool NEW2>(int s0, int last1, int last2) {
        const int s1 = s0 ^ 1;

        // Builds before the operand reads: no stream reads this buffer and the operands are dead.
        if constexpr(NEW1 && !ONE_CHUNK)
        {
            if((in_st[1].in_row >> 16) != 0)
            {
                table.build_next(tile_ahead);
            }
            end_reads();
        }
        // One chunk: half 0's start of a filter tile triggers the build, as with one-step tiles.
        // A build at half 1's start would race with half 0 reading the next filter tile's table.
        if constexpr(ONE_CHUNK)
        {
            if((in_st[0].next.in_row >> 16) != 0)
            {
                table.build_next(tile_ahead);
            }
            end_reads();
        }
        if constexpr(NEW2)
        {
            locate.template operator()<0>();
            load_offsets.template operator()<0>();
        }
        end_reads();
        read_in.template operator()<0>(s0);
        read_wgt.template operator()<0>(s0, wgt_0_reg);
        read_wgt.template operator()<1>(s0, wgt_1_reg);
        end_reads();
        fetch_in.template operator()<1, NEW1>(s1, last1);
        end_reads();
        if constexpr(NEW2)
            start_in.template operator()<0>();
        end_memory();
        run_compute.template operator()<0>();

        if constexpr(NEW2)
        {
            locate.template operator()<1>();
            load_offsets.template operator()<1>();
        }
        end_reads();
        read_in.template operator()<1>(s0);
        end_reads();
        fetch_wgt.template operator()<0, NEW2>(s0, last2);
        fetch_in.template operator()<0, NEW2>(s0, last2);
        fetch_wgt.template operator()<1, NEW2>(s0, last2);
        end_reads();
        if constexpr(NEW2)
            start_in.template operator()<1>();
        end_memory();
        run_compute.template operator()<1>();
    };

    // A filter position's chunk loop for an even chunk count, from stage 0.
    auto position_even = [&]() {
        for(int c = 0; c + 2 < pars.chunks; c += 2)
        {
            step.template operator()<false, false>(0, 0, c + 2 == pars.chunks - 1);
            step.template operator()<false, false>(1, 0, c + 3 == pars.chunks - 1);
        }
        step.template operator()<false, true>(0, 1, 0);
        step.template operator()<true, false>(1, 0, pars.chunks == 2);
    };

    // Odd chunk counts: a runtime stage keeps one copy of the position code; literal ones spill.
    int stage         = 0;
    auto position_odd = [&]() {
        for(int c = 0; c + 2 < pars.chunks; ++c)
        {
            step.template operator()<false, false>(
                bunnies::pin_sgpr(stage), 0, c + 2 == pars.chunks - 1);
            stage ^= 1;
        }
        step.template operator()<false, true>(bunnies::pin_sgpr(stage), 1, 0);
        stage ^= 1;
        step.template operator()<true, false>(bunnies::pin_sgpr(stage), 0, 0);
        stage ^= 1;
    };

    // Positions inner: chunks outer, the filter tile's positions unrolled as a sweep (Schedule).
    int chunk_off  = 0;                   // the sweep's chunk in bytes
    int f_tile     = 0;                   // the sweep's filter tile
    int wbase      = 0;                   // the tile's first entry in pars.seq
    int tab_buf    = 0;                   // this filter tile's table buffer, in bytes
    int in_soff[2] = {0, 0};              // the chunk offset of each input stream's next loads
    int w_t2       = pars.seq[0].wgt_off; // the weight offset of the position two steps ahead
    int wnext      = 0;     // with live filter tiles, the next one's first entry in pars.seq
    bool ftile_end = false; // and whether this one ends its X tile
    constexpr int CHUNK_BYTES = C_CHUNK * elem_bytes;

    // The chunk offset of the step D past step P of a sweep of kind K.
    auto chunk_at = [&]<int P, int D, SweepKind K>() {
        if constexpr(P + D < N)
            return chunk_off;
        else if constexpr(K.end)
            return 0;
        else
            return chunk_off + CHUNK_BYTES;
    };

    // Input half H's entries for a target at row Q, in this tile's buffer or the next's.
    using lds_int32x2 = __attribute__((address_space(3))) bunnies::int32x2;
    auto read_entries = [&]<int H, int Q, bool NEXT_TILE>() {
        const char* p = reinterpret_cast<const char*>(&in_tab[0][0][0]) + tab_lane +
                        (NEXT_TILE ? tab_buf ^ TAB_BYTES : tab_buf) + Q * TAB_ROW_BYTES +
                        H * TAB_HALF;
        tab_next = *(volatile lds_int32x2*)p;
    };
    // The stream's offsets from the entries, kept in this phase as start_in's.
    auto form_entries = [&]<int H>(int last, int soff) {
        in_st[H].voff[0] = tail(last, tab_next[0] + gran_off, in_oob);
        in_st[H].voff[1] = tail(last, tab_next[1] + gran_off, in_oob);
        in_soff[H]       = soff;
        asm volatile("" : "+v"(in_st[H].voff[0]), "+v"(in_st[H].voff[1]));
    };
    auto fetch_in_s = [&]<int H>(int stage) {
#pragma unroll
        for(int ii = 0; ii < 2; ++ii)
        {
            const int slot0 = wave_slot0 + ii * cfg.threads();
            dtl::load(in_rsrc(in_b[H]),
                      &in_half.template operator()<H>(stage)[slot0 * GRANULE],
                      in_st[H].voff[ii],
                      in_soff[H]);
        }
    };
    // Lanes keep half 0's weight offsets; half 1's lie wgt_half_bytes on, in soffset.
    const int wgt_half_bytes = HALF_K * (DGRAD ? 1 : filter_elems) * elem_bytes;
    auto fetch_wgt_s         = [&]<int H>(int stage, int soff, int last) {
#pragma unroll
        for(int ii = 0; ii < 2; ++ii)
        {
            const int slot0 = wave_slot0 + ii * cfg.threads();
            dtl::load(wgt_rsrc,
                      &wgt_half.template operator()<H>(stage)[slot0 * GRANULE],
                      wgt_tail(last, wgt_voff[ii]),
                      soff + H * wgt_half_bytes);
        }
    };

    // Step P of a sweep of kind K on stage S0 (Schedule); with ZERO its first MFMAs take SrcC 0.
    auto sstep = [&]<int P, SweepKind K, int S0, bool ZERO>() {
        constexpr int S1     = S0 ^ 1;
        constexpr int Q2     = (P + 2) % N;
        constexpr bool NEXT2 = P + 2 >= N && K.end;
        constexpr bool LAST2 = P + 2 < N ? K.last : K.next_last;
        constexpr int Q3     = (P + 3) % N;
        constexpr bool NEXT3 = P + 3 >= N && K.end;
        const int chunk2     = chunk_at.template operator()<P, 2, K>();
        bool cross           = false;
        Tile following{};
        const dtype* next_b = nullptr;
        if constexpr(SKIP && K.end && P == N - 3)
        {
            ftile_end = f_tile + 1 == table.cur_hi();
            wnext     = (ftile_end ? table.next_lo() : f_tile + 1) * N;
        }
        if constexpr(K.end && P == N - 2)
        {
            // Selected rather than branched: a branch here cost the K256 sweeps their spare VGPRs.
            if constexpr(SKIP)
                cross = ftile_end;
            else
                cross = ONE_FTILE || f_tile == args.cold().n_ftiles - 1;
            following = tile_ahead(1);
            tile_nxt  = following;
            next_b    = in_base_of(following);
        }

        if constexpr(K.end && P == N - 1 && !ONE_FTILE)
        {
            table.build_next(tile_ahead, nullptr, has_tile);
            end_reads();
        }
        read_entries.template operator()<0, Q2, NEXT2>();
        end_reads();
        read_in.template operator()<0>(S0);
        read_wgt.template operator()<0>(S0, wgt_0_reg);
        read_wgt.template operator()<1>(S0, wgt_1_reg);
        end_reads();
        fetch_in_s.template operator()<1>(S1);
        end_reads();
        form_entries.template operator()<0>(LAST2, chunk2);
        in_b[0] = cross ? next_b : in_b[0];
        end_memory();
        run_compute.template operator()<0, ZERO>();

        read_entries.template operator()<1, Q2, NEXT2>();
        end_reads();
        read_in.template operator()<1>(S0);
        end_reads();
        // The chunk's 128 bytes along a fibre are C_CHUNK whole weight rows in dgrad.
        const int wsoff = w_t2 + (DGRAD ? chunk2 * filter_elems : chunk2);
        wgt_rsrc        = cross ? wgt_rsrc_of(following.k0) : wgt_rsrc;
        fetch_wgt_s.template operator()<0>(S0, wsoff, LAST2);
        fetch_in_s.template operator()<0>(S0);
        fetch_wgt_s.template operator()<1>(S0, wsoff, LAST2);
        end_reads();
        form_entries.template operator()<1>(LAST2, chunk2);
        in_b[1] = cross ? next_b : in_b[1];
        if constexpr(SKIP)
            w_t2 = bunnies::pin_sgpr(pars.seq[(NEXT3 ? wnext : wbase) + Q3].wgt_off);
        else
            w_t2 = bunnies::pin_sgpr(pars.seq[wbase + (NEXT3 ? N : 0) + Q3].wgt_off);
        end_memory();
        run_compute.template operator()<1, ZERO, ONE_FTILE && K.end && P == N - 1>();
    };

    // A sweep starting at stage S0; for odd N the next one starts at the other stage.
    // FIRST: the X tile's first sweep, whose first step zeroes the accumulators.
    auto sweep = [&]<SweepKind K, int S0, bool FIRST = false>() {
        static_for<N>(
            [&]<int P>() { sstep.template operator()<P, K, S0 ^ (P & 1), FIRST && P == 0>(); });
    };

    // A filter tile's sweeps as straight-line code: a branch between sweeps spilled K256.
    constexpr SweepKind MID{false, false, false};
    constexpr SweepKind PRE{false, true, false};
    constexpr SweepKind LAST{true, false, true};
    constexpr SweepKind ONE{true, true, true};
    constexpr int FLIP         = N % 2; // a sweep of odd length moves the next to the other stage
    constexpr bool ODD_CHUNKS  = CHUNKS != Chunks::Even;
    constexpr bool PRE_SWEEP   = C_TAIL && CHUNKS != Chunks::One;
    constexpr SweepKind ENDING = C_TAIL && CHUNKS == Chunks::One ? ONE : LAST;
    // chunks - 1 - PRE_SWEEP middle sweeps, and their parity.
    constexpr int MIDS_ODD = ((ODD_CHUNKS ? 1 : 0) + 1 + (PRE_SWEEP ? 1 : 0)) % 2;
    // f: the filter tile within its X tile; g: within the workgroup's sequence.
    auto filter_tile = [&]<int S0>(int f, int g) {
        f_tile  = f;
        tab_buf = (g & 1) * TAB_BYTES;
        if constexpr(SKIP)
            wbase = f * N;
        const int mids        = pars.chunks - 1 - (PRE_SWEEP ? 1 : 0);
        constexpr int S1      = MIDS_ODD ? S0 ^ FLIP : S0;
        constexpr int S2      = PRE_SWEEP ? S1 ^ FLIP : S1;
        constexpr bool ONE_SW = CHUNKS == Chunks::One;
        if constexpr(ZERO_FIRST)
        {
            // Peels the first sweep and inlines every middle sweep, which fails to compile outlined
            // (docs/algorithms/details/direct/direct-flex-schedule.md).
            int m = MIDS_ODD;
            if constexpr(MIDS_ODD)
            {
                [[clang::always_inline]] sweep.template operator()<MID, S0, true>();
                chunk_off += CHUNK_BYTES;
            }
            else if constexpr(!ONE_SW)
            {
                [[clang::always_inline]] sweep.template operator()<MID, S0, true>();
                chunk_off += CHUNK_BYTES;
                [[clang::always_inline]] sweep.template operator()<MID, S0 ^ FLIP>();
                chunk_off += CHUNK_BYTES;
                m = 2;
            }
            for(; m < mids; m += 2)
            {
                [[clang::always_inline]] sweep.template operator()<MID, S1>();
                chunk_off += CHUNK_BYTES;
                [[clang::always_inline]] sweep.template operator()<MID, S1 ^ FLIP>();
                chunk_off += CHUNK_BYTES;
            }
            if constexpr(PRE_SWEEP)
            {
                sweep.template operator()<PRE, S1>();
                chunk_off += CHUNK_BYTES;
            }
            sweep.template operator()<ENDING, S2, ONE_SW>();
        }
        else
        {
            for(int m = MIDS_ODD; m < mids; m += 2)
            {
                sweep.template operator()<MID, S0>();
                chunk_off += CHUNK_BYTES;
                sweep.template operator()<MID, S0 ^ FLIP>();
                chunk_off += CHUNK_BYTES;
            }
            if constexpr(MIDS_ODD)
            {
                sweep.template operator()<MID, S0>();
                chunk_off += CHUNK_BYTES;
            }
            if constexpr(PRE_SWEEP)
            {
                sweep.template operator()<PRE, S1>();
                chunk_off += CHUNK_BYTES;
            }
            sweep.template operator()<ENDING, S2>();
        }
        chunk_off = 0;
        if constexpr(!SKIP)
            wbase = ONE_FTILE || f + 1 == pars.n_ftiles ? 0 : wbase + N;
    };

    const OutputWriter<T> writer(args, out, xi, ki, acc);

    // Epilogue phase between X tiles (Epilogue); its pairing with the other wave group is in
    // docs/algorithms/details/direct/direct-flex-schedule.md.
    // Without ZERO_FIRST it zeroes the accumulators.
    auto epilogue_phase = [&]() {
        __builtin_amdgcn_sched_barrier(0);
        // Evaluated at each use.
        // A flag held across the tile's stores occupies a 64-bit lane mask.
        auto dynamic       = [&] { return bunnies::pin_sgpr(dyn_bit) != 0; };
        auto claiming      = [&] { return (bunnies::pin_sgpr(dyn_bit) & ~ring[RING - 1]) < 0; };
        auto stealing      = [&] { return (bunnies::pin_sgpr(ring[RING - 1]) & STOLEN) != 0; };
        int mail           = -1;
        uint32_t front_old = 0;
        if(claiming())
        {
            mail = __scoped_atomic_load_n(
                dyn.mailbox(table.parity() ^ 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if(wave0() && !stealing())
                front_old = dyn.add_lane0(dyn.counter(order.run_x()), 1);
        }
        __builtin_amdgcn_sched_barrier(0);
        writer.store_tile(stored());
        tile_cur = tile_nxt;
        // Group 0 builds the whole table: group 1's stores finish after group 0's.
        if constexpr(ONE_FTILE)
        {
            if(in_group(0))
                table.template build_next<true>(tile_ahead);
            else
                table.skip_next();
        }
        if constexpr(!ZERO_FIRST)
            zero_acc();
        if(claiming() && wave0())
            dyn.post(table.parity(), dyn.claim(front_old, stealing()));
        ring[0] = ring[1];
        ring[1] = ring[2];
        if constexpr(ONE_STEP)
            ring[2] = ring[3];
        // One branch: with an if-else the compiler branched on a constant 64-bit lane mask.
        int t_new = order.static_next(
            ring[RING - 1], bunnies::pin_sgpr(run_end), bunnies::pin_sgpr(run_wgs));
        if(dynamic())
            t_new = claiming() ? __builtin_amdgcn_readfirstlane(mail) : -1;
        ring[RING - 1] = t_new;
        if constexpr(NARROW_K)
            arch::s_wait_vmcnt<0>();
        // With one filter tile, group 0 skips this barrier so both groups run their epilogue
        // phases at once.
        if constexpr(ONE_FTILE)
        {
            if(in_group(1))
                __builtin_amdgcn_s_barrier();
        }
        else
            __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // After an X tile's last step: the epilogue phase, or false past the workgroup's last tile.
    auto next_tile = [&]() {
        if(ring[1] < 0)
            return false;
        epilogue_phase();
        table.next_tile();
        return true;
    };

    // A tile without taps stores from the zeroed accumulators (Zero tiles).
    if constexpr(PHASED)
        if(taps == 0)
        {
            const bool dynamic = dyn.enabled();
            int* next          = &in_tab[0][0][0];
            int t              = ring[0];
#pragma nounroll
            for(int i = 1; t >= 0; ++i)
            {
                const bool claiming = dynamic && i >= STATIC_TILES;
                const bool stealing = (t & STOLEN) != 0;
                uint32_t front_old  = 0;
                if(claiming && wave == 0 && !stealing)
                    front_old = dyn.add_lane0(dyn.counter(order.run_x()), 1);
                writer.store_tile(order.decode(t & ~STOLEN));
                if(!claiming)
                {
                    t = !dynamic ? order.static_next(t) : i == 1 ? ring[1] : ring[2];
                    continue;
                }
                if(wave == 0)
                    next[i & 1] = dyn.claim(front_old, stealing);
                __syncthreads();
                t = __builtin_amdgcn_readfirstlane(next[i & 1]);
            }
            return;
        }

    // Prologue: the loads of steps -2 and -1, so step 0 starts in the steady state.
    const bool claim3 = dyn.enabled() && wave == 0;
    uint32_t front3   = 0;
    if(claim3)
        front3 = dyn.add_lane0(dyn.counter(order.run_x()), ONE_STEP ? 2 : 1);
    // Step 0's weight loads overlap the table build. Step 1's weight loads stay after step 0's
    // input loads: the vmcnt waits below and in step 0 count on step 0's loads being the oldest.
    if constexpr(N > 0 && !SKIP)
    {
        fetch_wgt_s.template operator()<0>(0, w_t2, ONE_CHUNK);
        fetch_wgt_s.template operator()<1>(0, w_t2, ONE_CHUNK);
    }
    // The first table is tile 0's, so tile0 replaces tile_ahead.
    // Not with several filter tiles, where reusing tile0 added K256 spills.
    if constexpr(ONE_FTILE)
        table.build_next([&](int) { return tile0; }, &base0);
    else
        table.build_next(tile_ahead, nullptr, has_tile);
    if constexpr(N == 0)
        table.build_next(tile_ahead);
    if(claim3)
    {
        if constexpr(ONE_STEP)
        {
            const int t3               = dyn.claim(front3, false);
            in_tab[0][TAB_ROWS - 1][0] = t3;
            dyn.post(STATIC_TILES % MAILBOX_SLOTS, dyn.claim(front3 + 1, (t3 & STOLEN) != 0));
        }
        else
            dyn.post(STATIC_TILES % MAILBOX_SLOTS, dyn.claim(front3, false));
    }
    arch::s_wait_lgkmcnt<0>();
    __builtin_amdgcn_s_barrier();
    if constexpr(ONE_STEP)
        if(dyn.enabled())
            ring[3] = __builtin_amdgcn_readfirstlane(in_tab[0][TAB_ROWS - 1][0]);
    if constexpr(N > 0)
    {
        // Steps 0 and 1: positions 0 and 1 of filter tile 0's first chunk.
        // With live filter tiles, the X tile's first.
        int seq0 = 0;
        if constexpr(SKIP)
        {
            seq0 = table.enter_tile(tile_ahead) * N;
            w_t2 = pars.seq[seq0].wgt_off;
        }
        const int w1 = pars.seq[seq0 + 1].wgt_off;
        if constexpr(SKIP)
            fetch_wgt_s.template operator()<0>(0, w_t2, ONE_CHUNK);
        read_entries.template operator()<0, 0, false>();
        form_entries.template operator()<0>(ONE_CHUNK, 0);
        fetch_in_s.template operator()<0>(0);
        if constexpr(SKIP)
            fetch_wgt_s.template operator()<1>(0, w_t2, ONE_CHUNK);
        read_entries.template operator()<1, 0, false>();
        form_entries.template operator()<1>(ONE_CHUNK, 0);
        fetch_in_s.template operator()<1>(0);
        fetch_wgt_s.template operator()<0>(1, w1, ONE_CHUNK);
        read_entries.template operator()<0, 1, false>();
        form_entries.template operator()<0>(ONE_CHUNK, 0);
        fetch_in_s.template operator()<0>(1);
        fetch_wgt_s.template operator()<1>(1, w1, ONE_CHUNK);
        read_entries.template operator()<1, 1, false>();
        form_entries.template operator()<1>(ONE_CHUNK, 0);
        w_t2 = pars.seq[seq0 + 2].wgt_off;
        // Builds the second table under the latency of steps 0 and 1's loads; its first reader
        // follows the barrier below.
        table.build_next(tile_ahead, nullptr, has_tile);
    }
    else
    {
        const int last1 = ONE_CHUNK || pars.chunks == 2;
        auto start      = [&]<int H>() {
            locate.template operator()<H>();
            load_offsets.template operator()<H>();
            start_in.template operator()<H>();
        };
        fetch_wgt.template operator()<0, true>(0, ONE_CHUNK);
        start.template operator()<0>();
        fetch_in.template operator()<0, true>(0, ONE_CHUNK);
        fetch_wgt.template operator()<1, true>(0, ONE_CHUNK);
        start.template operator()<1>();
        fetch_in.template operator()<1, true>(0, ONE_CHUNK);
        fetch_wgt.template operator()<0, ONE_CHUNK>(1, last1);
        if constexpr(ONE_CHUNK)
        {
            start.template operator()<0>();
        }
        fetch_in.template operator()<0, ONE_CHUNK>(1, last1);
        fetch_wgt.template operator()<1, ONE_CHUNK>(1, last1);
        // Step 0's input half 1 load starts a position, as phase B of a step -1 would have.
        if constexpr(ONE_CHUNK)
        {
            start.template operator()<1>();
        }
        // When half 0's second start began filter tile 1, filter tile 2's table reuses filter tile
        // 0's buffer once read (One-step tiles).
        if constexpr(ONE_CHUNK)
        {
            if((in_st[0].in_row >> 16) != 0)
            {
                arch::s_wait_lgkmcnt<0>();
                __builtin_amdgcn_s_barrier();
                table.build_next(tile_ahead);
            }
        }
        if constexpr(ONE_STEP)
        {
            if(taps == 1)
            {
                cross_ahead = 2;
            }
        }
    }
    arch::s_wait_vmcnt<8>();
    arch::s_wait_lgkmcnt<0>();
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Wave group 1 takes an extra barrier, running one phase behind wave group 0.
    if(in_group(1))
        __builtin_amdgcn_s_barrier();

    if constexpr(N > 0)
    {
        // f: the filter tile within the X tile; fs: within the workgroup's sequence.
        int f = 0, fs = 0;
        if constexpr(SKIP)
            f = table.cur_lo();
        auto next_ftile = [&]() {
            ++fs;
            if constexpr(SKIP)
            {
                if(++f < table.cur_hi())
                    return true;
                const bool more = next_tile();
                f               = more ? table.enter_tile(tile_ahead) : 0;
                return more;
            }
            if constexpr(!ONE_FTILE)
                if(++f < pars.n_ftiles)
                    return true;
            f = 0;
            return next_tile();
        };
        if constexpr(FLIP && ODD_CHUNKS)
            for(;;)
            {
                filter_tile.template operator()<0>(f, fs);
                if(!next_ftile())
                    break;
                filter_tile.template operator()<1>(f, fs);
                if(!next_ftile())
                    break;
            }
        else
            do
                filter_tile.template operator()<0>(f, fs);
            while(next_ftile());
    }
    else if constexpr(ONE_CHUNK)
    {
        // Every iteration is a filter position of its own, and may start an X tile.
        auto tile_steps = [&]() {
            if constexpr(PHASED)
                return taps;
            return pars.steps;
        };
        for(int left = tile_steps();;)
        {
            step.template operator()<true, true>(0, 1, 1);
            if(--left == 0)
            {
                if(!next_tile())
                    break;
                left = tile_steps();
            }
            step.template operator()<true, true>(1, 1, 1);
            if(--left == 0)
            {
                if(!next_tile())
                    break;
                left = tile_steps();
            }
        }
    }
    else if constexpr(CHUNKS == Chunks::Even)
    {
        do
            for(int pos = 0; pos < taps; ++pos)
                position_even();
        while(next_tile());
    }
    else
    {
        do
            for(int pos = 0; pos < taps; ++pos)
                position_odd();
        while(next_tile());
    }

    // Wave group 0 takes the barrier group 1 took before the loop, so both take the same number.
    // With one filter tile, group 1 instead skips the barrier after its last compute phase.
    if constexpr(!ONE_FTILE)
        if(in_group(0))
            __builtin_amdgcn_s_barrier();
    arch::s_wait_vmcnt<0>();
    if constexpr(SPLIT)
        writer.store_partial(stored(), [&](kernarg_pars& kp) {
            if constexpr(PHASED)
                return order.static_tile(0) / order.split_tiles();
            return order.static_tile(0) / kp.split_tiles;
        });
    else
        writer.store_tile(stored());
}

// Compiled for every registry target; the builtin gate leaves an empty kernel off gfx950.
template <Config cfg,
          DataType DT,
          bool DGRAD,
          bool NARROW_K,
          bool C_TAIL,
          Chunks CHUNKS,
          int N,
          bool ONE_FTILE,
          bool SPLIT,
          bool DEPTH>
__global__
__launch_bounds__(cfg.threads()) void direct_flex_cdna4(int tiles,
                                                        int blocks,
                                                        int in_full,
                                                        int k_group,
                                                        int x_tiles,
                                                        int rebase_in,
                                                        uint32_t group_mul,
                                                        uint32_t group_shift,
                                                        uint32_t kg_mul,
                                                        uint32_t kg_shift,
                                                        uint32_t q_mul,
                                                        uint32_t q_shift,
                                                        uint32_t p_mul,
                                                        uint32_t p_shift,
                                                        LayerPars pars,
                                                        const ToType<DT>* __restrict__ in,
                                                        const ToType<DT>* __restrict__ w,
                                                        ToType<DT>* __restrict__ out)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16))
    {
        const Hot hot{tiles,
                      blocks,
                      in_full,
                      k_group,
                      x_tiles,
                      rebase_in,
                      magic_of(group_mul, group_shift),
                      magic_of(kg_mul, kg_shift),
                      magic_of(q_mul, q_shift),
                      magic_of(p_mul, p_shift)};
        if constexpr(cfg.csplit != 1)
            direct_flex_csplit_impl<
                Traits<cfg, DT, DGRAD, NARROW_K, C_TAIL, CHUNKS, N, ONE_FTILE, SPLIT, DEPTH>>(
                pars, in, w, out);
        else
            direct_flex_impl<cfg, DT, DGRAD, NARROW_K, C_TAIL, CHUNKS, N, ONE_FTILE, SPLIT, DEPTH>(
                hot, pars, in, w, out);
    }
}

// A csplit tile's throughput index where it ranks first: above 1, direct_l1's highest.
constexpr float CSPLIT_FIRST = 1.25f;
// The two-way split's throughput index where it ranks first; below CSPLIT_FIRST, so the four-way
// split ranks first where both would.
constexpr float CSPLIT2_FIRST = 1.24f;

// The largest C at which the four-way split ranks first with at most half as many tiles as CUs.
//
// Above it, configs that split the reduction across workgroups ran faster on so few tiles.
constexpr int CSPLIT_SMALL_GRID_C = 10 * C_CHUNK;

// Whether the tile's C split fits the layer: the four-way split (csplit 4) takes C(64) chunks in
// pairs, so C must be a multiple of 2 * C_CHUNK.
inline bool csplit_fits(const Config& cfg, const ConvParams& layer)
{
    return cfg.csplit != 4 || as_fprop(layer).c % (2 * C_CHUNK) == 0;
}

// A tile's throughput relative to K256 at full use, measured (Tile set).
//
// The split tiles rank below the X(256) tiles unless CSPLIT_FIRST or CSPLIT2_FIRST applies; they
// are fastest on few layers (csplit_kernel.h).
constexpr float tile_rate(const Config& cfg)
{
    if(cfg.csplit != 1)
        return 0.5f;
    switch(cfg.k_tile())
    {
    case 256:
        return 1.0f;
    case 192:
        return 0.92f;
    default:
        return 0.70f;
    }
}

// One workgroup per CU, or per tile if fewer; a split runs one per piece (Grid).
inline int grid_size(const Config& cfg, const ConvParams& par, int cu_count, int pieces)
{
    const int tiles = tile_count(cfg, par);
    return pieces > 1 ? tiles * pieces : std::min(tiles, cu_count);
}

inline LaunchParams launch_params(const Config& cfg, int grid)
{
    LaunchParams launch;
    launch.grid                 = dim3(grid, 1, 1);
    launch.block_size           = dim3(cfg.threads(), 1, 1);
    launch.dynamic_shared_bytes = 0;
    return launch;
}

// A split's partials or the dynamic order's counters and mailboxes, never both.
inline size_t get_workspace_size(const Config& cfg, const ConvParams& par, int cu_count)
{
    const int pieces = split_pieces(cfg, par, cu_count);
    if(pieces > 1)
        return partials_bytes(cfg, par, pieces);
    const int grid = grid_size(cfg, par, cu_count, 1);
    return dynamic_order(cfg, par, grid)
               ? CTR_BYTES + static_cast<size_t>(grid) * MAILBOX_SLOTS * sizeof(int)
               : 0;
}

// The variant a layer runs on `tile`.
inline Variant variant_of(const Config& tile, const ConvParams& layer)
{
    const ConvParams par = as_fprop(layer);
    const bool dgrad     = layer.direction == Direction::Dgrad;
    const int sweep      = phased(layer) ? 0 : sweep_of(layer);
    const bool depth     = layer.dims == 3;
    return {tile,
            layer.input_type,
            layer.direction,
            !dgrad && par.k % GRANULE != 0,
            par.c % C_CHUNK != 0,
            sweep,
            !depth && sweep > 0 && par.kh * par.kw == sweep,
            depth};
}

// The split, grid and tile order a layer takes on a device; ConvLaunch keeps it.
//
// docs/algorithms/details/direct/direct-flex-launch.md, "Launch plan".
struct LaunchPlan
{
    ConvParams par; // as_fprop(layer)
    bool phase, split, dyn, one_step;
    int pieces, grid, chunks;
    DgradPlan plan;
    ReducePars rp; // split only
    int reduce_blocks;
    bool needs_workspace() const { return split || dyn; }
};

inline LaunchPlan launch_plan(const Variant& v, const Device& dev, const ConvParams& layer)
{
    const Config& cfg = v.tile;
    if(variant_of(cfg, layer) != v || !csplit_fits(cfg, layer))
        throw std::invalid_argument("direct_flex: the layer does not match the config's variant");
    LaunchPlan p{};
    // A dgrad runs as as_fprop's layer, as phases on positions outer.
    p.par   = as_fprop(layer);
    p.phase = phased(layer);
    if(p.phase)
        p.plan = dgrad_plan(cfg, layer, dev.cu_count);
    p.pieces = p.phase ? p.plan.pieces : split_pieces(cfg, p.par, dev.cu_count);
    p.split  = p.phase ? p.plan.split : p.pieces > 1;
    p.grid   = p.phase ? p.plan.grid : grid_size(cfg, p.par, dev.cu_count, p.pieces);
    p.dyn    = p.phase ? p.plan.dynamic() : !p.split && dynamic_order(cfg, p.par, p.grid);
    // Chunks per tile or piece; single-step tiles take their own kernel (One-step tiles).
    p.chunks = (p.par.c + C_CHUNK - 1) / C_CHUNK / p.pieces;
    if(v.sweep == 0)
    {
        p.one_step = !p.split && p.chunks == 1 && filter_taps(p.par) == 1;
        for(int s = 0; p.phase && !p.split && p.chunks == 1 && s < p.plan.n_segs; ++s)
            p.one_step |= p.plan.segs[s].nr * p.plan.segs[s].nc == 1;
    }
    if(!p.split)
        return p;
    p.rp = p.phase ? reduce_pars(cfg, p.plan, phased_pars(cfg, layer, p.plan, nullptr, nullptr))
                   : contig_reduce_pars(cfg, p.par, p.pieces);
    int64_t n = 0;
    for(int f = 0; f < p.rp.n_phases; ++f)
        n += static_cast<int64_t>(p.rp.phases[f].rows) * ((p.rp.k + 3) / 4);
    p.reduce_blocks = static_cast<int>(std::min<int64_t>((n + REDUCE_THREADS - 1) / REDUCE_THREADS,
                                                         8 * static_cast<int64_t>(dev.cu_count)));
    return p;
}

// A layer reaches only its own variant; launch picks the chunk parity and the split.
template <Variant v>
void launch_impl(const LaunchParams& lp,
                 const Device& dev,
                 const ConvParams& layer,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    constexpr Config cfg  = v.tile;
    constexpr DataType DT = v.type;
    constexpr bool DGRAD  = v.direction == Direction::Dgrad;
    using dtype           = ToType<DT>;
    const auto* kept      = static_cast<const LaunchPlan*>(lp.kernel_state.get());
    std::optional<LaunchPlan> made;
    if(!kept)
        kept = &made.emplace(launch_plan(v, dev, layer));
    const LaunchPlan& lpl = *kept;
    if(lpl.needs_workspace() && !workspace)
        throw HipError(hipErrorInvalidValue, "direct_flex: this launch needs its workspace");
    const ConvParams& par = lpl.par;
    const bool phase      = lpl.phase;
    const DgradPlan& plan = lpl.plan;
    const int pieces      = lpl.pieces;
    const bool split      = lpl.split;
    const int grid        = lpl.grid;
    auto* partials        = split ? static_cast<float*>(workspace) : nullptr;
    auto* ws              = lpl.dyn ? static_cast<uint32_t*>(workspace) : nullptr;
    if(ws)
        HIP_CHECK(hipMemsetAsync(ws, 0, phase ? plan_counter_bytes(plan) : CTR_BYTES, stream));
    auto go = [&]<Chunks CHUNKS>() {
        LayerPars pars;
        if constexpr(DGRAD && v.sweep == 0)
            pars = phased_pars(cfg, layer, plan, ws, partials);
        else
        {
            pars               = layer_pars(cfg, par, v.sweep, pieces, DGRAD);
            pars.blocks        = grid;
            pars.ws            = ws;
            pars.partials      = partials;
            constexpr int XCDS = persistent::NUM_XCD;
            for(int x = 0; x < XCDS; ++x)
            {
                const Share t     = share_of(pars.tiles, XCDS, x);
                pars.dyn_first[x] = t.first + STATIC_TILES * share_of(grid, XCDS, x).len;
                pars.run_end[x]   = t.first + t.len;
            }
        }
        auto kernel = [&] {
            if constexpr(cfg.csplit != 1)
                return direct_flex_cdna4<cfg,
                                         DT,
                                         DGRAD,
                                         false,
                                         false,
                                         CHUNKS,
                                         v.sweep,
                                         true,
                                         false,
                                         false>;
            else if constexpr(CHUNKS == Chunks::OneStep)
                return direct_flex_cdna4<cfg,
                                         DT,
                                         DGRAD,
                                         v.narrow_k,
                                         v.c_tail,
                                         CHUNKS,
                                         0,
                                         false,
                                         false,
                                         v.depth>;
            else
                return split ? direct_flex_cdna4<cfg,
                                                 DT,
                                                 DGRAD,
                                                 v.narrow_k,
                                                 v.c_tail,
                                                 CHUNKS,
                                                 v.sweep,
                                                 v.one_ftile,
                                                 true,
                                                 v.depth>
                             : direct_flex_cdna4<cfg,
                                                 DT,
                                                 DGRAD,
                                                 v.narrow_k,
                                                 v.c_tail,
                                                 CHUNKS,
                                                 v.sweep,
                                                 v.one_ftile,
                                                 false,
                                                 v.depth>;
        }();
        const Hot h = hot_args(pars);
        kernel<<<dim3(grid), lp.block_size, lp.dynamic_shared_bytes, stream>>>(
            h.tiles,
            h.blocks,
            h.in_full,
            h.k_group,
            h.x_tiles,
            h.rebase_in,
            h.group_div.multiplier,
            h.group_div.shift,
            h.kg_div.multiplier,
            h.kg_div.shift,
            h.q_div.multiplier,
            h.q_div.shift,
            h.p_div.multiplier,
            h.p_div.shift,
            pars,
            static_cast<const dtype*>(in),
            static_cast<const dtype*>(wei),
            static_cast<dtype*>(out));
    };
    const int chunks    = lpl.chunks;
    const bool one_step = lpl.one_step;
    if constexpr(cfg.csplit != 1)
    {
        // The parity of a wave group's chunks: all of them split two ways, alternate ones split
        // four ways (csplit_kernel.h).
        if((cfg.csplit == 4 ? chunks / 2 : chunks) % 2 == 0)
            go.template operator()<Chunks::Even>();
        else
            go.template operator()<Chunks::Odd>();
    }
    else if(one_step)
    {
        if constexpr(v.sweep == 0)
            go.template operator()<Chunks::OneStep>();
    }
    else if(chunks == 1)
    {
        // Several filter tiles, no channel tail: Chunks::One compiles to Chunks::Odd's ISA.
        if constexpr(v.sweep > 0 && !v.one_ftile && !v.c_tail)
            go.template operator()<Chunks::Odd>();
        else
            go.template operator()<Chunks::One>();
    }
    else if(chunks % 2 == 0)
        go.template operator()<Chunks::Even>();
    else
        go.template operator()<Chunks::Odd>();
    if(!split)
        return;
    const ReducePars& rp = lpl.rp;
    const int blocks     = lpl.reduce_blocks;
    auto* dst            = static_cast<dtype*>(out);
    if(phase)
        direct_flex_reduce<DT, v.depth, false>
            <<<blocks, REDUCE_THREADS, 0, stream>>>(partials, dst, rp);
    else
        direct_flex_reduce<DT, false, true>
            <<<blocks, REDUCE_THREADS, 0, stream>>>(partials, dst, rp);
}

class DirectFlex_ConvKernel : public DirectConvKernel
{
public:
    constexpr DirectFlex_ConvKernel(const Variant& v, LaunchFn launch_fn)
        : DirectConvKernel(launch_fn)
        , v_(v)
        , cfg_(v.tile)
    {
    }

    std::string_view name() const override { return "direct_flex"; }

    // Conv3d runs on the depth variants (Conv3d).
    bool supports_dims(int dims) const override { return dims == 2 || dims == 3; }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(v_); }

    // Skips DirectConvKernel::is_applicable: any stride and dilation are served.
    bool is_applicable(const Device& dev, const ConvParams& layer) const override
    {
        if(layer.input_type != DataType::fp16 && layer.input_type != DataType::bf16)
            return false;
        if(layer.input_type != layer.weight_type || layer.input_type != layer.output_type)
            return false;
        if(layer.order != TensorOrder::NHWC)
            return false;
        // A conv3d's depth-tap filter rows must fit the position list and kernel arguments.
        const bool depth = layer.dims == 3;
        if(depth && (layer.kd > MAX_FILTER || filter_taps(layer) > MAX_FILTER * MAX_FILTER ||
                     rect_ftiles(layer.kd * layer.kh, layer.kw) > MAX_FTILES))
            return false;
        if(layer.direction == Direction::Dgrad)
        {
            // The weights' rows run over C, whose granules a dgrad loads whole.
            if(layer.c % GRANULE != 0)
                return false;
            // A strided dgrad's phases take taps at the stride; no layer measured dilates too.
            const bool strided = stride_phases(layer) > 1;
            const bool dilated =
                layer.dilation_h > 1 || layer.dilation_w > 1 || (depth && layer.dilation_d > 1);
            if(strided && (dilated || stride_phases(layer) > MAX_PHASES))
                return false;
            if(phased(layer) && !phases_fit(cfg_, layer))
                return false;
        }
        else if(layer.direction != Direction::Fprop)
            return false;
        if(layer.groups != 1)
            return false;
        const ConvParams par = as_fprop(layer);
        // A 16-byte granule must stay inside one pixel's channels.
        if(par.c % GRANULE != 0)
            return false;
        // A pixel's in-image masks hold one bit per filter row and column.
        if(par.kh > MAX_FILTER || par.kw > MAX_FILTER)
            return false;
        // Descriptors rebase per workgroup: one X tile's input rows and the weights need 32 bits.
        const ConvSize sz(par);
        if(phased(layer))
            return sz.weight_bytes() <= INT32_MAX;
        return sz.weight_bytes() <= INT32_MAX && out_rows(par) <= INT32_MAX - 256 &&
               span_input(par) <= INT32_MAX / 2;
    }

    // Each layer runs one variant per tile.
    bool is_valid_config(const Device& /*dev*/, const ConvParams& par) const override
    {
        return csplit_fits(cfg_, par) && variant_of(cfg_, par) == v_;
    }

    LaunchParams get_launch_params(const Device& dev, const ConvParams& par) const override
    {
        auto plan       = std::make_shared<const LaunchPlan>(launch_plan(v_, dev, par));
        LaunchParams lp = launch_params(cfg_, plan->grid);
        lp.kernel_state = std::move(plan);
        return lp;
    }

    size_t get_workspace_size(const Device& dev, const ConvParams& par) const override
    {
        return phased(par) ? plan_workspace(cfg_, par, dev.cu_count)
                           : direct_flex::get_workspace_size(cfg_, as_fprop(par), dev.cu_count);
    }

    // Ranked below the other direct families, scaled by K-tile fill and the tile's rate.
    float get_weighted_throughput_index(const Device& dev, const ConvParams& layer) const override
    {
        const ConvParams par = as_fprop(layer);
        if(cfg_.csplit != 1)
        {
            const int t     = tile_count(cfg_, par);
            const bool band = 2 * t > dev.cu_count && t <= dev.cu_count;
            if(cfg_.csplit == 4 &&
               (band || (2 * t <= dev.cu_count && par.c <= CSPLIT_SMALL_GRID_C)))
                return CSPLIT_FIRST;
            if(cfg_.csplit == 2 && band)
                return CSPLIT2_FIRST;
        }
        const int tiles = (par.k + cfg_.k_tile() - 1) / cfg_.k_tile();
        const float use = static_cast<float>(par.k) / (tiles * cfg_.k_tile());
        return 0.25f * use * tile_rate(cfg_);
    }

private:
    const Variant& v_;
    const Config& cfg_;
};

} // namespace hipconv::cdna4::direct_flex

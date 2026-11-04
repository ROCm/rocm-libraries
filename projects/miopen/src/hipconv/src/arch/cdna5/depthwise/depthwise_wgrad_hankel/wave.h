#pragma once

// A wave of the block, whole: its staging rings, its reads, its fragments and
// accumulator, its row loop and the epilogue that spills it.

#include <hip/hip_runtime.h>

#include "bunnies.hpp"
#include "bunnies_mi400.hpp"
#include "config_table.h"
#include "detail.h"
#include "lds.h"
#include "types.h"

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;

// The engine work of a wave that drives none. The three roles differ only by
// which of these the step is instantiated with.
struct NoIssue
{
    __device__ void prepare(int) const {}
    __device__ void wait_landed(int) const {}
    __device__ void send(int) const {}
};

template <Config cfg, DataType DT, bool PITCH_ALIGNED>
struct Wave
{
    using ElemT               = ToType<DT>;
    using arch                = bunnies::arch_mi400;
    static constexpr auto fmt = DT == DataType::bf16 ? bunnies::fpfmt::e8m7 : bunnies::fpfmt::e5m10;
    using MatA                = arch::matrix<fmt, 16, 32, bunnies::use::A>;
    using MatB                = arch::matrix<fmt, 32, 16, bunnies::use::B>;
    using MatAcc              = arch::matrix<bunnies::fpfmt::e8m23, 16, 16, bunnies::use::Acc>;
    using Frag8               = ElemT __attribute__((ext_vector_type(8)));

    static constexpr StageGeometry GEO = stage_geometry(cfg);
    static constexpr int CH            = cfg.channels_per_wmma();
    static constexpr int NW            = cfg.wmmas_per_wave;
    static constexpr int PF            = cfg.prefetch_depth;
    static constexpr int QW            = cfg.q_per_wave();
    static constexpr int QW_HALF       = QW / 2;
    static constexpr bool A_SEL_WHOLE  = cfg.a_from_tr16() && cfg.kw <= 3;
    static constexpr int DR            = GEO.delta_rows;
    static constexpr bool B_FROM_TR16  = PITCH_ALIGNED;
    static_assert(!PITCH_ALIGNED || cfg.staged_channels() % TR_GROUP == 0,
                  "a transposing read spans a whole group of staged channels");

    static constexpr int B_FRAGS = QW / 16;

    // Where the window's age a sits: step y takes its row out of slot
    // (y - y_begin) % delta_pool_rows.
    static constexpr int win_slot(int a) { return (GEO.delta_pool_rows - a) % GEO.delta_pool_rows; }

    static constexpr int B_DWORDS      = MatB::num_items * sizeof(ElemT) / sizeof(unsigned);
    static constexpr int B_LIVE_DWORDS = B_FRAGS * sizeof(Frag8) / sizeof(unsigned);
    static_assert(B_LIVE_DWORDS <= B_DWORDS, "a lane cannot hold more than the fragment");
    static_assert(B_LIVE_DWORDS == B_DWORDS || QW == 16, "the fragment is filled or half");

    const unsigned char* input_stage;
    const unsigned char* delta_pool;
    int lane;
    int wave_c;
    int c_base;
    int col_off;
    int pitch;
    int py;
    int y_begin;
    int y_end;
    int C;
    int input_row;

    const int a_row   = MatA::map({lane, 0})[0];
    const int a_ca    = a_row / cfg.kw;
    const int a_s     = a_row - a_ca * cfg.kw;
    const bool a_live = a_row < CH * cfg.kw;
    const int a_chan  = wave_c + a_ca;
    const int b_col   = MatB::map({lane, 0})[1];
    const int b_cb    = b_col / cfg.kh;
    const int b_r     = b_col - b_cb * cfg.kh;
    const bool b_live = b_col < CH * cfg.kh;
    const int b_age   = b_r;
    static_assert(CH * NW >= 2 && CH * NW <= 2 * TR_GROUP && 2 * TR_GROUP % (CH * NW) == 0,
                  "the wave's channels divide the two transpose groups a read brings");
    const int delta_tr_base            = wave_c + lane / TR_GROUP % 2 * TR_GROUP;
    const int delta_tr_chan            = delta_tr_base < pitch ? delta_tr_base : 0;
    const int delta_tr_lane            = (lane % TR_GROUP) * pitch + delta_tr_chan;
    const unsigned b_rot_sel           = (unsigned)(b_col > 0 ? b_col - 1 : 0);
    static constexpr unsigned SEL_EVEN = 0x05040100u;
    static constexpr unsigned SEL_ODD  = 0x07060302u;
    static constexpr bool SEL_WHOLE    = cfg.stride == 1 && cfg.kw <= 3;
    static constexpr int COARSE_MAX    = SEL_WHOLE ? 0 : (cfg.kw - 1) / 2;
    const int a_coarse                 = SEL_WHOLE ? 0 : a_s / 2;
    const unsigned a_sel               = SEL_WHOLE ? 0x03020100u + (unsigned)(2 * a_s) * 0x01010101u
                                         : cfg.stride == 1 ? 0x03020100u + (unsigned)(2 * (a_s & 1)) * 0x01010101u
                                                           : ((a_s & 1) ? SEL_ODD : SEL_EVEN);
    static constexpr int A_BLOCKS      = (GEO.input_cols + TR_GROUP - 1) / TR_GROUP;
    static constexpr int QB_N          = QW / TR_GROUP;
    static constexpr int OWN           = QW_HALF * cfg.stride / TR_GROUP;
    static_assert(OWN == 1 || OWN == 2, "one column block a half, or two");
    static constexpr int NOUT   = QW_HALF / 2;
    static constexpr int E_N    = cfg.stride * (NOUT - 1) + COARSE_MAX + 2;
    static constexpr int HALO_D = E_N - 4 * OWN;
    static constexpr int NHALO  = (HALO_D + 3) / 4;
    static_assert(HALO_D >= 1 && NHALO <= 2, "the window's tail is one block past, or two");
    static_assert(OWN == 2 || NHALO == 1, "at one own block the tail comes from the group");
    static_assert(A_BLOCKS >= 2 * OWN + NHALO, "the window's tail needs its blocks");

    const int tr_grp   = lane / TR_GROUP;
    const int tr_sub   = lane % TR_GROUP;
    const int tr_half  = lane / 16;
    const int own_blk  = tr_half * OWN + (tr_grp - 2 * tr_half);
    const int halo_blk = tr_half * OWN + OWN;

    MatAcc acc[NW]{};
    MatA a[NW]{};
    MatB b[2][NW]{};

    unsigned input_off = 0;
    unsigned delta_off = 0;

    // A new image starts from the state the constructor left: both rings at their
    // first slot and the fragments zero. The accumulator alone carries over.
    __device__ void reset_image()
    {
        input_off = 0;
        delta_off = 0;
        // Unconditionally, which is the point. prime_b and gather_a write these
        // under a lane predicate, so without a write the allocator has to carry
        // every one of them across the loop and cannot lend them to the prologue.
        static_for<NW>([&]<int G>() {
            a[G]    = MatA{};
            b[0][G] = MatB{};
            b[1][G] = MatB{};
        });
    }

    // One row on, wrapped where the ring ends. Both spans are compile-time, so
    // the walk is an add and a select whatever the rung.
    __device__ void advance_input()
    {
        constexpr unsigned span = (unsigned)PF * (unsigned)GEO.input_stage_slot_bytes;
        input_off += (unsigned)GEO.input_stage_slot_bytes;
        if(input_off == span)
            input_off = 0;
    }

    // The delta pool's ring, walked the same way.
    __device__ void advance_delta()
    {
        constexpr unsigned span = (unsigned)GEO.delta_pool_rows * (unsigned)GEO.delta_row_bytes;
        delta_off += (unsigned)GEO.delta_row_bytes;
        if(delta_off == span)
            delta_off = 0;
    }

    // A's addresses: own[] is the half's own pair of column blocks, halo[] the
    // block past them that the tap shift reaches into.
    struct InputAddrs
    {
        unsigned own[2];
        unsigned halo[OWN == 2 ? 2 * NHALO : 1];
    };

    // Formed apart from the reads so they can sit above the step's barrier.
    __device__ InputAddrs input_addrs() const
    {
        const ElemT* input_slot = reinterpret_cast<const ElemT*>(input_stage + input_off);
        const ElemT* c0 = input_slot + (col_off + own_blk * TR_GROUP + tr_sub) * input_row + wave_c;
        InputAddrs a;
        auto at = [](const ElemT* p) {
            return (unsigned)(unsigned long long)reinterpret_cast<uintptr_t>(p);
        };
        a.own[0] = at(c0);
        a.own[1] = at(c0 + TR_GROUP);
        if constexpr(OWN == 2)
            static_for<NHALO>([&]<int H>() {
                const ElemT* c1 = input_slot +
                                  (col_off + (halo_blk + H) * TR_GROUP + tr_sub) * input_row +
                                  wave_c;
                a.halo[2 * H]     = at(c1);
                a.halo[2 * H + 1] = at(c1 + TR_GROUP);
            });
        return a;
    }

    // The pool row this step multiplies against, which is the ring's own place.
    __device__ const ElemT* delta_row_of() const
    {
        return reinterpret_cast<const ElemT*>(delta_pool + delta_off);
    }

    // One row's transposing read, held in the registers it lands in from the
    // step that issues it to the step that multiplies it.
    struct InputRow
    {
        Frag8 own[2];
        Frag8 halo[OWN == 2 ? 2 * NHALO : 1];
    };

    // Issued and left in flight; the step that consumes it opens by landing it.
    // Steps the input ring, the reads walking it in order.
    __device__ InputRow read_input_row()
    {
        const InputAddrs ad = input_addrs();
        InputRow r;
        r.own[0] = ds_load_tr16<Frag8>(ad.own[0]);
        r.own[1] = ds_load_tr16<Frag8>(ad.own[1]);
        if constexpr(OWN == 2)
            static_for<2 * NHALO>([&]<int H>() { r.halo[H] = ds_load_tr16<Frag8>(ad.halo[H]); });
        advance_input();
        return r;
    }

    // B's addresses, one per q block of the half.
    struct DeltaAddrs
    {
        unsigned a[B_FRAGS];
    };

    // Split from the read for the same reason A's is.
    __device__ DeltaAddrs delta_addrs(const ElemT* row) const
    {
        DeltaAddrs r{};
        if constexpr(B_FROM_TR16)
            static_for<B_FRAGS>([&]<int R>() { r.a[R] = tr16_addr(row, R * TR_GROUP); });
        return r;
    }

    // The reads themselves, against what delta_addrs already formed.
    __device__ void read_delta_tr(const DeltaAddrs& ad, unsigned (&out)[B_LIVE_DWORDS]) const
    {
        if constexpr(B_FROM_TR16)
        {
            static_for<B_FRAGS>([&]<int R>() {
                const Frag8 raw = ds_load_tr16<Frag8>(ad.a[R]);
                __builtin_memcpy(out + R * 4, &raw, 16);
            });
        }
    }

    // A lane's address inside a delta row: its place in the transpose group and
    // the wave's channel base, both invariant, plus the q its half reads.
    __device__ unsigned tr16_addr(const ElemT* row, int q) const
    {
        return (unsigned)(unsigned long long)reinterpret_cast<uintptr_t>(
            row + (tr_half * QW_HALF + q) * pitch + delta_tr_lane);
    }

    // The lane's own channel picked out of what the read spread across the half,
    // committed only where this step takes a row off the pool.
    template <int P, int G>
    __device__ void permute_b(const unsigned (&src)[B_LIVE_DWORDS], bool b_now)
    {
        static constexpr int u_c = G * CH;
        const unsigned sel       = b_live ? (unsigned)(u_c + b_cb) : 0u;
        // A select, not a branch: prime_b issues DR * NW of these.
        unsigned got[B_LIVE_DWORDS];
        __builtin_memcpy(got, &b[P][G].data, sizeof(got));
        static_for<B_LIVE_DWORDS>([&]<int D>() {
            const unsigned moved = __builtin_amdgcn_permlane16_var(0u, src[D], sel, false, false);
            got[D]               = b_now ? moved : got[D];
        });
        __builtin_memcpy(&b[P][G].data, got, sizeof(got));
    }

    // Zeroes the fragment where a lane's tap has no delta row this step.
    template <int P, int G>
    __device__ void clear_b(bool off)
    {
        unsigned got[B_LIVE_DWORDS];
        __builtin_memcpy(got, &b[P][G].data, sizeof(got));
        static_for<B_LIVE_DWORDS>([&]<int D>() { got[D] = off ? 0u : got[D]; });
        __builtin_memcpy(&b[P][G].data, got, sizeof(got));
    }

    // The element fallback for a pitch the transposing read cannot address.
    template <int P, int G>
    __device__ void gather_b(const ElemT* row, bool b_now)
    {
        static constexpr int u_c = G * CH;
        const int c              = wave_c + u_c + b_cb;
        if(b_now)
        {
            const int q0 = tr_half * QW_HALF;
            ElemT v[QW_HALF];
            static_for<QW_HALF>([&]<int Q>() { v[Q] = row[(q0 + Q) * pitch + c]; });
            __builtin_memcpy(&b[P][G].data, v, sizeof(v));
        }
    }

    // A's fragment out of the transposed row: one permute a source block, then
    // the tap shift as a byte select over each dword pair.
    template <int G>
    __device__ void gather_a(const InputRow& r)
    {
        constexpr int RD     = G * CH / TR_GROUP;
        constexpr int SIG    = G * CH % TR_GROUP;
        const Frag8& own     = r.own[RD];
        const unsigned sigma = (unsigned)(SIG + a_ca);
        unsigned cur[4];
        __builtin_memcpy(cur, &own, 16);
        unsigned e[E_N];
        static_for<OWN>([&]<int I>() {
            static_for<4>([&]<int D>() {
                e[4 * I + D] = __builtin_amdgcn_permlane16_var(
                    0u, cur[D], sigma + (unsigned)(I * TR_GROUP), false, false);
            });
        });
        if constexpr(OWN == 1)
            static_for<HALO_D>([&]<int D>() {
                e[4 + D] = __builtin_amdgcn_permlane16_var(
                    0u, cur[D], sigma + (unsigned)TR_GROUP, false, false);
            });
        else
            static_for<NHALO>([&]<int H>() {
                unsigned nxt[4];
                __builtin_memcpy(nxt, &r.halo[2 * H + RD], 16);
                static_for < HALO_D - 4 * H<4 ? HALO_D - 4 * H : 4>([&]<int D>() {
                    e[4 * OWN + 4 * H + D] =
                        __builtin_amdgcn_permlane16_var(0u, nxt[D], sigma, false, false);
                });
            });
        if(a_live)
        {
            auto* out = reinterpret_cast<unsigned*>(&a[G].data);
            static_for<NOUT>([&]<int J>() {
                constexpr int I0 = cfg.stride * J;
                unsigned lo = e[I0], hi = e[I0 + 1];
                static_for<COARSE_MAX>([&]<int C>() {
                    if(a_coarse == C + 1)
                    {
                        lo = e[I0 + C + 1];
                        hi = e[I0 + C + 2];
                    }
                });
                out[J] = __builtin_amdgcn_perm(hi, lo, a_sel);
            });
        }
    }

    static constexpr int MUL_GROUPS = NW;
    static constexpr int A_UP       = MUL_GROUPS > 2 ? 2 : MUL_GROUPS;

    // One group's multiply, where its fragments were built.
    template <int P, int G>
    __device__ void multiply()
    {
        constexpr int AG = G;
        arch::mma<>::wmma(acc[AG], a[G], b[P][G], acc[AG]);
    }

    // Hands each column's fragment one filter row along, which is what makes the
    // row a lane read at age 0 still the row it wants DR steps later.
    template <int P, int G>
    __device__ void rotate_b()
    {
        {
            unsigned src[B_LIVE_DWORDS], moved[B_LIVE_DWORDS];
            __builtin_memcpy(src, &b[P][G].data, sizeof(src));
#pragma unroll
            for(int j = 0; j < B_LIVE_DWORDS; ++j)
                moved[j] = __builtin_amdgcn_permlane16_var(0u, src[j], b_rot_sel, false, false);
            __builtin_memcpy(&b[1 - P][G].data, moved, sizeof(moved));
        }
    }

    // The window the prologue left, taken into the registers that carry it. Read
    // once a chunk and at full width, every lane's age being different here.
    __device__ void prime_b()
    {
        if constexpr(B_FROM_TR16)
        {
            unsigned raw[DR][B_LIVE_DWORDS];
            static_for<DR>([&]<int A>() {
                read_delta_tr(delta_addrs(reinterpret_cast<const ElemT*>(
                                  delta_pool + win_slot(A) * GEO.delta_row_bytes)),
                              raw[A]);
            });
            const int age = b_live ? b_age : 0;
            static_for<NW>([&]<int G>() {
                static_for<DR>([&]<int A>() { permute_b<0, G>(raw[A], b_live && age == A); });
            });
            if constexpr(cfg.stride != 1)
                static_for<NW>(
                    [&]<int G>() { clear_b<0, G>((y_begin + py - b_r) % cfg.stride != 0); });
        }
        else
        {
            const int age    = b_live ? b_age : 0;
            const ElemT* row = reinterpret_cast<const ElemT*>(
                delta_pool +
                ((GEO.delta_pool_rows - age) % GEO.delta_pool_rows) * GEO.delta_row_bytes);
            static_for<NW>([&]<int G>() { gather_b<0, G>(row, b_live); });
            if constexpr(cfg.stride != 1)
                static_for<NW>(
                    [&]<int G>() { clear_b<0, G>((y_begin + py - b_r) % cfg.stride != 0); });
        }
    }

    // One row. PIPE reads a step ahead, FIRST has no shift to take, HAS_NEXT has
    // a row after it, P is the B set's parity, ROT shifts, Issue is engine work.
    template <bool PIPE, bool FIRST, bool HAS_NEXT, int P, bool ROT, class Issue>
    __device__ void step(int y, InputRow& cur, InputRow& nxt, Issue& issue)
    {
        const ElemT* delta_row = delta_row_of();
        DeltaAddrs delta_ad[1];
        delta_ad[0] = delta_addrs(delta_row);
        advance_delta();

        issue.wait_landed(y);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier_signal(-1);
        issue.prepare(y);
        __builtin_amdgcn_s_barrier_wait(-1);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        issue.send(y);
        if constexpr(!(PIPE && !FIRST))
        {
            cur = read_input_row();
        }
        __builtin_amdgcn_sched_barrier(0);
        unsigned delta_raw[1][B_LIVE_DWORDS]{};
        static_for<1>([&]<int A>() { read_delta_tr(delta_ad[A], delta_raw[A]); });
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(ROT)
        {
            static_for<NW>([&]<int G>() { rotate_b<1 - P, G>(); });
            __builtin_amdgcn_sched_barrier(0);
        }

        // At stride 2 the pool row pairs with tap 0 only on every other step.
        const bool b_now = b_live && b_r == 0;
        const bool fresh = cfg.stride == 1 || (y + py) % cfg.stride == 0;
        auto shuffle_a   = [&]<int G>() { gather_a<G>(cur); };
        auto shuffle_b   = [&]<int G>() {
            if constexpr(!B_FROM_TR16)
                gather_b<P, G>(delta_row, b_now);
            else
                permute_b<P, G>(delta_raw[0], b_now);
            if constexpr(cfg.stride != 1)
                clear_b<P, G>(b_now && !fresh);
        };
        auto mul = [&]<int G>() { multiply<P, G>(); };
        static_for<A_UP>([&]<int G>() { shuffle_a.template operator()<G>(); });
        if constexpr(PIPE && HAS_NEXT)
            nxt = read_input_row();
        __builtin_amdgcn_sched_barrier(0);
        static_for<MUL_GROUPS>([&]<int G>() {
            shuffle_b.template operator()<G>();
            if constexpr(G >= A_UP)
                shuffle_a.template operator()<G>();
            mul.template operator()<G>();
        });
    }

    // The chunk's rows, walked two at a time so the B set's parity stays a
    // template parameter.
    template <class Issue>
    __device__ void run(Issue& issue)
    {
        InputRow t0, t1;
        int y = y_begin;
        if(y < y_end)
        {
            step<false, true, false, 0, false>(y, t0, t1, issue);
            for(++y; y + 1 < y_end; y += 2)
            {
                step<false, true, false, 1, true>(y, t0, t1, issue);
                step<false, true, false, 0, true>(y + 1, t0, t1, issue);
            }
            if(y < y_end)
                step<false, true, false, 1, true>(y, t0, t1, issue);
        }
    }

    static constexpr int ACC_ITEMS  = sizeof(MatAcc::data) / sizeof(float);
    static constexpr int ACC_HALVES = CH * cfg.kw > ACC_ITEMS ? 2 : 1;

    // The rows of accumulator half H belonging to channel CB of the group: the
    // item they start at, how many there are, and which tap the first one is.
    template <int H, int CB>
    struct Slice
    {
        static constexpr int base = H * ACC_ITEMS;
        static constexpr int row0 = CB * cfg.kw;
        static constexpr int cap  = base + ACC_ITEMS < CH * cfg.kw ? base + ACC_ITEMS : CH * cfg.kw;
        static constexpr int lo   = row0 > base ? row0 : base;
        static constexpr int hi   = row0 + cfg.kw < cap ? row0 + cfg.kw : cap;
        static constexpr int n    = hi > lo ? hi - lo : 0;
        static constexpr int item = lo - base;
        static constexpr int tap  = lo - row0;
    };

    // Stage this wave's share of the block's slice; the engine issue that moves
    // it is the launch's, the slice being the whole block's.
    __device__ void store()
    {
        float* base     = reinterpret_cast<float*>(const_cast<unsigned char*>(input_stage));
        float* const p0 = base + wave_c * cfg.kh * cfg.kw + b_r * cfg.kw;
        if(b_live)
        {
            static_for<NW>([&]<int G>() {
                const int chan0 = c_base + wave_c + G * CH;
                static_for<CH>([&]<int CB>() {
                    if(chan0 + CB >= C || b_cb != CB)
                        return;
                    static_for<ACC_HALVES>([&]<int H>() {
                        using S = Slice<H, CB>;
                        if constexpr(S::n > 0)
                        {
                            if((lane >> 4) != H)
                                return;
                            float run[S::n];
#pragma unroll
                            for(int t = 0; t < S::n; ++t)
                                run[t] = acc[G].data[S::item + t];
                            constexpr int OFF = (G * CH + CB) * cfg.kh * cfg.kw + S::tap;
                            store_n(p0 + OFF, run);
                        }
                    });
                });
            });
        }
    }
};

} // namespace hipconv::cdna5::depthwise_wgrad_hankel

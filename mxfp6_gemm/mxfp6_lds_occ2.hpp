#pragma once
// ============================================================================
// occ2 MXFP6 LDS GEMM — Phase 2: axis-3 mfma<->buffer_load drip interleave.
//
// Forks the production lds_gemm_db (DB=true path only) into lds_gemm_occ2 to add
// axis-3: instead of issuing the next tile's 9 buffer_loads as a top-of-loop burst
// (saturates the single wave's vmem issue port -> the residual ~13% issue stall that
// survived occ2's 30%->13% reduction), DRIP them one per MFMA quartet boundary so
// the load issues overlap the MFMA execution. scale loads + the 9th tile load (L8)
// stay in the compute prologue (lead-decided v1 schedule: even 1-load/quartet).
//
// REUSES everything from mxfp6_lds.hpp (load_tile_lds/read_op/tile_scale/issue_tile/
// asm_load_*; mfma/clear_acc/store_acc_t via mxfp6_asm_utils.hpp) -> zero redefinition.
// Adds only: issue_nth_load<N> (compile-time single load), for_seq (C++17 unroll),
// lds_gemm_occ2 (DB-only drip hot loop).
//
// Correctness is STRUCTURAL, not timing: the 9 dripped loads still all issue AFTER the
// tile barrier into the NEXT buffer (WAR unchanged vs RDB), AND a wait_vmcnt(0) before
// each barrier guarantees the next buffer fully landed before it is read (RAW). The
// shallow KT128 window (512cyc < 880cyc load latency) plus drip pushing some loads
// later would otherwise shrink the RAW margin; wait_vmcnt(0) removes the timing bet.
// occ2's co-resident 2nd WG hides the wait. Drip still spreads the ISSUE (the stall we
// attack is issue-port saturation, not completion), so the experiment stays valid.
//
// DOES NOT touch mxfp6_lds.hpp / test_lds.cpp.
// ============================================================================
#include "mxfp6_lds.hpp"
#include <utility>

namespace mxfp6 {

// Issue the Nth of a tile's (A_ISSUES + B_ISSUES) cooperative buffer_load_dwordx4 loads
// as a SINGLE load (compile-time N -> no dynamic index/spill). rsrcA/rsrcB are the A/B
// buffer descriptors, hoisted by the caller (kernel-invariant: depend only on Ag/Bg).
// next_base = LDS base of the buffer being filled; kb = kt*KT_BYTES of the tile to fetch.
// s_nop 0 guards the SALU-writes-M0 -> buffer_load(lds) 1-wait-state hazard (set_m0 sits
// immediately before the load here; in the burst loop the compiler interleaved other instrs).
template <int M_TILE, int N_TILE, int KT_BYTES, int CPR, int N>
__device__ __forceinline__ void issue_nth_load(uint32_t next_base, const v4i& rsrcA,
                                               const v4i& rsrcB, int A_row_bytes,
                                               int B_row_bytes, int kb, int wave_id, int lane) {
    constexpr int A_ISSUES = M_TILE * CPR / 256;
    constexpr int A_BYTES  = M_TILE * KT_BYTES;
    if constexpr (N < A_ISSUES) {
        constexpr int i = N;
        int chunk = i * 256 + wave_id * 64 + lane;
        int m = chunk / CPR, ck = chunk % CPR;
        uint32_t voff = (uint32_t)(m * A_row_bytes + kb + ck * 16);
        set_m0(__builtin_amdgcn_readfirstlane(next_base + (uint32_t)((i * 256 + wave_id * 64) * 16)));
        asm volatile("s_nop 0\n buffer_load_dwordx4 %0, %1, 0 offen lds"
                     : : "v"(voff), "s"(rsrcA) : "memory");
    } else {
        constexpr int i = N - A_ISSUES;
        int chunk = i * 256 + wave_id * 64 + lane;
        int m = chunk / CPR, ck = chunk % CPR;
        uint32_t voff = (uint32_t)(m * B_row_bytes + kb + ck * 16);
        set_m0(__builtin_amdgcn_readfirstlane(next_base + (uint32_t)A_BYTES +
                                              (uint32_t)((i * 256 + wave_id * 64) * 16)));
        asm volatile("s_nop 0\n buffer_load_dwordx4 %0, %1, 0 offen lds"
                     : : "v"(voff), "s"(rsrcB) : "memory");
    }
}

// C++17 compile-time unroll: invoke f(integral_constant<int,Q>) for Q in the sequence.
template <class F, int... Qs>
__device__ __forceinline__ void for_seq(F&& f, std::integer_sequence<int, Qs...>) {
    (f(std::integral_constant<int, Qs>{}), ...);
}

// Drip schedule = BACK-HEAVY (ATT: per-cluster stall climbs monotonically; the last 2-3
// loads eat 30-40% of the cluster stall once the vmem issue queue saturates). So issue the
// cheap front (LOADS-DRIP_TAIL) loads as an early burst (they queue freely) and DRIP only
// the expensive tail DRIP_TAIL loads, one per the LAST DRIP_TAIL quartets, with the MFMA
// quartets between them draining an issue slot before each heavy tail load. -DMXFP6_DRIP_TAIL=N
// lets the profiler A/B the tail count (e.g. 2/3/4) without source edits.
//
// MEASURED @8192^3 FP16 SWZ0 (best-of-5 single-buffer, same harness as production):
//   bare occ2-DB (lds_gemm_db<128,256,128,2,2,2,0,true>) = 1640
//   drip TAIL=2/3/4 = 1652 / 1659 / 1669  (dose-response, real but small)
//   TAIL=4 is the structural max (KT128 -> 4 quartets) and the best: +1.52% vs bare.
//   vs production occ1 (lds_gemm_db<256,256,192,...>) same harness = 1941 best-of / 1781 rotating.
//   => occ2 net -14.3% vs production: it BREAKS the buffer_load issue wall (ATT 30%->13%->6.8%,
//      occ=2 hardware-confirmed) but the forced 8-acc small tile (arith-intensity 1.33 vs 2.0)
//      trades it for a higher LDS-read wall (lgkmcnt+ds_read ~66-68%) that is structurally
//      unsolvable in the occ2 budget. drip's buffer_load saving is ~1:1 cancelled by the
//      wait_vmcnt(0) RAW guard; the +1.52% is a secondary reorder effect, not the drip itself.
//   Default set to the best (4) so an out-of-box build is the strongest occ2 config.
#ifndef MXFP6_DRIP_TAIL
#define MXFP6_DRIP_TAIL 4
#endif

// Double-buffered deep-K LDS GEMM with axis-3 drip (DB=true only). Same template params
// as lds_gemm_db so a driver can swap instantiations. MIN_OCC=2 for occupancy-2.
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 2,
          int SWZ = 0, bool DB = true, typename OutT = float>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_occ2(const void* __restrict__ A, const void* __restrict__ B,
                  const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                  OutT* __restrict__ D, int N, int k_iters, int A_row_bytes, int B_row_bytes,
                  const uint8_t* __restrict__ sA_plain = nullptr,
                  const uint8_t* __restrict__ sB_plain = nullptr) {
    static_assert(DB, "lds_gemm_occ2 is the DB=true drip path only");
    (void)sA_plain; (void)sB_plain;  // no K-tail: callers K-pad to K64_PER_TILE | k_iters
    constexpr int KT_BYTES   = K_TILE * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int K64_PER_TILE = K_TILE / 64;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WAVES_M, N_PW = N_BLKS / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int BUF = A_BYTES + B_BYTES;
    constexpr int LOADS_PER_TILE = (M_TILE * ROW_CHUNKS + N_TILE * ROW_CHUNKS) / 256;  // 9
    constexpr int QUARTETS = K64_PER_TILE * N_PW;                                       // 8
    static_assert(LOADS_PER_TILE >= 1, "need >=1 tile load");
    // Back-heavy drip: DRIP_TAIL expensive tail loads dripped across the last DRIP_TAIL
    // quartets; the remaining EARLY cheap loads burst in the prologue.
    constexpr int DRIP_TAIL = (MXFP6_DRIP_TAIL < QUARTETS)
                                  ? (MXFP6_DRIP_TAIL < LOADS_PER_TILE ? MXFP6_DRIP_TAIL : LOADS_PER_TILE)
                                  : (QUARTETS < LOADS_PER_TILE ? QUARTETS : LOADS_PER_TILE);
    constexpr int EARLY = LOADS_PER_TILE - DRIP_TAIL;     // cheap front loads (prologue burst)
    constexpr int FIRST_DRIP_Q = QUARTETS - DRIP_TAIL;    // first quartet that drips a tail load

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m, wg_n;
    if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs;
        wg_n = fn + r % gs;
    } else {
        wg_m = blockIdx.x;
        wg_n = blockIdx.y;
    }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_row_bytes;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_row_bytes;

    // Hoisted buffer descriptors (kernel-invariant; SGPR, not VGPR). Same fmt as load_tile_lds.
    uint64_t ba = reinterpret_cast<uint64_t>(Ag);
    v4i rsrcA{(int)(uint32_t)ba, (int)((uint32_t)(ba >> 32) & 0xFFFF), (int)0x7FFFFFFF, (int)0x00020000};
    uint64_t bb = reinterpret_cast<uint64_t>(Bg);
    v4i rsrcB{(int)(uint32_t)bb, (int)((uint32_t)(bb >> 32) & 0xFFFF), (int)0x7FFFFFFF, (int)0x00020000};

    AccTileA acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    constexpr int NDA = SA_PAD / 4, NDB = SB_PAD / 4;
    static_assert(NDA == 1 && NDB == 1, "tiled-scale path assumes <=4 blocks/wave");
    int sa_grp = wg_m * WAVES_M + wm, sb_grp = wg_n * WAVES_N + wn;
    int k_tiles = k_iters / K64_PER_TILE;

    // Burst prefetch (prologue only, outside the steady state): scales + all 9 tile loads.
    // Identical to lds_gemm_db's prefetch; used only to fill tile 0.
    auto prefetch_burst = [&](int kt, uint32_t base, int (*sa)[NDA], int (*sb)[NDB]) {
        int kb = kt * KT_BYTES;
#ifndef NOSCALE
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int ta[K64_PER_TILE], tb[K64_PER_TILE];
        asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
        asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
        for (int sub = 0; sub < K64_PER_TILE; sub++) { sa[sub][0] = ta[sub]; sb[sub][0] = tb[sub]; }
#else
        (void)sa; (void)sb;
#endif
        issue_tile<M_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + 0, Ag, A_row_bytes, kb, wave, lane);
        issue_tile<N_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + A_BYTES, Bg, B_row_bytes, kb, wave, lane);
    };

    // compute one K-tile from `cur` while (if do_pf) DRIPPING the next tile's prefetch into
    // `next_base` (+ next scales into sa_nx/sb_nx): scales + L8 in the prologue, tile loads
    // L0..L7 one per quartet boundary (after b_next's ds_read, before the mi-MFMA quartet).
    auto compute_drip = [&](uint32_t cur, const int (*sa)[NDA], const int (*sb)[NDB],
                            bool do_pf, uint32_t next_base, int (*sa_nx)[NDA],
                            int (*sb_nx)[NDB], int kt_next) {
        int kb_next = kt_next * KT_BYTES;
        // --- prologue: next-tile scales (no-wait) + the 9th tile load (L8) ---
        if (do_pf) {
#ifndef NOSCALE
            const char* pa = reinterpret_cast<const char*>(sA) +
                             (size_t)((sa_grp * k_tiles + kt_next) * 64 + lane) * K64_PER_TILE * SA_PAD;
            const char* pb = reinterpret_cast<const char*>(sB) +
                             (size_t)((sb_grp * k_tiles + kt_next) * 64 + lane) * K64_PER_TILE * SB_PAD;
            int ta[K64_PER_TILE], tb[K64_PER_TILE];
            asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
            asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
            for (int s = 0; s < K64_PER_TILE; s++) { sa_nx[s][0] = ta[s]; sb_nx[s][0] = tb[s]; }
#else
            (void)sa_nx; (void)sb_nx;
#endif
            // EARLY cheap front loads burst here (they queue freely before saturation).
            for_seq([&](auto Ic) {
                issue_nth_load<M_TILE, N_TILE, KT_BYTES, ROW_CHUNKS, Ic.value>(
                    next_base, rsrcA, rsrcB, A_row_bytes, B_row_bytes, kb_next, wave, lane);
            }, std::make_integer_sequence<int, EARLY>{});
        }

        // --- b stream prologue: first b (sub0, ni0) ---
        v6i b_cur = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + 0, 0, lane);
#ifdef NOSCALE
        int sbv_cur = 127;
#else
        int sbv_cur = sb[0][0] & 0xff;
#endif
        v6i a[M_PW];
        int sav[M_PW];
        for_seq([&](auto Qc) {
            constexpr int Q   = Qc.value;
            constexpr int sub = Q / N_PW;
            constexpr int ni  = Q % N_PW;
            // fresh a[] + a-scales at the start of each sub
            if constexpr (ni == 0) {
#pragma unroll
                for (int mi = 0; mi < M_PW; mi++) {
                    int blk = wm * M_PW + mi;
                    a[mi] = read_op<KT_BYTES>(smem, cur, blk, sub, lane);
#ifdef NOSCALE
                    sav[mi] = 127;
#else
                    sav[mi] = (sa[sub][mi / 4] >> (8 * (mi % 4))) & 0xff;
#endif
                }
            }
            // BACK-HEAVY drip: only the last DRIP_TAIL quartets each drip one expensive tail
            // load (EARLY + offset), so the 2 MFMA per quartet drain an issue slot before it.
            if constexpr (Q >= FIRST_DRIP_Q) {
                if (do_pf)
                    issue_nth_load<M_TILE, N_TILE, KT_BYTES, ROW_CHUNKS, EARLY + (Q - FIRST_DRIP_Q)>(
                        next_base, rsrcA, rsrcB, A_row_bytes, B_row_bytes, kb_next, wave, lane);
            }
            // next b in the continuous stream
            v6i b_next;
            int sbv_next = 0;
            constexpr bool has_next = (Q + 1 < QUARTETS);
            if constexpr (has_next) {
                constexpr int nsub = (ni + 1 < N_PW) ? sub : sub + 1;
                constexpr int nni  = (ni + 1 < N_PW) ? ni + 1 : 0;
                b_next = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + nni, nsub, lane);
#ifdef NOSCALE
                sbv_next = 127;
#else
                sbv_next = (sb[nsub][nni / 4] >> (8 * (nni % 4))) & 0xff;
#endif
            }
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b_cur, sav[mi], sbv_cur);
            b_cur = b_next;
            sbv_cur = sbv_next;
        }, std::make_integer_sequence<int, QUARTETS>{});
    };

    // RDB double buffer + drip. wait_vmcnt(0) before each barrier makes RAW structural
    // (next buffer fully landed before read) since drip shrinks the deep-K timing margin.
    int sa0[K64_PER_TILE][NDA], sa1[K64_PER_TILE][NDA];
    int sb0[K64_PER_TILE][NDB], sb1[K64_PER_TILE][NDB];
    prefetch_burst(0, 0, sa0, sb0);  // tile 0 -> buf0
    int kt = 0;
    for (; kt + 1 < k_tiles; kt += 2) {
        wait_vmcnt(0); __syncthreads();                       // RAW(buf0) + WAR
        compute_drip(0, sa0, sb0, true, BUF, sa1, sb1, kt + 1);   // read buf0, drip tile kt+1 -> buf1
        bool pf = (kt + 2 < k_tiles);
        wait_vmcnt(0); __syncthreads();                       // RAW(buf1) + WAR
        compute_drip(BUF, sa1, sb1, pf, 0, sa0, sb0, kt + 2);     // read buf1, drip tile kt+2 -> buf0
    }
    if (kt < k_tiles) {                                       // odd tail (buf0 already loaded)
        wait_vmcnt(0); __syncthreads();
        compute_drip(0, sa0, sb0, false, 0, sa0, sb0, 0);
    }

#pragma unroll
    for (int mi = 0; mi < M_PW; mi++) {
        int m = wg_m * M_TILE + (wm * M_PW + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int n = wg_n * N_TILE + (wn * N_PW + ni) * 32;
            store_acc_t<OutT>(D, N, acc[mi][ni].vec, m, n);
        }
    }
}

}  // namespace mxfp6

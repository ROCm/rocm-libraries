#pragma once
// ############################################################################
// STATUS: NO-GO — ARCHIVED REFERENCE (8-wave wave-level ping-pong; Idea 2, ported
//         from HipKittens arXiv:2511.08083 / HazyResearch/HipKittens 8_wave.cu).
//         NOT production. Production = deep-K lds_gemm_db (mxfp6_lds.hpp, 1741).
//
// RESULT: MS1 best F16 @8192^3 = 1432 (-17.8% vs deep-K 1741); occ-2 held, 0 spill
//         (F16 128/128, F32 61/128 VGPR/AGPR); 6/6 bit-exact.
//
// ROOT CAUSE: cooperative glds->LDS staging forces a CORRECTNESS-REQUIRED per-cluster
//   vmcnt(0) drain + s_barrier at the cross-wave hand-off (operands gather across
//   waves' chunks; TOPSYNC=0 -> err=2e2 race). At 8-wave shallow K_STEP=64 it fires
//   every ~8 MFMAs, 12-27x more often than deep-K's ~96-MFMA window. Deep-K's large
//   MFMA window amortizes this unavoidable drain; occ-2's 2x is eaten by sync. A
//   deeper prefetch ring (lds_gemm_pp8_ring) doesn't help — the drain empties it
//   every iter (perf flat across nbuf 2/3/4/6).
//
// KERNELS: lds_gemm_pp8 (MS0 single-buf scaffold), lds_gemm_pp8_pingpong (MS1
//   ping-pong; best = template PF_FRONT=1, LGK_MANUAL=0), lds_gemm_pp8_ring (MS1
//   attempt-3 deep ring; TOPSYNC=1 for correctness). SEE memory
//   mxfp6_8wave_pingpong_nogo + team knowledge.md.
// ############################################################################
// ============================================================================
// MS0: 8-wave scaffold (clean-room) — Idea 2 step 0.
//
// Goal: prove "256x256 block unchanged, MFMA 32x32x64 unchanged, 4 waves -> 8
// waves" can (a) accumulate 256x256 CORRECTLY and (b) reach occ-2 (2 waves/SIMD)
// WITHOUT register spill. NO ping-pong, NO conditional-barrier stagger, NO
// 16x16x128, NO scale-path change — those are MS1/MS2/MS3.
//
// Reuses mxfp6_lds.hpp device primitives verbatim:
//   load_tile_lds  (cooperative global_load_lds, 256-thread / 4-wave primitive)
//   read_op        (ds_read one 32x32 FP6 MFMA operand)
//   tile_scale / TiledScale  (host tiled-scale layout)
//   mfma_scale_f32_32x32x64_fp6 / clear_acc / store_acc_t  (asm_utils)
//
// Wave/team split (mirrors HK MXFP8_8wave/8_wave.cu: WARPS_ROW=2, WARPS_COL=4):
//   8 waves = WARPS_M(2) x WARPS_N(4).  wm = wave/WARPS_N in {0,1} (two M-teams,
//   2 waves/SIMD each); wn = wave%WARPS_N in {0..3}.  Per-wave tile = 128x64 =
//   M_PW(4) x N_PW(2) of 32x32 acc = 128 fp32 acc/lane (block stays 256x256).
//
// Cooperative load with an 8-wave block: load_tile_lds is a 256-thread (4-wave)
// primitive (chunk = i*256 + wave_id*64 + lane). We drive it by WAVE-SPLITTING the
// block in half: waves 0..3 cooperatively load the A tile, waves 4..7 the B tile
// (each tile = 256 rows x CPR=3 = 768 chunks = 3 issues x 256 threads). This reuses
// the primitive UNMODIFIED. Requires exactly 8 waves (HALF = 4 waves per tile).
// ============================================================================
#include "mxfp6_lds.hpp"

namespace mxfp6 {

// MS0 8-wave single-buffer deep-K=64 GEMM. K_STEP=64 (one 32x32x64 MFMA-K/iter,
// K64_PER_TILE=1). Single LDS buffer, straight load->ds_read->MFMA loop (no overlap).
template <int M_TILE, int N_TILE, int K_STEP, int WARPS_M, int WARPS_N,
          typename OutT = float>
__global__ void __launch_bounds__(512, 2)
    lds_gemm_pp8(const void* __restrict__ A, const void* __restrict__ B,
                 const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                 OutT* __restrict__ D, int N, int k_iters, int A_rs, int B_rs) {
    constexpr int NWAVES   = WARPS_M * WARPS_N;
    static_assert(NWAVES == 8, "MS0 is an 8-wave scaffold (WARPS_M*WARPS_N==8)");
    static_assert(WARPS_M == 2 && WARPS_N == 4, "MS0 team split = 2x4 (HK 8_wave)");
    constexpr int K64_PER_TILE = K_STEP / 64;
    static_assert(K64_PER_TILE == 1, "MS0 K_STEP=64 (no ping-pong, no deep K)");

    constexpr int KT_BYTES = K_STEP * 6 / 8;   // 48
    constexpr int ROW_CHUNKS    = KT_BYTES / 16;    // 3
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;        // 8, 8
    constexpr int M_PW = M_BLKS / WARPS_M, N_PW = N_BLKS / WARPS_N;  // 4, 2
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int HALF = NWAVES / 2;            // 4 waves cooperatively load one tile
    // load_tile_lds needs exactly HALF*64 = 256 threads (4 waves) per tile call.
    static_assert(M_TILE * ROW_CHUNKS % 256 == 0, "A tile chunks must be 256-multiple");
    static_assert(N_TILE * ROW_CHUNKS % 256 == 0, "B tile chunks must be 256-multiple");

    // tiled-scale unpack (reuse lds_gemm_db layout): SA_PAD/SB_PAD round group to 4.
    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;  // 4,4

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WARPS_N, wn = wave % WARPS_N;   // wm in {0,1}, wn in {0..3}

    int wg_m = blockIdx.x, wg_n = blockIdx.y;       // plain decode (no swz in MS0)
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_rs;

    AccTileA acc[M_PW][N_PW];   // 4x2 = 8 tiles x 16 fp32 = 128 fp32 acc/lane
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    const int k_tiles = k_iters;                    // K64_PER_TILE==1
    const int sa_grp = wg_m * WARPS_M + wm;
    const int sb_grp = wg_n * WARPS_N + wn;

    for (int kt = 0; kt < k_tiles; kt++) {
        const int kb = kt * KT_BYTES;
        // Wave-split cooperative load: first HALF waves -> A tile, rest -> B tile.
        if (wave < HALF)
            load_tile_lds<M_TILE, KT_BYTES, ROW_CHUNKS>(smem, 0, Ag, A_rs, kb, wave, lane);
        else
            load_tile_lds<N_TILE, KT_BYTES, ROW_CHUNKS>(smem, A_BYTES, Bg, B_rs, kb,
                                                   wave - HALF, lane);
        // Tiled scales: one dword per wave holds this wave's M_PW (N_PW) block scales
        // for this k64 (tile-grouped layout, K64_PER_TILE=1 -> dwordx1). Loaded no-wait.
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int sa0 = asm_load_dword_nowait(pa);
        int sb0 = asm_load_dword_nowait(pb);
        wait_vmcnt(0);
        __syncthreads();

        v6i a[M_PW], b[N_PW];
        int sav[M_PW], sbv[N_PW];
#pragma unroll
        for (int mi = 0; mi < M_PW; mi++) {
            int blk = wm * M_PW + mi;
            a[mi] = read_op<KT_BYTES>(smem, 0, blk, 0, lane);
#ifdef NOSCALE
            sav[mi] = 127;
#else
            sav[mi] = (sa0 >> (8 * (mi % 4))) & 0xff;
#endif
        }
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int blk = wn * N_PW + ni;
            b[ni] = read_op<KT_BYTES>(smem, A_BYTES, blk, 0, lane);
#ifdef NOSCALE
            sbv[ni] = 127;
#else
            sbv[ni] = (sb0 >> (8 * (ni % 4))) & 0xff;
#endif
        }
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++)   // N-major: B-operand reuse outer
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b[ni], sav[mi], sbv[ni]);
        __syncthreads();
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

// ============================================================================
// MS1: wave-level ping-pong on the MS0 scaffold. Same 256x256 + 32x32x64 +
// contiguous acc[4][2] + tiled scales. Adds: 2 LDS buffers (tic/toc, prefetch
// k+1 while computing k), warp_m team stagger, per-cluster s_setprio/s_barrier
// ping-pong handshake. Attempt-1 = correctness-first (conservative waits):
//   vmcnt(0) at each k-iter top (drains the prefetch of the tile we now compute)
//   lgkmcnt(0) before each cluster's MFMAs (drains the operand ds_reads).
// Overlap relaxation (raise the waits) is a Profiler co-tune AFTER 6/6 + occ-2.
//
// Cluster structure (researcher conformance card): 4 clusters (one per mi), each
// = reload a[mi] + 1 glds prefetch chunk (memory slot) + lgkmcnt(0) + s_barrier
// + s_setprio(2){2 MFMAs}s_setprio(0) + s_barrier [+ sched_barrier(0) at M-half].
//
// BARRIER BALANCE (hang-safety, provably equal counts): EVERY s_barrier is
// collective EXCEPT exactly two: prologue `if(warp_m==1)` (team1 +1 stagger) and
// pre-store `if(warp_m==0)` (team0 +1 realign). Per-team total = 9*k_tiles + 1.
//
// Issue ONE of a wave-split tile's glds issues (chunk-group i). Mirrors
// load_tile_lds's loop body for a single i so the prefetch can be spread across
// clusters (memory work overlapping the SIMD-mate's compute).
template <int ROWS, int KT_BYTES, int CPR>
__device__ __forceinline__ void issue_glds_one(char* smem, uint32_t lds_base,
        const char* gbase, int row_stride, int kt_byte, int wave_id, int lane, int i) {
    int chunk = i * 256 + wave_id * 64 + lane;
    int m  = chunk / CPR;
    int ck = chunk % CPR;
    const void* g = gbase + (size_t)m * row_stride + kt_byte + ck * 16;
    set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)((i * 256 + wave_id * 64) * 16)));
    async_load_lds_b128(smem, g);
}

// Attempt-2 tuning knobs (researcher: real overlap lever = glds vmcnt distance;
// compiler owns lgkmcnt for ds_read, so manual lgkmcnt is belt-and-suspenders):
//   PF_FRONT = 0 -> spread the A_ISSUES glds prefetch chunks over clusters 0..A_ISSUES-1 (HK paced)
//   PF_FRONT = 1 -> front-load all prefetch in cluster 0's memory slot (our occ-1 finding)
//   LGK_MANUAL = 1 -> manual wait_lgkmcnt(0) per cluster; 0 -> rely on compiler-inserted only
template <int M_TILE, int N_TILE, int K_STEP, int WARPS_M, int WARPS_N,
          typename OutT = float, int PF_FRONT = 0, int LGK_MANUAL = 1>
__global__ void __launch_bounds__(512, 2)
    lds_gemm_pp8_pingpong(const void* __restrict__ A, const void* __restrict__ B,
                          const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                          OutT* __restrict__ D, int N, int k_iters, int A_rs, int B_rs) {
    constexpr int NWAVES = WARPS_M * WARPS_N;
    static_assert(NWAVES == 8 && WARPS_M == 2 && WARPS_N == 4, "MS1 8-wave 2x4");
    constexpr int K64_PER_TILE = K_STEP / 64;
    static_assert(K64_PER_TILE == 1, "MS1 K_STEP=64 (shallow-K double buffer)");
    constexpr int KT_BYTES = K_STEP * 6 / 8;   // 48
    constexpr int ROW_CHUNKS    = KT_BYTES / 16;    // 3
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WARPS_M, N_PW = N_BLKS / WARPS_N;   // 4, 2
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int BUF  = A_BYTES + B_BYTES;     // one buffer's bytes (24KB)
    constexpr int HALF = NWAVES / 2;            // 4 waves cooperatively load one tile
    constexpr int A_ISSUES = (M_TILE * ROW_CHUNKS) / 256;   // 3 glds issues per tile half
    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    static_assert(M_PW <= 4 && N_PW <= 4, "scale unpack assumes <=4 blocks/group");

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WARPS_N, wn = wave % WARPS_N;   // wm in {0,1}, wn in {0..3}
    int warp_m = wm;                                // ping-pong team id

    int wg_m = blockIdx.x, wg_n = blockIdx.y;       // plain decode
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_rs;

    AccTileA acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    const int k_tiles = k_iters;                    // K64_PER_TILE==1
    const int sa_grp = wg_m * WARPS_M + wm;
    const int sb_grp = wg_n * WARPS_N + wn;

    // Issue one wave-split prefetch chunk-group i of tile kt into buffer `base`.
    auto prefetch_issue = [&](int kt, uint32_t base, int i) {
        int kb = kt * KT_BYTES;
        if (wave < HALF)
            issue_glds_one<M_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + 0, Ag, A_rs, kb, wave, lane, i);
        else
            issue_glds_one<N_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + A_BYTES, Bg, B_rs, kb,
                                                    wave - HALF, lane, i);
    };

    // PROLOGUE: prefetch tile 0 into buf0 (all issues), then team1 stagger.
#pragma unroll
    for (int i = 0; i < A_ISSUES; i++) prefetch_issue(0, 0, i);
    if (warp_m == 1) __builtin_amdgcn_s_barrier();   // stagger (team1 +1)

    for (int k = 0; k < k_tiles; k++) {
        uint32_t cur = (k & 1) ? BUF : 0;
        uint32_t nxt = (k & 1) ? 0 : BUF;
        // Current tile scales (vmem no-wait), drained by the vmcnt(0) below.
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + k) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + k) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int sa0 = asm_load_dword_nowait(pa);
        int sb0 = asm_load_dword_nowait(pb);
        wait_vmcnt(0);                       // cur tile glds (prev prefetch) + scales landed
        __builtin_amdgcn_s_barrier();        // collective: all waves see cur
        // B operands held for the whole k-iter (reused by all clusters); A reloaded per cluster.
        v6i b0 = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + 0, 0, lane);
        v6i b1 = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + 1, 0, lane);
#ifdef NOSCALE
        int sbv0 = 127, sbv1 = 127;
#else
        int sbv0 = (sb0 >> 0) & 0xff, sbv1 = (sb0 >> 8) & 0xff;
#endif
#pragma unroll
        for (int mi = 0; mi < M_PW; mi++) {
            // memory slot: prefetch the next tile's A_ISSUES glds chunks.
            if (k + 1 < k_tiles) {
                if constexpr (PF_FRONT) {    // front-load all issues in cluster 0
                    if (mi == 0) {
#pragma unroll
                        for (int i = 0; i < A_ISSUES; i++) prefetch_issue(k + 1, nxt, i);
                    }
                } else {                     // spread one chunk per cluster 0..A_ISSUES-1
                    if (mi < A_ISSUES) prefetch_issue(k + 1, nxt, mi);
                }
            }
            v6i a = read_op<KT_BYTES>(smem, cur + 0, wm * M_PW + mi, 0, lane);
#ifdef NOSCALE
            int sav = 127;
#else
            int sav = (sa0 >> (8 * mi)) & 0xff;
#endif
            if constexpr (LGK_MANUAL) wait_lgkmcnt(0);   // a + b0/b1 landed (else compiler-inserted)
            __builtin_amdgcn_s_barrier();    // collective: compute/memory handoff
            __builtin_amdgcn_s_setprio(2);
            mfma_scale_f32_32x32x64_fp6<0>(acc[mi][0], a, b0, sav, sbv0);
            mfma_scale_f32_32x32x64_fp6<0>(acc[mi][1], a, b1, sav, sbv1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();    // collective: handoff
            if (mi == 0 || mi == 2) __builtin_amdgcn_sched_barrier(0);  // M-half boundaries
        }
    }
    if (warp_m == 0) __builtin_amdgcn_s_barrier();   // realign (team0 +1)

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

// ============================================================================
// MS1 attempt-3: DEEP-PREFETCH RING. The decisive test. 2-buffer ping-pong caps
// lookahead at ~1 tile (~256-512cyc) < L2-miss latency (~880cyc) -> single misses
// still poke through even at occ-2. A NBUF-deep ring keeps NBUF-1 tiles in flight
// (~(NBUF-1)*256cyc lookahead) -> the occ-2 analog of deep-K's enlarged window,
// hiding the 880cyc tail across iters instead of within one shallow iter.
//
// Strips all manual waits (researcher Q1-Q4): NO top-of-iter vmcnt(0) (the
// compiler inserts its own vmcnt before cur's ds_read; the manual full drain only
// kills the in-flight ring), NO manual lgkmcnt (compiler owns ds_read dep), NO
// top-of-iter barrier (cluster barriers already fence cur's LDS readiness; scales
// are register loads needing no cross-wave barrier). Prefetch issued front-loaded
// at iter top and RIDES across the s_barriers, drained lazily by the compiler when
// its tile becomes cur. Best attempt-2 knobs baked in (front-load + compiler-lgkm).
//
// ⚠️ occ-2 razor: each ring level adds prefetch-address bookkeeping. LDS is NOT the
// gate (NBUF*24KB; NBUF=6 -> 144KB < 160KB, 1 block/CU). Registers are: F16 sits at
// 128/128 (2.00); if a deeper ring pushes arch/AGPR >128, F16 drops to occ-1 — that
// is itself part of the answer (occ-2 + deep-prefetch may be unaffordable for F16).
//
// BARRIER BALANCE: 8 collective/iter + the two conditionals (warp_m==1 prologue,
// warp_m==0 pre-store). Per-team total = 8*k_tiles + 1, equal -> no hang.
template <int M_TILE, int N_TILE, int K_STEP, int WARPS_M, int WARPS_N,
          typename OutT = float, int NBUF = 4, int TOPSYNC = 0>
__global__ void __launch_bounds__(512, 2)
    lds_gemm_pp8_ring(const void* __restrict__ A, const void* __restrict__ B,
                      const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                      OutT* __restrict__ D, int N, int k_iters, int A_rs, int B_rs) {
    constexpr int NWAVES = WARPS_M * WARPS_N;
    static_assert(NWAVES == 8 && WARPS_M == 2 && WARPS_N == 4, "MS1 8-wave 2x4");
    constexpr int K64_PER_TILE = K_STEP / 64;
    static_assert(K64_PER_TILE == 1, "MS1 K_STEP=64");
    static_assert(NBUF >= 2, "ring needs >=2 buffers");
    constexpr int KT_BYTES = K_STEP * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WARPS_M, N_PW = N_BLKS / WARPS_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int BUF = A_BYTES + B_BYTES;
    constexpr int HALF = NWAVES / 2;
    constexpr int A_ISSUES = (M_TILE * ROW_CHUNKS) / 256;
    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    static_assert(M_PW <= 4 && N_PW <= 4, "scale unpack assumes <=4/group");

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WARPS_N, wn = wave % WARPS_N;
    int warp_m = wm;
    int wg_m = blockIdx.x, wg_n = blockIdx.y;
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_rs;

    AccTileA acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    const int k_tiles = k_iters;
    const int sa_grp = wg_m * WARPS_M + wm;
    const int sb_grp = wg_n * WARPS_N + wn;

    auto prefetch_tile = [&](int kt, int slot) {
        uint32_t base = (uint32_t)(slot * BUF);
        int kb = kt * KT_BYTES;
#pragma unroll
        for (int i = 0; i < A_ISSUES; i++) {
            if (wave < HALF)
                issue_glds_one<M_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + 0, Ag, A_rs, kb, wave, lane, i);
            else
                issue_glds_one<N_TILE, KT_BYTES, ROW_CHUNKS>(smem, base + A_BYTES, Bg, B_rs, kb,
                                                        wave - HALF, lane, i);
        }
    };

    // PROLOGUE: fill ring with NBUF-1 tiles (lookahead), then team1 stagger.
#pragma unroll
    for (int t = 0; t < NBUF - 1; t++)
        if (t < k_tiles) prefetch_tile(t, t);
    if (warp_m == 1) __builtin_amdgcn_s_barrier();   // stagger (team1 +1)

    for (int k = 0; k < k_tiles; k++) {
        uint32_t cur = (uint32_t)((k % NBUF) * BUF);
        // Issue the deepest prefetch (tile k+NBUF-1) into the freed slot; it RIDES.
        int pf = k + (NBUF - 1);
        if (pf < k_tiles) prefetch_tile(pf, pf % NBUF);
        // current tile scales (vmem nowait; compiler drains with cur's ds_read)
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + k) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + k) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int sa0 = asm_load_dword_nowait(pa);
        int sb0 = asm_load_dword_nowait(pb);
        if constexpr (TOPSYNC) { wait_vmcnt(0); __builtin_amdgcn_s_barrier(); }  // diag: cross-wave handoff
        v6i b0 = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + 0, 0, lane);
        v6i b1 = read_op<KT_BYTES>(smem, cur + A_BYTES, wn * N_PW + 1, 0, lane);
#ifdef NOSCALE
        int sbv0 = 127, sbv1 = 127;
#else
        int sbv0 = (sb0 >> 0) & 0xff, sbv1 = (sb0 >> 8) & 0xff;
#endif
#pragma unroll
        for (int mi = 0; mi < M_PW; mi++) {
            v6i a = read_op<KT_BYTES>(smem, cur + 0, wm * M_PW + mi, 0, lane);
#ifdef NOSCALE
            int sav = 127;
#else
            int sav = (sa0 >> (8 * mi)) & 0xff;
#endif
            __builtin_amdgcn_s_barrier();    // compute/memory handoff (compiler owns waits)
            __builtin_amdgcn_s_setprio(2);
            mfma_scale_f32_32x32x64_fp6<0>(acc[mi][0], a, b0, sav, sbv0);
            mfma_scale_f32_32x32x64_fp6<0>(acc[mi][1], a, b1, sav, sbv1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            if (mi == 0 || mi == 2) __builtin_amdgcn_sched_barrier(0);
        }
    }
    if (warp_m == 0) __builtin_amdgcn_s_barrier();   // realign (team0 +1)

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

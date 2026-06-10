#pragma once
// HYBRID experiment: A staged in LDS (deep-K double-buffer, unchanged), B streamed
// DIRECT from HBM -> VGPR with a compile-time register ring buffer of depth PFD.
//
// Hypothesis under test: the slow second hop (LDS ds_read, ~258cyc exposed at sub head)
// is bypassed for B by reading B straight into registers; the per-tile MFMA window
// (48 MFMA x 32cyc = 1536cyc) hides the direct global_load latency IF the ring is deep
// enough to stay ahead. Blocker = VGPR budget (16 acc = 256 AccVGPR already). PFD is
// swept to find the deepest ring that keeps .vgpr_spill_count == 0.
//
// B operand global address is derived from the LDS path: the LDS B tile was filled by
//   LDS[m*KT_BYTES + ck*16] = global[Bg + m*B_row_bytes + kt_byte + ck*16]
// and read_op read LDS[(blk*32+lane%32)*KT_BYTES + sub*48 + (lane>>5)*24], i.e. 24
// contiguous bytes within row m. So the SAME 24 bytes live in global at:
//   Bg + (blk*32 + lane%32)*B_row_bytes + kt*KT_BYTES + sub*48 + (lane>>5)*24
#include "mxfp6_lds.hpp"
namespace mxfp6 {

// 24-byte (32 FP6 = one MFMA B operand) typed load HBM -> v6i (compiler-managed vmcnt,
// so SIInsertWaitcnts schedules + overlaps it; aligned(4) -> dwordx-style loads).
template <int KT_BYTES>
__device__ __forceinline__ v6i load_b_global(const char* Bg, int B_row_bytes,
                                             int blk, int kt_byte, int sub, int lane) {
    using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
    const char* p = Bg + (size_t)(blk * 32 + (lane & 31)) * B_row_bytes
                       + kt_byte + sub * 48 + (lane >> 5) * 24;
    v6i_a x = *reinterpret_cast<const v6i_a*>(p);
    return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
}

// Coalesced B operand load from a PRESHUFFLED B (preshuffle_B layout): per 32x64 tile,
// section0 = lane*16 (dwordx4), section1 = 1024 + lane*8 (dwordx2). Consecutive lanes read
// consecutive bytes -> fully coalesced (vs the raw column-major scatter of load_b_global).
__device__ __forceinline__ v6i load_b_shuf(const char* Bsh, int k64iters, int bg, int ki, int lane) {
    using v4i_a = int __attribute__((__vector_size__(16), __aligned__(4)));
    using v2i_a = int __attribute__((__vector_size__(8), __aligned__(4)));
    const char* base = Bsh + (size_t)(bg * k64iters + ki) * 1536;
    v4i_a lo = *reinterpret_cast<const v4i_a*>(base + lane * 16);
    v2i_a hi = *reinterpret_cast<const v2i_a*>(base + 1024 + lane * 8);
    return v6i{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1]};
}

// Issue A cooperative buffer_load_lds chunks [i0,i1) for ONE tile (MUBUF, M0-implicit).
// Split out of load_tile_lds so the hybrid can DRIP A's 9 loads across the compute window
// (1 per MFMA quartet) instead of bursting them after the barrier — the burst is the top
// ATT stall (318 cyc/hit issue backpressure). s_nop guards the SALU-writes-M0 -> load(lds)
// 1-wait-state hazard (drip places set_m0 + load adjacent, unlike the burst the compiler
// spread out). Caller manages vmcnt (these loads are compiler-invisible, M0-implicit asm).
template <int CPR>
__device__ __forceinline__ void issue_A_chunks(uint32_t lds_base, int row_stride, int kt_byte,
                                               int wave, int lane, const v4i& rsrc, int i0, int i1) {
    for (int i = i0; i < i1; i++) {
        int chunk = i * 256 + wave * 64 + lane;
        int m = chunk / CPR, ck = chunk % CPR;
        uint32_t voff = (uint32_t)(m * row_stride + kt_byte + ck * 16);
        set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)((i * 256 + wave * 64) * 16)));
        asm volatile("s_nop 0\n buffer_load_dwordx4 %0, %1, 0 offen lds"
                     : : "v"(voff), "s"(rsrc) : "memory");
    }
}

// HYBRID + DRIP-A: same as lds_gemm_hybrid (A-LDS, B-direct coalesced ring) but A's 9
// cooperative buffer_loads for the NEXT tile are dripped across THIS tile's MFMA quartets
// instead of bursted. HARD_WAIT=true puts wait_vmcnt(0) before each DB barrier (hard RAW
// guarantee for the dripped A; cheap because A is issued early). SHUF is forced true.
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          int SWZ = 0, bool DB = true, typename OutT = float, int PFD = 6, bool HARD_WAIT = true>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_hybrid_dripA(const void* __restrict__ A, const void* __restrict__ B,
                          const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                          OutT* __restrict__ D, int N, int k_iters, int A_row_bytes, int B_row_bytes) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int K64_PER_TILE = K_TILE / 64;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WAVES_M, N_PW = N_BLKS / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES;
    constexpr int NB = K64_PER_TILE * N_PW;            // 12 b-stream positions / tile
    constexpr int ISSUES_A = (M_TILE * ROW_CHUNKS) / 256;  // 9 A loads / tile

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m, wg_n;
    if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs; wg_n = fn + r % gs;
    } else { wg_m = blockIdx.x; wg_n = blockIdx.y; }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_row_bytes;

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

    // A buffer descriptor (Ag constant across the kernel)
    uint64_t ab = reinterpret_cast<uint64_t>(Ag);
    v4i arsrc{(int)(uint32_t)ab, (int)((uint32_t)(ab >> 32) & 0xFFFF), (int)0x7FFFFFFF, (int)0x00020000};

    auto load_scales = [&](int kt, int (*sa)[NDA], int (*sb)[NDB]) {
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int ta[K64_PER_TILE], tb[K64_PER_TILE];
        asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
        asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
        for (int sub = 0; sub < K64_PER_TILE; sub++) { sa[sub][0] = ta[sub]; sb[sub][0] = tb[sub]; }
    };

    // compute tile kt_cur from buffer `cur`; if adrip, drip A(kt_nxt) into buffer nxt_base.
    auto compute = [&](uint32_t cur, uint32_t nxt_base, int kt_cur, int kt_nxt, bool adrip,
                       const int (*sa)[NDA], const int (*sb)[NDB]) {
        int kb_nxt = kt_nxt * KT_BYTES;
        v6i bring[PFD];
        int sbring[PFD];
#pragma unroll
        for (int q = 0; q < PFD; q++) if (q < NB) {
            int s = q / N_PW, n = q % N_PW, blk = wn * N_PW + n;
            bring[q] = load_b_shuf(reinterpret_cast<const char*>(B), k_iters, wg_n * N_BLKS + blk, kt_cur * K64_PER_TILE + s, lane);
            sbring[q] = (sb[s][n / 4] >> (8 * (n % 4))) & 0xff;
        }
        v6i a[M_PW];
        int sav[M_PW];
#pragma unroll
        for (int p = 0; p < NB; p++) {
            int sub = p / N_PW, ni = p % N_PW;
            if (ni == 0) {
#pragma unroll
                for (int mi = 0; mi < M_PW; mi++) {
                    int blk = wm * M_PW + mi;
                    a[mi] = read_op<KT_BYTES>(smem, cur, blk, sub, lane);
                    sav[mi] = (sa[sub][mi / 4] >> (8 * (mi % 4))) & 0xff;
                }
            }
            v6i b_cur = bring[p % PFD];
            int sbv_cur = sbring[p % PFD];
            if (p + PFD < NB) {
                int np = p + PFD, ns = np / N_PW, nn = np % N_PW, nblk = wn * N_PW + nn;
                bring[(p + PFD) % PFD] = load_b_shuf(reinterpret_cast<const char*>(B), k_iters, wg_n * N_BLKS + nblk, kt_cur * K64_PER_TILE + ns, lane);
                sbring[(p + PFD) % PFD] = (sb[ns][nn / 4] >> (8 * (nn % 4))) & 0xff;
            }
            // DRIP A: one chunk per quartet for the first ISSUES_A quartets (early => margin)
            if (adrip && p < ISSUES_A)
                issue_A_chunks<ROW_CHUNKS>(nxt_base, A_row_bytes, kb_nxt, wave, lane, arsrc, p, p + 1);
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b_cur, sav[mi], sbv_cur);
        }
    };

    int sa0[K64_PER_TILE][NDA], sa1[K64_PER_TILE][NDA], sb0[K64_PER_TILE][NDB], sb1[K64_PER_TILE][NDB];
    // prologue: tile 0 A bursted into buf0 (no compute to drip into yet) + its scales
    load_scales(0, sa0, sb0);
    issue_A_chunks<ROW_CHUNKS>(0, A_row_bytes, 0, wave, lane, arsrc, 0, ISSUES_A);
    int kt = 0;
    for (; kt + 1 < k_tiles; kt += 2) {
        if (HARD_WAIT) wait_vmcnt(0);
        __syncthreads();
        load_scales(kt + 1, sa1, sb1);
        compute(0, A_BYTES, kt, kt + 1, true, sa0, sb0);          // compute buf0, drip A(kt+1)->buf1
        bool pf = (kt + 2 < k_tiles);
        if (HARD_WAIT) wait_vmcnt(0);
        __syncthreads();
        if (pf) load_scales(kt + 2, sa0, sb0);
        compute(A_BYTES, 0, kt + 1, kt + 2, pf, sa1, sb1);        // compute buf1, drip A(kt+2)->buf0
    }
    if (kt < k_tiles) { wait_vmcnt(0); __syncthreads(); compute(0, 0, kt, 0, false, sa0, sb0); }

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

template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          int SWZ = 0, bool DB = true, typename OutT = float, int PFD = 4, bool SHUF = false>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_hybrid(const void* __restrict__ A, const void* __restrict__ B,
                    const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                    OutT* __restrict__ D, int N, int k_iters, int A_row_bytes, int B_row_bytes) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int ROW_CHUNKS = KT_BYTES / 16;
    constexpr int K64_PER_TILE = K_TILE / 64;
    constexpr int M_BLKS = M_TILE / 32, N_BLKS = N_TILE / 32;
    constexpr int M_PW = M_BLKS / WAVES_M, N_PW = N_BLKS / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES;   // ONLY A staged in LDS now
    constexpr int NB = K64_PER_TILE * N_PW;      // b-stream positions per tile (=12)

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    int wg_m, wg_n;
    if constexpr (SWZ > 0) {
        int mb = gridDim.x, nb = gridDim.y, pid = blockIdx.y * mb + blockIdx.x;
        const int G = SWZ, span = G * mb;
        int grp = pid / span, fn = grp * G, gs = (nb - fn) < G ? (nb - fn) : G, r = pid % span;
        wg_m = r / gs; wg_n = fn + r % gs;
    } else { wg_m = blockIdx.x; wg_n = blockIdx.y; }
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_row_bytes;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_row_bytes;

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

    // prefetch ONLY A into LDS buffer `base`, + A & B scales (regs, tiled layout).
    auto prefetch_A = [&](int kt, uint32_t base, int (*sa)[NDA], int (*sb)[NDB]) {
        int kb = kt * KT_BYTES;
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * K64_PER_TILE * SB_PAD;
        int ta[K64_PER_TILE], tb[K64_PER_TILE];
        asm_load_dwordxN_nowait(ta, pa, K64_PER_TILE);
        asm_load_dwordxN_nowait(tb, pb, K64_PER_TILE);
#pragma unroll
        for (int sub = 0; sub < K64_PER_TILE; sub++) { sa[sub][0] = ta[sub]; sb[sub][0] = tb[sub]; }
        issue_tile<M_TILE, KT_BYTES, ROW_CHUNKS>(smem, base, Ag, A_row_bytes, kb, wave, lane);
    };

    // compute one tile: A operands from LDS (cur); B streamed direct from HBM with a
    // depth-PFD register ring so each b is in flight ~PFD quartets before it's consumed.
    auto compute = [&](uint32_t cur, int kt, const int (*sa)[NDA], const int (*sb)[NDB]) {
        int kb = kt * KT_BYTES;
        v6i bring[PFD];
        int sbring[PFD];
#pragma unroll
        for (int q = 0; q < PFD; q++) if (q < NB) {
            int s = q / N_PW, n = q % N_PW, blk = wn * N_PW + n;
            if constexpr (SHUF)
                bring[q] = load_b_shuf(reinterpret_cast<const char*>(B), k_iters, wg_n * N_BLKS + blk, kt * K64_PER_TILE + s, lane);
            else
                bring[q] = load_b_global<KT_BYTES>(Bg, B_row_bytes, blk, kb, s, lane);
            sbring[q] = (sb[s][n / 4] >> (8 * (n % 4))) & 0xff;
        }
        v6i a[M_PW];
        int sav[M_PW];
#pragma unroll
        for (int p = 0; p < NB; p++) {
            int sub = p / N_PW, ni = p % N_PW;
            if (ni == 0) {  // fresh sub: pull this sub's A operands from LDS
#pragma unroll
                for (int mi = 0; mi < M_PW; mi++) {
                    int blk = wm * M_PW + mi;
                    a[mi] = read_op<KT_BYTES>(smem, cur, blk, sub, lane);
                    sav[mi] = (sa[sub][mi / 4] >> (8 * (mi % 4))) & 0xff;
                }
            }
            v6i b_cur = bring[p % PFD];
            int sbv_cur = sbring[p % PFD];
            if (p + PFD < NB) {  // keep the ring full: prefetch p+PFD
                int np = p + PFD, ns = np / N_PW, nn = np % N_PW, nblk = wn * N_PW + nn;
                if constexpr (SHUF)
                    bring[(p + PFD) % PFD] = load_b_shuf(reinterpret_cast<const char*>(B), k_iters, wg_n * N_BLKS + nblk, kt * K64_PER_TILE + ns, lane);
                else
                    bring[(p + PFD) % PFD] = load_b_global<KT_BYTES>(Bg, B_row_bytes, nblk, kb, ns, lane);
                sbring[(p + PFD) % PFD] = (sb[ns][nn / 4] >> (8 * (nn % 4))) & 0xff;
            }
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
                mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b_cur, sav[mi], sbv_cur);
        }
    };

    int sa0[K64_PER_TILE][NDA], sa1[K64_PER_TILE][NDA], sb0[K64_PER_TILE][NDB], sb1[K64_PER_TILE][NDB];
    if constexpr (DB) {
        prefetch_A(0, 0, sa0, sb0);
        int kt = 0;
        for (; kt + 1 < k_tiles; kt += 2) {
            __syncthreads();
            prefetch_A(kt + 1, A_BYTES, sa1, sb1);
            compute(0, kt, sa0, sb0);
            bool pf = (kt + 2 < k_tiles);
            __syncthreads();
            if (pf) prefetch_A(kt + 2, 0, sa0, sb0);
            compute(A_BYTES, kt + 1, sa1, sb1);
        }
        // odd tail (buf0 already loaded). When k_tiles==1 the loop above never ran, so
        // there is NO compute-window margin for the prologue A buffer_load to land ->
        // explicit wait_vmcnt(0) (the A loads are M0-implicit asm, invisible to the
        // compiler; __syncthreads is lgkmcnt+s_barrier only and does NOT wait on vmem).
        if (kt < k_tiles) { wait_vmcnt(0); __syncthreads(); compute(0, kt, sa0, sb0); }
    } else {
        for (int kt = 0; kt < k_tiles; kt++) {
            prefetch_A(kt, 0, sa0, sb0);
            wait_vmcnt(0); __syncthreads();
            compute(0, kt, sa0, sb0);
            __syncthreads();
        }
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
} // namespace mxfp6

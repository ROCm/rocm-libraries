#pragma once
// LDS deep-K staged MXFP6 GEMM kernel + tile-grouped scale layout.
// Paradigm: deep K staged in LDS enlarges the MFMA window > load latency
// (32x32x64 MFMA, NOT CK's 16x16x128). Beats V17 +3~5% on large aligned shapes.
// Extracted from test_lds.cpp; shared by test_lds.cpp and the combined dispatcher.
#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include <vector>
namespace mxfp6 {
// Tile-grouped scale layout: a wave's `group` consecutive 32-blocks AND its SUBS
// k64 sub-slabs of one K-tile become contiguous per lane, so the kernel fetches a
// whole K-tile's scales in ONE dwordx{SUBS} load (NDA=1). Layout:
//   out[(((g*k_tiles+kt)*64 + lane)*SUBS + sub)*group_pad + j]
//        = scale of (block g*group+j, k64 = kt*SUBS+sub)
struct TiledScale { std::vector<uint8_t> data; };
static TiledScale tile_scale(const PreprocessedScale& ps, int group, int subs) {
    int group_pad = (group + 3) / 4 * 4;
    int k_tiles = ps.k64_iters / subs;
    int ng = ps.num_tiles / group;
    TiledScale ts;
    ts.data.assign((size_t)ng * k_tiles * 64 * subs * group_pad, 0);
    for (int g = 0; g < ng; g++)
        for (int kt = 0; kt < k_tiles; kt++)
            for (int lane = 0; lane < 64; lane++)
                for (int sub = 0; sub < subs; sub++)
                    for (int j = 0; j < group; j++) {
                        int k64 = kt * subs + sub, tile = g * group + j;
                        ts.data[(((size_t)(g * k_tiles + kt) * 64 + lane) * subs + sub) * group_pad + j] =
                            ps.data[(size_t)(tile * ps.k64_iters + k64) * 64 + lane];
                    }
    return ts;
}

// Cooperative async load of a (ROWS x K_TILE) FP6 tile into LDS via
// global_load_lds_dwordx4. The tile's flattened layout in LDS is row-major:
//   LDS[lds_base + (m*KT_BYTES + ck*16)] = global[gbase + m*row_stride + kt_byte + ck*16]
// where KT_BYTES = K_TILE*6/8 and CPR = KT_BYTES/16 chunks per row. Each lane
// fetches its own 16B; global_load_lds forces LDS dest = M0 + lane*16, so we walk
// chunks in (m-major, ck-minor) order = exactly the row-major LDS layout above.
template <int ROWS, int KT_BYTES, int CPR>
__device__ __forceinline__ void load_tile_lds(char* smem, uint32_t lds_base,
                                              const char* gbase, int row_stride,
                                              int kt_byte, int wave_id, int lane) {
    constexpr int TOTAL = ROWS * CPR;       // total 16B chunks in the tile
    static_assert(TOTAL % 256 == 0, "tile chunks must be a multiple of 256 (4 waves x 64)");
    constexpr int ISSUES = TOTAL / 256;
#pragma unroll
    for (int i = 0; i < ISSUES; i++) {
        int chunk = i * 256 + wave_id * 64 + lane;
        int m  = chunk / CPR;
        int ck = chunk % CPR;
        const void* g = gbase + (size_t)m * row_stride + kt_byte + ck * 16;
        // M0 must be scalar (SGPR); wave_id is uniform per-wave but the compiler
        // treats tid/64 as a VGPR -> readfirstlane forces it into an SGPR.
        set_m0(__builtin_amdgcn_readfirstlane(lds_base + (uint32_t)((i * 256 + wave_id * 64) * 16)));
        async_load_lds_b128(smem, g);  // -> LDS[lds_base + (i*256+wave*64+lane)*16]
    }
}

// ds_read one 32x32 MFMA operand (32 FP6 / lane = 24B) from an LDS tile staged by
// load_tile_lds. blk = 32-row block within the tile, sub = which 64-K sub-slab,
// kh/m from lane. Operand layout matches load: row (blk*32+lane%32) slab at
// row*KT_BYTES, sub-slab at sub*48, k-half at (lane/32)*24.
template <int KT_BYTES>
__device__ __forceinline__ v6i read_op(const char* smem, uint32_t lds_base, int blk,
                                      int sub, int lane) {
    uint32_t off = lds_base + (uint32_t)((blk * 32 + (lane & 31)) * KT_BYTES + sub * 48 +
                                         (lane >> 5) * 24);
    return ds_read_fp6x32_plain(smem, off);
}

// Inline-asm global load to VGPR WITHOUT a waitcnt (caller manages vmcnt). Keeps
// scale loads off the compiler's vmcnt accounting so they don't force a drain of
// the in-flight global_load_lds prefetch (the typed-load/glds vmcnt conflict that
// otherwise costs ~13%). "memory" clobber preserves ordering vs the manual waits.
__device__ __forceinline__ int asm_load_dword_nowait(const void* a) {
    int v;
    asm volatile("global_load_dword %0, %1, off" : "=v"(v) : "v"(a) : "memory");
    return v;
}

// Wide no-wait loads: bring SUBS contiguous scale dwords for a whole K-tile in ONE
// instruction (tile-grouped scale layout) instead of one dword per sub. Cuts scale
// vmem op count ~SUBS x. out[] gets the dwords. Only NW in {2,3,4} (gfx950 dwordx3).
__device__ __forceinline__ void asm_load_dwordxN_nowait(int* out, const void* a, int nw) {
    if (nw == 2) {
        int2 v; asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v.x; out[1] = v.y;
    } else if (nw == 3) {
        v3i v; asm volatile("global_load_dwordx3 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
    } else if (nw == 4) {
        v4i v; asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; out[3] = v[3];
    } else {
        out[0] = asm_load_dword_nowait(a);
    }
}

// Issue (no wait) the async loads for one K-tile of A or B into an LDS buffer.
// Returns nothing; caller manages vmcnt. ISSUES = ROWS*CPR/256 loads.
template <int ROWS, int KT_BYTES, int CPR>
__device__ __forceinline__ void issue_tile(char* smem, uint32_t lds_base, const char* gbase,
                                           int row_stride, int kt_byte, int wave_id, int lane) {
    load_tile_lds<ROWS, KT_BYTES, CPR>(smem, lds_base, gbase, row_stride, kt_byte, wave_id, lane);
}

// Double-buffered deep-K LDS GEMM: prefetch tile kt+1 while computing tile kt.
// vmcnt(LPT) after issuing the prefetch drains cur's LPT loads (vmcnt decrements
// in issue order) while nxt's LPT stay in flight -> real load/compute overlap.
template <int M_TILE, int N_TILE, int K_TILE, int WAVES_M, int WAVES_N, int MIN_OCC = 1,
          int SWZ = 0, bool DB = true>
__global__ void __launch_bounds__(256, MIN_OCC)
    lds_gemm_db(const void* __restrict__ A, const void* __restrict__ B,
                const uint8_t* __restrict__ sA, const uint8_t* __restrict__ sB,
                float* __restrict__ D, int N, int k_iters, int A_rs, int B_rs) {
    constexpr int KT_BYTES = K_TILE * 6 / 8;
    constexpr int A_CPR    = KT_BYTES / 16;
    constexpr int SUBS     = K_TILE / 64;
    constexpr int MB = M_TILE / 32, NB = N_TILE / 32;
    constexpr int M_PW = MB / WAVES_M, N_PW = NB / WAVES_N;
    constexpr int A_BYTES = M_TILE * KT_BYTES, B_BYTES = N_TILE * KT_BYTES;
    constexpr int BUF = A_BYTES + B_BYTES;                 // one buffer's bytes
    constexpr int LPT = (M_TILE * A_CPR + N_TILE * A_CPR) / 256;  // loads per tile

    extern __shared__ char smem[];
    int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    int wm = wave / WAVES_N, wn = wave % WAVES_N;
    // L2-aware WG remap: walk WGs down M within a band of SWZ consecutive n-blocks so
    // neighboring in-flight WGs share a B band and keep it hot in L2 (V17 +4.4%).
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
    const char* Ag = reinterpret_cast<const char*>(A) + (size_t)(wg_m * M_TILE) * A_rs;
    const char* Bg = reinterpret_cast<const char*>(B) + (size_t)(wg_n * N_TILE) * B_rs;

    AccTileA acc[M_PW][N_PW];
#pragma unroll
    for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) clear_acc(acc[mi][ni]);

    // Coalesced scales (see preshuffle_scale): a wave's M_PW (N_PW) consecutive
    // 32-blocks are byte-contiguous per lane -> one wide load per k64. Loaded via
    // asm_load_dword_nowait (manual vmcnt) and double-buffered alongside the LDS
    // tiles, so they never force a compiler vmcnt drain of the glds prefetch.
    constexpr int SA_PAD = ((M_PW + 3) / 4) * 4, SB_PAD = ((N_PW + 3) / 4) * 4;
    constexpr int NDA = SA_PAD / 4, NDB = SB_PAD / 4;
    static_assert(NDA == 1 && NDB == 1, "tiled-scale path assumes <=4 blocks/wave");
    constexpr int LPT_TOT = LPT + 2;  // glds + 1 wide A-scale + 1 wide B-scale load
    int sa_grp = wg_m * WAVES_M + wm, sb_grp = wg_n * WAVES_N + wn;
    int k_tiles = k_iters / SUBS;

    // Prefetch tile kt into LDS buffer `base` + scale regs sa/sb (asm, manual vmcnt).
    // Scales use the TILE-GROUPED layout: a wave's SUBS k64 scales are contiguous per
    // lane, so ONE dwordx{SUBS} load brings the whole tile's scales (vs SUBS loads).
    // sa/sb are caller-named arrays (compile-time buf0/buf1) -> no dynamic index/spill.
    auto prefetch = [&](int kt, uint32_t base, int (*sa)[NDA], int (*sb)[NDB]) {
        int kb = kt * KT_BYTES;
        issue_tile<M_TILE, KT_BYTES, A_CPR>(smem, base + 0, Ag, A_rs, kb, wave, lane);
        issue_tile<N_TILE, KT_BYTES, A_CPR>(smem, base + A_BYTES, Bg, B_rs, kb, wave, lane);
        const char* pa = reinterpret_cast<const char*>(sA) +
                         (size_t)((sa_grp * k_tiles + kt) * 64 + lane) * SUBS * SA_PAD;
        const char* pb = reinterpret_cast<const char*>(sB) +
                         (size_t)((sb_grp * k_tiles + kt) * 64 + lane) * SUBS * SB_PAD;
        int ta[SUBS], tb[SUBS];
        asm_load_dwordxN_nowait(ta, pa, SUBS);
        asm_load_dwordxN_nowait(tb, pb, SUBS);
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) { sa[sub][0] = ta[sub]; sb[sub][0] = tb[sub]; }
    };
    auto compute = [&](uint32_t cur, const int (*sa)[NDA], const int (*sb)[NDB]) {
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) {
            v6i a[M_PW], b[N_PW];
            int sav[M_PW], sbv[N_PW];
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
#pragma unroll
            for (int ni = 0; ni < N_PW; ni++) {
                int blk = wn * N_PW + ni;
                b[ni] = read_op<KT_BYTES>(smem, cur + A_BYTES, blk, sub, lane);
#ifdef NOSCALE
                sbv[ni] = 127;
#else
                sbv[ni] = (sb[sub][ni / 4] >> (8 * (ni % 4))) & 0xff;
#endif
            }
#pragma unroll
            for (int mi = 0; mi < M_PW; mi++)
#pragma unroll
                for (int ni = 0; ni < N_PW; ni++)
                    mfma_scale_f32_32x32x64_fp6<0>(acc[mi][ni], a[mi], b[ni], sav[mi], sbv[ni]);
        }
    };

    int sa0[SUBS][NDA], sa1[SUBS][NDA], sb0[SUBS][NDB], sb1[SUBS][NDB];
    if constexpr (DB) {
        // 2x-unrolled ping-pong: buf0 for even tiles, buf1 for odd. Compile-time
        // buffer/scale-array selection (no dynamic index). Prefetch next tile
        // (glds+scales) while computing current; wait_vmcnt(LPT_TOT) drains current's
        // loads in issue order while next's stay in flight -> real load/compute overlap.
        prefetch(0, 0, sa0, sb0);  // prologue: tile 0 -> buf0
        int kt = 0;
        for (; kt + 1 < k_tiles; kt += 2) {
            prefetch(kt + 1, BUF, sa1, sb1);          // tile kt+1 -> buf1
            wait_vmcnt(LPT_TOT); __syncthreads();
            compute(0, sa0, sb0);                      // compute buf0
            __syncthreads();
            bool pf = (kt + 2 < k_tiles);
            if (pf) prefetch(kt + 2, 0, sa0, sb0);     // tile kt+2 -> buf0
            wait_vmcnt(pf ? LPT_TOT : 0); __syncthreads();
            compute(BUF, sa1, sb1);                    // compute buf1
            __syncthreads();
        }
        if (kt < k_tiles) {                            // odd tail (buf0 already loaded)
            wait_vmcnt(0); __syncthreads();
            compute(0, sa0, sb0);
        }
    } else {
        // Single LDS buffer (for K-tiles too deep to double-buffer within 160KB LDS,
        // e.g. 256x256 KT256). Load and compute serialize (no overlap) but the deep
        // MFMA window amortizes; lets us reach valid SUBS|k_iters deep tiles.
        for (int kt = 0; kt < k_tiles; kt++) {
            prefetch(kt, 0, sa0, sb0);
            wait_vmcnt(0); __syncthreads();
            compute(0, sa0, sb0);
            __syncthreads();
        }
    }

#pragma unroll
    for (int mi = 0; mi < M_PW; mi++) {
        int m = wg_m * M_TILE + (wm * M_PW + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < N_PW; ni++) {
            int n = wg_n * N_TILE + (wn * N_PW + ni) * 32;
            store_acc_f32(D, N, acc[mi][ni], m, n);
        }
    }
}
} // namespace mxfp6

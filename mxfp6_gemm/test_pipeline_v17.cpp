#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"

using namespace mxfp6;

// v17: production dispatcher = v14/v15 tile-shrink + V2 mixed-accumulator N640 +
// depth-1 software prefetch (occ1) + L2-aware WG swizzle (large-N square grids).
// One kernel: NPW_A cols in AGPR, NPW_V cols overflow into idle Arch VGPR.
//   N128:(2,0)occ4  N256:(4,0)occ2  N512:(8,0)occ1  N640:(8,2)occ1[V2]
// V2 (N640, 20 acc) only applies when N%640==0 (640=2^7*5, excludes pow2 N) and
// its grid lands closer to a 256-CU multiple — then +5~12% (real LLM dims: 5120).
// SWZ=16 on N512 when N>=8192 (full n-band) keeps B hot in L2: +0.8~4.4% — notably
// breaks the pow2-square plateau (8192^3: 1491 -> 1557). See choose_swz.
template <int M_TILE, int NPW_A, int NPW_V, int N_WAVES, int MIN_OCC, int WAVES_M = 2,
          int WAVES_N = 2, int SWZ = 0>
__global__ void __launch_bounds__(256, MIN_OCC)
    mxfp6_gemm_pipeline(const void* __restrict__ A_packed, const void* __restrict__ B_shuffled,
                        const uint8_t* __restrict__ scale_A, const uint8_t* __restrict__ scale_B,
                        float* __restrict__ D, int D_stride, int k_iters, int A_row_stride) {
    constexpr int M_PER_WAVE = (M_TILE / 32) / WAVES_M;
    constexpr int NPW = NPW_A + NPW_V;
    constexpr int N_TILE = WAVES_N * NPW * 32;
    constexpr int NPW_V_ALLOC = NPW_V > 0 ? NPW_V : 1;
    int tid = threadIdx.x, wave_id = tid / 64, lane = tid % 64;
    int wave_m = wave_id / WAVES_N, wave_n = wave_id % WAVES_N;
    // L2-aware WG remap (swz): instead of (blockIdx.x,y)=(m,n), walk WGs down M within a
    // band of SWZ consecutive n-blocks, so neighboring WGs in flight share the same B band
    // and keep it hot in L2. Only pays off on large square grids (nb>=SWZ); see choose_swz.
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
    int m_tile_base = wg_m * (M_TILE / 32), n_tile_base = wg_n * (N_TILE / 32);

    AccTileA acc_a[M_PER_WAVE][NPW_A];
    AccTileV acc_v[M_PER_WAVE][NPW_V_ALLOC];
#pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++) {
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++) clear_acc(acc_a[mi][ni]);
#pragma unroll
        for (int ni = 0; ni < NPW_V; ni++) clear_acc(acc_v[mi][ni]);
    }
    int lane_m = lane & 31, lane_kh = lane >> 5;
    auto load_B = [&](int t) -> v6i {
        const char* p = reinterpret_cast<const char*>(B_shuffled) + t * 1536;
        float4 lo = *reinterpret_cast<const float4*>(p + lane * 16);
        double hr = *reinterpret_cast<const double*>(p + 1024 + lane * 8);
        int2 hi = *reinterpret_cast<const int2*>(&hr);
        return v6i{__float_as_int(lo.x),
                   __float_as_int(lo.y),
                   __float_as_int(lo.z),
                   __float_as_int(lo.w),
                   hi.x,
                   hi.y};
    };
    auto load_A = [&](int m_tile, int ki) -> v6i {
        using v6i_a = int __attribute__((__vector_size__(24), __aligned__(4)));
        const char* a = reinterpret_cast<const char*>(A_packed) +
                        (size_t)(m_tile * 32 + lane_m) * A_row_stride + ki * 48 + lane_kh * 24;
        v6i_a x = *reinterpret_cast<const v6i_a*>(a);
        return v6i{x[0], x[1], x[2], x[3], x[4], x[5]};
    };
    // Coalesced scales (see preshuffle_scale): a wave's M_PER_WAVE / NPW consecutive
    // 32-blocks are packed byte-contiguous per lane, so each lane fetches all of them in
    // ONE wide load (group<=4 -> dword, =8 -> dwordx2) + byte-extract in VGPR — killing
    // the per-MFMA vmcnt cascade that gated on one scale byte each. group=(tile_base+wave*g)/g.
    constexpr int SA_PAD = ((M_PER_WAVE + 3) / 4) * 4, SB_PAD = ((NPW + 3) / 4) * 4;
    using sav = int __attribute__((__vector_size__(SA_PAD), __aligned__(4)));
    using sbv = int __attribute__((__vector_size__(SB_PAD), __aligned__(4)));
    int sa_grp = wg_m * WAVES_M + wave_m, sb_grp = wg_n * WAVES_N + wave_n;
    auto ld_a = [&](v6i* a, int ki) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) a[mi] = load_A(m_tile_base + wave_m * M_PER_WAVE + mi, ki);
    };
    auto ld_b = [&](v6i* b, int ki) {
#pragma unroll
        for (int ni = 0; ni < NPW; ni++) b[ni] = load_B((n_tile_base + wave_n * NPW + ni) * k_iters + ki);
    };
    auto ld_sa = [&](int* sa, int ki) {
        sav w = *reinterpret_cast<const sav*>(reinterpret_cast<const char*>(scale_A) +
                                              (size_t)((sa_grp * k_iters + ki) * 64 + lane) * SA_PAD);
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) sa[mi] = (w[mi / 4] >> (8 * (mi % 4))) & 0xff;
    };
    auto ld_sb = [&](int* sb, int ki) {
        sbv w = *reinterpret_cast<const sbv*>(reinterpret_cast<const char*>(scale_B) +
                                              (size_t)((sb_grp * k_iters + ki) * 64 + lane) * SB_PAD);
#pragma unroll
        for (int ni = 0; ni < NPW; ni++) sb[ni] = (w[ni / 4] >> (8 * (ni % 4))) & 0xff;
    };
    auto do_mfma = [&](v6i* a, v6i* b, int* sa, int* sb) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WAVE; mi++) {
#pragma unroll
            for (int ni = 0; ni < NPW_A; ni++)
                mfma_scale_f32_32x32x64_fp6<0>(acc_a[mi][ni], a[mi], b[ni], sa[mi], sb[ni]);
#pragma unroll
            for (int ni = 0; ni < NPW_V; ni++)
                mfma_scale_f32_32x32x64_fp6<0>(acc_v[mi][ni], a[mi], b[NPW_A + ni], sa[mi],
                                               sb[NPW_A + ni]);
        }
    };
    // Depth-1 software prefetch ONLY for occ1 (latency-bound) tiles: issuing ki+1's loads
    // before ki's MFMAs overlaps the ~876cyc load latency via MLP (+1~15%). Compile-time
    // double buffer (dynamic reg-array index spills). For occ>=2 tiles (N128/N256) the 2nd
    // wave already hides latency, so the extra prefetch registers only cut occupancy -> skip.
    if constexpr (MIN_OCC == 1) {
        v6i ac[M_PER_WAVE], bc[NPW]; int sac[M_PER_WAVE], sbc[NPW];
        ld_a(ac, 0); ld_b(bc, 0); ld_sa(sac, 0); ld_sb(sbc, 0);
        for (int ki = 0; ki < k_iters; ki++) {
            v6i an[M_PER_WAVE], bn[NPW]; int san[M_PER_WAVE], sbn[NPW];
            if (ki + 1 < k_iters) { ld_a(an, ki + 1); ld_b(bn, ki + 1); ld_sa(san, ki + 1); ld_sb(sbn, ki + 1); }
            do_mfma(ac, bc, sac, sbc);
#pragma unroll
            for (int mi = 0; mi < M_PER_WAVE; mi++) { ac[mi] = an[mi]; sac[mi] = san[mi]; }
#pragma unroll
            for (int ni = 0; ni < NPW; ni++) { bc[ni] = bn[ni]; sbc[ni] = sbn[ni]; }
        }
    } else {
        for (int ki = 0; ki < k_iters; ki++) {
            v6i ac[M_PER_WAVE], bc[NPW]; int sac[M_PER_WAVE], sbc[NPW];
            ld_a(ac, ki); ld_b(bc, ki); ld_sa(sac, ki); ld_sb(sbc, ki);
            do_mfma(ac, bc, sac, sbc);
        }
    }
#pragma unroll
    for (int mi = 0; mi < M_PER_WAVE; mi++) {
        int m = wg_m * M_TILE + (wave_m * M_PER_WAVE + mi) * 32;
#pragma unroll
        for (int ni = 0; ni < NPW_A; ni++) {
            int n = wg_n * N_TILE + (wave_n * NPW + ni) * 32;
            store_acc_f32(D, D_stride, acc_a[mi][ni], m, n);
        }
#pragma unroll
        for (int ni = 0; ni < NPW_V; ni++) {
            int n = wg_n * N_TILE + (wave_n * NPW + NPW_A + ni) * 32;
            store_acc_f32(D, D_stride, acc_v[mi][ni], m, n);
        }
    }
}

// ---- dispatcher ----
// Mixed-acc tile FAMILY (each covers different non-pow2 N divisibility):
//   N512=2^9 (16acc) / N576=2^6*9 (18acc) / N640=2^7*5 (20acc).
enum Tile { T128, T256, T512, T576, T640 };
static int tile_nt(Tile t) {
    return t == T128 ? 128 : t == T256 ? 256 : t == T512 ? 512 : t == T576 ? 576 : 640;
}
// NPW (=NPW_A+NPW_V) per tile for the default 2x2-wave family. Drives scale_B grouping.
static int tile_npw(Tile t) {
    return t == T128 ? 2 : t == T256 ? 4 : t == T512 ? 8 : t == T576 ? 9 : 10;
}
// Wave shape: N512 can run as 2x8 (WAVES_M2,NPW8) or 4x4 (WAVES_M1,WAVES_N4,NPW4) — same
// 16 acc / occ1 / WG-tile 128x512, but 4x4's square per-wave tile loads 8 vs 10 tiles per
// 16 MFMA (perimeter 8 vs 10) => +4~6% on most shapes; 2x8's wider A-reuse wins only on the
// largest square (both dims >=8192). mpw/npw set the coalesced scale groups (= per-wave M/N).
// 4x4 was a win over plain 2x8, but depth-1 prefetch on 2x8 supersedes it on every shape
// (2x8+PF >= 4x4+PF >= 4x4); keep the 4x4 path available but default the dispatch to 2x8+PF.
static bool choose_square(Tile, int, int) { return false; }
// L2-aware swizzle (SWZ=16) on the N512 2x8 path: walking WGs down M inside a 16-wide
// n-band keeps B hot in L2. Measured win iff the n-band is fully populated, i.e.
// nb = N/512 >= 16 (N >= 8192): +0.8~4.4% across M (2048..8192). For nb < 16 the band is
// short and it's neutral-to-negative (-0.6~2.1%), so gate strictly on N. (M is irrelevant:
// at tiny M the remap degenerates to the identity mapping, so it never hurts.)
static int choose_swz(Tile t, int /*M*/, int N) {
    if (t != T512) return 0;
    return (N / 512 >= 16) ? 16 : 0;
}
static void shape_of(Tile t, bool square, int& mpw, int& npw) {
    if (t == T512 && square) { mpw = 4; npw = 4; }   // 4x4
    else                     { mpw = 2; npw = tile_npw(t); }
}
static const char* tile_nm(Tile t) {
    return t == T128   ? "N128"
           : t == T256 ? "N256"
           : t == T512 ? "N512"
           : t == T576 ? "N576[V2]"
                       : "N640[V2]";
}

// cost(tile) = ceil(WG/256) / (WG * eff): lower=faster. Accounts for grid-balance
// (ceil = wasted tail passes) and per-WG efficiency (bigger mixed-acc tile amortizes A).
// per-WG efficiency, MEASURED relative to N512 at full fill (8192-class):
//   N128≈0.55  N256≈0.77  N512=1.0  N576≈1.06  N640≈1.08
// (big mixed-acc tiles amortize A's uncoalesced load far better; small tiles
//  fill more CUs but each WG is much less efficient.)
static double tile_eff(Tile t) {
    return t == T640 ? 1.08 : t == T576 ? 1.06 : t == T512 ? 1.00 : t == T256 ? 0.77 : 0.55;
}
static double tile_cost(Tile t, int M, int N) {
    int nt = tile_nt(t);
    if (N % nt != 0) return 1e30;  // tile must divide N
    double wg = (double)(M / 128) * (N / nt);
    if (wg < 1) return 1e30;
    double passes = ceil(wg / 256.0);  // ceil = wasted tail passes (grid imbalance)
    return passes / (wg * tile_eff(t));
}
// UNIFIED dispatcher: one cost model over all 5 tiles covers the whole (M,N) space —
// small-matrix shrink (N256/N128 fill more CUs) AND large mixed-acc (N512/576/640
// amortize A) AND non-pow2 N divisibility, all from ceil(WG/256)/(WG*eff).
static Tile choose_tile(int M, int N) {
    Tile best = T512;
    double bc = 1e30;
    for (Tile t : {T128, T256, T512, T576, T640}) {
        double c = tile_cost(t, M, N);
        if (c < bc) {
            bc = c;
            best = t;
        }
    }
    return best;
}

// ---- B preshuffle for a given N_TILE-agnostic 32-col block layout (reuse preshuffle_B) ----
// preprocess provides preshuffle_B producing 1536B/32-col-tile; tile index uses 32-col blocks.

static void launch(Tile t, bool square, int swz, int M, int N, int K, const void* dA,
                   const void* dB, const uint8_t* dsA, const uint8_t* dsB, float* dD, int prb) {
    dim3 block(256);
    int kit = K / 64;
    switch (t) {
        case T128: {
            dim3 g(M / 128, N / 128);
            mxfp6_gemm_pipeline<128, 2, 0, 4, 4><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            break;
        }
        case T256: {
            dim3 g(M / 128, N / 256);
            mxfp6_gemm_pipeline<128, 4, 0, 4, 2><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            break;
        }
        case T512: {
            dim3 g(M / 128, N / 512);
            if (square)  // 4x4: WAVES_M=1, WAVES_N=4, NPW_A=4
                mxfp6_gemm_pipeline<128, 4, 0, 4, 1, 1, 4><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            else if (swz == 16)  // 2x8 + L2-aware swizzle (large square grids)
                mxfp6_gemm_pipeline<128, 8, 0, 4, 1, 2, 2, 16><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            else         // 2x8: default WAVES_M=2, WAVES_N=2, NPW_A=8
                mxfp6_gemm_pipeline<128, 8, 0, 4, 1><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            break;
        }
        case T576: {
            dim3 g(M / 128, N / 576);
            mxfp6_gemm_pipeline<128, 8, 1, 4, 1><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            break;
        }
        case T640: {
            dim3 g(M / 128, N / 640);
            mxfp6_gemm_pipeline<128, 8, 2, 4, 1><<<g, block>>>(dA, dB, dsA, dsB, dD, N, kit, prb);
            break;
        }
    }
}

// mpw/npw = per-wave M/N blocks of the launched config; drive scale_A/scale_B coalesce groups.
static void prep(int M, int N, int K, int mpw, int npw, std::vector<float>& Af,
                 std::vector<float>& Bf, void** dA, void** dB, uint8_t** dsA, uint8_t** dsB,
                 float** dD, int* prb) {
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    PreprocessedScale saP = preprocess_scale(Aq.scales.data(), M, K);
    PreprocessedScale sbP = preprocess_scale(Bq.scales.data(), N, K);
    CoalescedScale saC = preshuffle_scale(saP, mpw);
    CoalescedScale sbC = preshuffle_scale(sbP, npw);
    PreshuffledB pbB = preshuffle_B(Bq);
    hipMalloc(dA, Aq.packed_data.size());
    hipMalloc(dB, pbB.data.size());
    hipMalloc(dsA, saC.data.size());
    hipMalloc(dsB, sbC.data.size());
    hipMalloc(dD, (size_t)M * N * 4);
    hipMemcpy(*dA, Aq.packed_data.data(), Aq.packed_data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dB, pbB.data.data(), pbB.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dsA, saC.data.data(), saC.data.size(), hipMemcpyHostToDevice);
    hipMemcpy(*dsB, sbC.data.data(), sbC.data.size(), hipMemcpyHostToDevice);
    *prb = Aq.packed_row_bytes;
}

static bool correct(int M, int N, int K, Tile t, bool square, int swz = 0) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    QuantizedMatrix Aq = quantize_to_mxfp6(Af.data(), M, K);
    QuantizedMatrix Bq = preprocess_B(Bf.data(), K, N);
    std::vector<float> Dref((size_t)M * N);
    mxfp6_gemm_ref(Aq, Bq, Dref.data(), M, K, N);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    int prb, mpw, npw;
    shape_of(t, square, mpw, npw);
    prep(M, N, K, mpw, npw, Af, Bf, &dA, &dB, &dsA, &dsB, &dD, &prb);
    hipMemset(dD, 0, (size_t)M * N * 4);
    launch(t, square, swz, M, N, K, dA, dB, dsA, dsB, dD, prb);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("  err %s\n", hipGetErrorString(hipGetLastError()));
        return false;
    }
    std::vector<float> Dg((size_t)M * N);
    hipMemcpy(Dg.data(), dD, (size_t)M * N * 4, hipMemcpyDeviceToHost);
    float e = 0, mx = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        e = fmaxf(e, fabsf(Dg[i] - Dref[i]));
        mx = fmaxf(mx, fabsf(Dref[i]));
    }
    printf("  %s%s%s M=%d N=%d K=%d: err=%.3e %s\n", tile_nm(t), square ? "[4x4]" : "",
           swz ? "[swz]" : "", M, N, K, e, e < 1e-2f * fmaxf(1.f, mx) ? "PASS" : "FAIL");
    hipFree(dA);
    hipFree(dB);
    hipFree(dsA);
    hipFree(dsB);
    hipFree(dD);
    return e < 1e-2f * fmaxf(1.f, mx);
}

static double bench(int M, int N, int K, Tile t, bool square, int swz = 0) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
    for (auto& x : Af) x = d(rng);
    for (auto& x : Bf) x = d(rng);
    void *dA, *dB;
    uint8_t *dsA, *dsB;
    float* dD;
    int prb, mpw, npw;
    shape_of(t, square, mpw, npw);
    prep(M, N, K, mpw, npw, Af, Bf, &dA, &dB, &dsA, &dsB, &dD, &prb);
    for (int i = 0; i < 10; i++) launch(t, square, swz, M, N, K, dA, dB, dsA, dsB, dD, prb);
    hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        for (int i = 0; i < 20; i++) launch(t, square, swz, M, N, K, dA, dB, dsA, dsB, dD, prb);
        hipEventRecord(b);
        hipDeviceSynchronize();
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a);
        hipEventDestroy(b);
        best = fmin(best, ms / 20.0);
    }
    hipFree(dA);
    hipFree(dB);
    hipFree(dsA);
    hipFree(dsB);
    hipFree(dD);
    return 2.0 * M * N * K / (best * 1e-3) / 1e12;
}

int main() {
    printf("=== Correctness (incl V2 N640) ===\n");
    int ok = 0, tot = 0;
    struct C {
        int M, N, K;
        Tile t;
        bool sq;
        int swz;
    };
    C cs[] = {{256, 512, 256, T512, false, 0}, {256, 512, 256, T512, true, 0},  // 2x8 and 4x4
              {512, 8192, 256, T512, false, 16},  // swizzle WG-remap (nb=16 band)
              {256, 576, 256, T576, false, 0}, {256, 640, 256, T640, false, 0},
              {512, 1280, 512, T640, false, 0}, {256, 256, 256, T256, false, 0},
              {256, 128, 256, T128, false, 0}};
    for (auto& c : cs) {
        tot++;
        if (correct(c.M, c.N, c.K, c.t, c.sq, c.swz)) ok++;
    }
    printf("%d/%d\n", ok, tot);
    if (ok != tot) return 1;

    printf("\n=== Dispatch + benchmark (heuristic (tile,shape) vs all) ===\n");
    struct S {
        int M, N, K;
    };
    S sh[] = {
        {8192, 8192, 8192}, {4096, 4096, 8192}, {8192, 4096, 8192}, {4096, 8192, 8192},
        {2048, 8192, 8192}, {2048, 4096, 8192}, {2048, 2048, 8192}, {1024, 4096, 4096},
        {8192, 5120, 8192}, {4096, 5120, 8192}, {8192, 9216, 8192}, {8192, 7680, 8192},
    };
    int opt = 0, miss = 0;
    for (auto& s : sh) {
        Tile pick = choose_tile(s.M, s.N);
        bool psq = choose_square(pick, s.M, s.N);
        int psw = choose_swz(pick, s.M, s.N);
        double best = 0;
        Tile bt = pick; bool bsq = psq; int bsw = psw;
        char line[384] = {0};
        int off = 0;
        for (Tile t : {T128, T256, T512, T576, T640}) {
            if (s.N % tile_nt(t)) continue;
            // candidate (square, swz) variants: only N512 has 4x4 and the swizzle path.
            for (auto v : {std::pair<bool, int>{false, 0}, {true, 0}, {false, 16}}) {
                bool sq = v.first; int sw = v.second;
                if ((sq || sw) && t != T512) continue;  // 4x4 / swz only defined for N512
                double tf = bench(s.M, s.N, s.K, t, sq, sw);
                off += snprintf(line + off, sizeof(line) - off, " %s%s%s=%.0f", tile_nm(t),
                                sq ? "[4x4]" : "", sw ? "[swz]" : "", tf);
                if (tf > best) { best = tf; bt = t; bsq = sq; bsw = sw; }
            }
        }
        bool ok = (pick == bt && psq == bsq && psw == bsw);
        opt += ok;
        miss += !ok;
        printf("M=%-4d N=%-4d K=%-5d pick=%s%s%s |%s | %s\n", s.M, s.N, s.K, tile_nm(pick),
               psq ? "[4x4]" : "", psw ? "[swz]" : "", line, ok ? "OPT" : "MISS");
        if (!ok) printf("        ^^ MISS: pick %s%s%s but best %s%s%s=%.0f\n", tile_nm(pick),
                        psq ? "[4x4]" : "", psw ? "[swz]" : "", tile_nm(bt), bsq ? "[4x4]" : "",
                        bsw ? "[swz]" : "", best);
    }
    printf("\n%d/%d OPT, %d MISS\n", opt, opt + miss, miss);
    return 0;
}

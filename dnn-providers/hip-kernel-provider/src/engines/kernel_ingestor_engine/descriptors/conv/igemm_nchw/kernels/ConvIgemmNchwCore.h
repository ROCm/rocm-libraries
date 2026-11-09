// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K7: channels-first backward data as an implicit GEMM with the col2im fused, reading dy, the
// K C S filter and writing dx in place (no staging copies, no workspace).
// HIPRTC-compatible: no standard headers.
//
// The convolution runs along the width; the depth and height positions are rows the
// convolution maps one to one. With x + pad = stride * q + phase,
//   dx[c][x] = sum over k, j of w[k][c][phase + stride * j] * dy[k][q - j].
// MIOpen's GEMM path computes col[c][s][q] = sum_k w[k][c][s] dy[k][q] (one GEMM: rows (c, s),
// contiguous in the K C S filter) and then scatters col into dx (col2im). Here a block owns BN
// channels, BMQ consecutive q and all stride phases, so every product lands in its own
// accumulator: phase p takes tap p + stride * j as the A rows shifted by j. Per k-step the block
// stages the contiguous filter run w[k][c0 .. c0 + BN][0 .. S) and the dy row window
// dy[k][q0 - (TAPS - 1) .. q0 + BMQ), so both operands are read straight from the graph's
// tensors, coalesced, and nothing is written twice.
//
// The matrix-core kernel reuses K1's atoms (one 16x16 or 32x32 step per arch family): A rows are
// q, B rows are channels, the reduction is k. Its LDS tiles are row-major in k (As[q][k],
// Bs[tap][c][k]), filled by transposing stores: each thread loads the same VEC positions of KV
// consecutive k rows and stores them as KV-element words. The FMA kernel keeps k-major tiles
// (As[k][q], Bs[k][c][tap]) and gives each thread TQ phase indices of TC channels.
#pragma once

#include "ConvIgemmCore.h"
#include "ConvIgemmNchwArgs.h"

/// VEC consecutive elements of one row: positions pos .. pos + VEC - 1 at element offset off
/// of the buffer, those outside [0, limit) (or all, when !rowOk) as zero. A whole-vector load
/// when the row holds all of them; else one element at a time, since the buffer's range check
/// is per dword and a straddling 16-bit load would drop elements that are in range.
template <class T, int VEC>
__device__ inline typename K1Raw<VEC * sizeof(T)>::type
    k7LoadRun(__amdgpu_buffer_rsrc_t rsrc, int off, int pos, int limit, bool rowOk)
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    typedef typename K1Raw<sizeof(T)>::type One;
    if(rowOk && pos >= 0 && pos + VEC <= limit)
    {
        return k1BufferLoad<Raw>(rsrc, static_cast<unsigned>(off) * sizeof(T));
    }
    Raw v;
    One* e = reinterpret_cast<One*>(&v);
#pragma unroll
    for(int i = 0; i < VEC; ++i)
    {
        const bool ok = rowOk && pos + i >= 0 && pos + i < limit;
        e[i] = k1BufferLoad<One>(rsrc,
                                 ok ? static_cast<unsigned>(off + i) * sizeof(T) : K1_OUT_OF_RANGE);
    }
    return v;
}

/// The block's global loads and their LDS stores. A: the dy window, WQ positions of BK
/// k rows from q0w. B: the filter run of BN channels by S taps (run = BN * S positions from
/// bStart = c0 * S) of BK k rows. A unit is KV consecutive k rows by VEC positions; positions
/// vary fastest across lanes, so a wave's loads read whole runs of each row.
///
/// The matrix-core A tile is row-major in k: position r's row starts at r * LDK plus a 16-byte
/// skew per VEC rows (aRow), so the transposing stores of a wave, whose positions sit VEC rows
/// apart, spread over the banks.
template <class T, int VEC, int KV, int BK, int WQ, int BN, int SMAX, int THREADS>
struct K7Tiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int KP = BK / KV;
    static constexpr int NVA = WQ / VEC; // A vectors per k row
    static constexpr int A_UNITS = KP * NVA;
    static constexpr int A_PASSES = (A_UNITS + THREADS - 1) / THREADS;
    static constexpr int B_UNITS = KP * (BN * SMAX / VEC); // at the largest filter
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static constexpr int SKEW = 16 / static_cast<int>(sizeof(T));
    static_assert(BK % KV == 0 && WQ % VEC == 0 && (BN * SMAX) % VEC == 0, "K7 tile loads");

    __amdgpu_buffer_rsrc_t aRsrc, bRsrc;
    int tid, rows, row, wy, k, cs, bStart, q0w, taps;
    // B units: k pair (KP when the pass is idle), position in the run, its channel and tap.
    int bK[B_PASSES], bPos[B_PASSES], bC[B_PASSES], bS[B_PASSES];
    Raw ra[A_PASSES * KV], rb[B_PASSES * KV];

    /// Element offset of A tile position @p r's row (matrix-core layout).
    template <int LDK>
    __host__ __device__ static constexpr int aRow(int r)
    {
        return r * LDK + (r / VEC) * SKEW;
    }
    __device__ static int aK(int p, int t)
    {
        return (t + p * THREADS) / NVA;
    }
    __device__ static int aV(int p, int t)
    {
        return (t + p * THREADS) % NVA;
    }
    __device__ static bool aActive(int p, int t)
    {
        if constexpr(A_UNITS % THREADS == 0)
        {
            return true;
        }
        return t + p * THREADS < A_UNITS;
    }
    __device__ bool bActive(int p) const
    {
        return bK[p] < KP;
    }

    __device__ void init(const K7Args& a, int img, int rw, int q0, int c0, int t, int halo)
    {
        tid = t;
        rows = a.rows;
        row = rw;
        wy = a.wy;
        k = a.k;
        taps = a.fs;
        cs = a.c * a.fs;
        bStart = c0 * a.fs;
        q0w = q0 - halo;
        const T* dy = static_cast<const T*>(a.dy) + static_cast<long long>(img) * a.dyImage;
        aRsrc = k1Rsrc(dy, static_cast<long long>(a.n - img) * a.dyImage * sizeof(T));
        bRsrc = k1Rsrc(a.w, static_cast<long long>(a.k) * cs * sizeof(T));
        // The run's vectors per k row at this filter (BN * S is a multiple of 16).
        const int nvb = BN * a.fs / VEC;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            bK[p] = u / nvb;
            bPos[p] = (u - bK[p] * nvb) * VEC;
            bC[p] = bPos[p] / taps;
            bS[p] = bPos[p] - bC[p] * taps;
        }
    }

    /// Registers for the k rows from k0.
    __device__ void load(int k0)
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            if(!aActive(p, tid))
                continue;
            const int q = q0w + aV(p, tid) * VEC;
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const int kk = k0 + aK(p, tid) * KV + j;
                const bool ok = kk < k;
                const int off = ok ? (kk * rows + row) * wy + q : 0;
                ra[p * KV + j] = k7LoadRun<T, VEC>(aRsrc, off, q, wy, ok);
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bActive(p))
                continue;
            const int pos = bStart + bPos[p];
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const int kk = k0 + bK[p] * KV + j;
                const bool ok = kk < k;
                rb[p * KV + j] = k7LoadRun<T, VEC>(bRsrc, ok ? kk * cs + pos : 0, pos, cs, ok);
            }
        }
    }

    /// One word of KV elements: element e of the unit's KV registers from @p r.
    __device__ static void storeWord(T* d, const Raw* r, int e)
    {
        if constexpr(KV == 2)
        {
            const unsigned short* lo = reinterpret_cast<const unsigned short*>(&r[0]);
            const unsigned short* hi = reinterpret_cast<const unsigned short*>(&r[1]);
            *reinterpret_cast<unsigned*>(d)
                = static_cast<unsigned>(lo[e]) | (static_cast<unsigned>(hi[e]) << 16);
        }
        else
        {
            *d = reinterpret_cast<const T*>(&r[0])[e];
        }
    }

    /// Matrix-core tiles, row-major in k: As[q][k] (rows at aRow), Bs[tap][c][k] (LDK per row).
    template <int LDK>
    __device__ void storeRows(T* As, T* Bs) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            if(!aActive(p, tid))
                continue;
            T* d = As + aRow<LDK>(aV(p, tid) * VEC) + aK(p, tid) * KV;
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                storeWord(d + e * LDK, &ra[p * KV], e);
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bActive(p))
                continue;
            int c = bC[p], s = bS[p];
            T* d = Bs + bK[p] * KV;
#pragma unroll
            for(int e = 0; e < VEC; ++e)
            {
                storeWord(d + (s * BN + c) * LDK, &rb[p * KV], e);
                if(++s == taps)
                {
                    s = 0;
                    ++c;
                }
            }
        }
    }

    /// FMA tiles, k-major (KV 1): As[k][q] (LDA per k), Bs[k][c][tap] (CP per channel, LDB per k).
    template <int LDA, int LDB, int CP>
    __device__ void storeColumns(T* As, T* Bs) const
    {
        static_assert(KV == 1, "k-major tiles take one k row per unit");
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            if(!aActive(p, tid))
                continue;
            *reinterpret_cast<Raw*>(As + aK(p, tid) * LDA + aV(p, tid) * VEC) = ra[p];
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bActive(p))
                continue;
            int c = bC[p], s = bS[p];
            T* d = Bs + bK[p] * LDB;
            const T* v = reinterpret_cast<const T*>(&rb[p]);
#pragma unroll
            for(int e = 0; e < VEC; ++e)
            {
                d[c * CP + s] = v[e];
                if(++s == taps)
                {
                    s = 0;
                    ++c;
                }
            }
        }
    }
};

/// The block's coordinates: image and row (grid z), first channel c0, and its dx positions
/// x0 .. x0 + BMQ * ST - 1 (x0 a multiple of the tile, so no tile is mostly empty). With
/// pad = ST * pq + r, x0 + pad = ST * q0 + r: phases from r take q0 + i at tile index i, phases
/// below r take q0 + 1 + i, and both land at x0 + i * ST + ((phase - r) mod ST).
struct K7Block
{
    int img, row, c0, x0, q0, r;
};

template <int ST>
__device__ inline K7Block k7Block(const K7Args& a, int bmq, int bn)
{
    K7Block b;
    b.img = static_cast<int>(blockIdx.z) / a.rows;
    b.row = static_cast<int>(blockIdx.z) - b.img * a.rows;
    b.c0 = static_cast<int>(blockIdx.y) * bn;
    b.x0 = static_cast<int>(blockIdx.x) * bmq * ST;
    b.q0 = static_cast<int>(blockIdx.x) * bmq + a.pad / ST;
    b.r = a.pad % ST;
    return b;
}

/// Stores RUN consecutive dx elements of channel c from @p src (dx positions x .. x + RUN - 1),
/// those outside [0, wx) and channels past the last skipped: whole vectors where the row holds
/// all VEC of them.
template <class T, int VEC>
__device__ inline void k7StoreRun(const K7Args& a, const K7Block& b, int c, int x, const T* src)
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    if(c >= a.c)
        return;
    T* dst = static_cast<T*>(a.dx) + static_cast<long long>(b.img) * a.dxImage
             + (static_cast<long long>(c) * a.rows + b.row) * a.wx + x;
    if(x >= 0 && x + VEC <= a.wx)
    {
        *reinterpret_cast<Raw*>(dst) = *reinterpret_cast<const Raw*>(src);
        return;
    }
#pragma unroll
    for(int e = 0; e < VEC; ++e)
        if(x + e >= 0 && x + e < a.wx)
            dst[e] = src[e];
}

// ---------------------------------------------------------------------------------------------
// Matrix-core kernel. Block: WM x WN x WP waves (phase indices x channels x phase groups); tile
// BMQ phase indices (all ST phases) x BN channels; k-step BK. Each wave holds FQ x FN fragments
// for each of its PW = ST / WP phases: a fragment of A (one shift) serves the wave's phases and
// channel fragments, one of B (one tap) its phase indices. TAPS: the most taps a phase has (the
// launcher checks ceil(S / ST) <= TAPS).

template <class T,
          class Atom,
          int BMQ,
          int BN,
          int BK,
          int WM,
          int WN,
          int WP,
          int VEC,
          int ST,
          int TAPS>
__device__ void k7DgradMma(const K7Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WP * WAVE;
    constexpr int DIM = Atom::DIM;
    constexpr int WTQ = BMQ / WM, WTN = BN / WN;
    constexpr int FQ = WTQ / DIM, FN = WTN / DIM;
    constexpr int PW = ST / WP;
    static_assert(ST % WP == 0, "K7 phase groups");
    constexpr int KV = sizeof(T) == 2 ? 2 : 1;
    constexpr int LDK = BK + 16 / static_cast<int>(sizeof(T));
    constexpr int HALO = TAPS - 1;
    // The window: q0 - HALO .. q0 + BMQ (the phases below r reach one further), whole vectors.
    constexpr int WQ = BMQ + VEC;
    constexpr int SMAX = ST * TAPS;
    typedef K7Tiles<T, VEC, KV, BK, WQ, BN, SMAX, THREADS> Tiles;
    constexpr int A_LDS = Tiles::template aRow<LDK>(WQ);
    constexpr int B_LDS = SMAX * BN * LDK;
    constexpr int OLD = BMQ * ST + 16 / static_cast<int>(sizeof(T));
    constexpr int OUT_LDS = BN * OLD;
    constexpr int LDS = A_LDS + B_LDS > OUT_LDS ? A_LDS + B_LDS : OUT_LDS;
    static_assert(WTQ % DIM == 0 && WTN % DIM == 0 && BK % Atom::KA == 0, "K7 wave tile");
    static_assert(TAPS <= VEC && (BMQ * ST) % VEC == 0, "K7 window and epilogue");

    __shared__ __attribute__((aligned(16))) T lds[LDS];
    T* As = lds;
    T* Bs = lds + A_LDS;

    const K7Block b = k7Block<ST>(a, BMQ, BN);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wq = wave % WM, wn = (wave / WM) % WN, wp = wave / (WM * WN);
    const int taps = a.fs;
    // The wave's phases are p0 .. p0 + PW - 1.
    const int p0 = wp * PW;

    Tiles tiles;
    tiles.init(a, b.img, b.row, b.q0, b.c0, tid, HALO);

    typename Atom::acc acc[PW][FQ][FN];
#pragma unroll
    for(int p = 0; p < PW; ++p)
#pragma unroll
        for(int i = 0; i < FQ; ++i)
#pragma unroll
            for(int n = 0; n < FN; ++n)
                acc[p][i][n] = typename Atom::acc{};

    const int fragRow = lane % DIM;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    // The lane's A row at tile index i, shift d: aRow0 + i DIM + d.
    const int aRow0 = wq * WTQ + fragRow + HALO;
    const T* bBase = Bs + (wn * WTN + fragRow) * LDK + kb;

    const int iters = (a.k + BK - 1) / BK;
    tiles.load(0);
    for(int it = 0; it < iters; ++it)
    {
        __syncthreads();
        tiles.template storeRows<LDK>(As, Bs);
        __syncthreads();
        if(it + 1 < iters)
        {
            tiles.load((it + 1) * BK);
        }
        // Phase p at tile index i reads dy[q0 + i + below(p) - j] for its tap p + ST j: the A
        // fragments shifted by d = below(p) - j serve every (p, j) pair with that shift.
#pragma unroll
        for(int kk = 0; kk < BK; kk += Atom::KA)
        {
#pragma unroll
            for(int d = 1 - TAPS; d <= 1; ++d)
            {
                // Whether any of the wave's phases has a tap at this shift (wave-uniform).
                bool used = false;
#pragma unroll
                for(int p = 0; p < PW; ++p)
                {
                    const int j = (p0 + p < b.r ? 1 : 0) - d;
                    used = used || (j >= 0 && j < TAPS && p0 + p + j * ST < taps);
                }
                if(!used)
                {
                    continue;
                }
                typename Atom::frag af[FQ];
#pragma unroll
                for(int i = 0; i < FQ; ++i)
                    af[i] = k1Frag<Atom>(As + Tiles::template aRow<LDK>(aRow0 + i * DIM + d) + kk
                                         + kb);
#pragma unroll
                for(int p = 0; p < PW; ++p)
                {
                    const int j = (p0 + p < b.r ? 1 : 0) - d;
                    const int s = p0 + p + j * ST;
                    if(j >= 0 && j < TAPS && s < taps)
                    {
                        typename Atom::frag bf[FN];
#pragma unroll
                        for(int n = 0; n < FN; ++n)
                            bf[n] = k1Frag<Atom>(bBase + (s * BN + n * DIM) * LDK + kk);
#pragma unroll
                        for(int i = 0; i < FQ; ++i)
#pragma unroll
                            for(int n = 0; n < FN; ++n)
                                acc[p][i][n] = Atom::mma(af[i], bf[n], acc[p][i][n]);
                    }
                }
            }
        }
    }

    // Epilogue: the tile into LDS as Os[c][x - x0], then each channel's contiguous dx run out in
    // vectors.
    __syncthreads();
    T* Os = lds;
#pragma unroll
    for(int p = 0; p < PW; ++p)
    {
        // Tile position of phase p0 + p at index i: (i + below) * ST + p0 + p - r.
        const int px = p0 + p < b.r ? ST + p0 + p - b.r : p0 + p - b.r;
#pragma unroll
        for(int i = 0; i < FQ; ++i)
#pragma unroll
            for(int n = 0; n < FN; ++n)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const int q = wq * WTQ + i * DIM + Atom::dRow(lane, e);
                    const int c = wn * WTN + n * DIM + lane % DIM;
                    Os[c * OLD + q * ST + px] = K1Elem<T>::store(acc[p][i][n][e]);
                }
    }
    __syncthreads();
    constexpr int XV = BMQ * ST / VEC;
    for(int v = tid; v < BN * XV; v += THREADS)
    {
        const int c = v / XV;
        const int xv = (v - c * XV) * VEC;
        k7StoreRun<T, VEC>(a, b, b.c0 + c, b.x0 + xv, Os + c * OLD + xv);
    }
}

// ---------------------------------------------------------------------------------------------
// FMA kernel: every arch and dtype (fp32 on RDNA, everything on gfx906). Threads TX (channels)
// x TY (tile indices); each computes TQ consecutive tile indices, all ST phases, of TC channels
// TX apart. Tile BMQ = TY TQ by BN = TX TC; k-step BK.

/// The FMA filter tile's pitch per channel, in elements: the taps rounded up to an odd number of
/// 16-byte chunks, so the lanes' per-channel tap vectors (channels one pitch apart) spread over
/// the banks.
__host__ __device__ constexpr int k7FmaPitch(int smax, int pade)
{
    const int chunks = (smax + pade - 1) / pade;
    return (chunks % 2 == 0 ? chunks + 1 : chunks) * pade;
}

/// The FMA main loop for r = pad mod ST = R: phase p's tap p + ST j reads shift below(p) - j,
/// below(p) = p < R, so every register index is compile-time. The launcher checks the filter
/// has exactly TAPS taps in its fullest phase, so only the last tap of a phase can be past the
/// filter's end: FULL (S = ST TAPS) has no such tap and no branch, so the FMAs of different
/// phases interleave (and pair into dual-issue instructions on gfx11 / gfx12); otherwise one
/// uniform branch per phase skips it.
///
/// STAGES 1: one LDS tile, stored between two barriers per k-step. STAGES 2: two tiles in turn;
/// the next k-step's tile is stored after this one's FMAs, one barrier per k-step.
template <class T,
          int BMQ,
          int BN,
          int BK,
          int TQ,
          int TC,
          int VEC,
          int ST,
          int TAPS,
          int STAGES,
          int R,
          bool FULL,
          class Tiles>
// Inlined at all 2 ST call sites: out of line, acc (passed by reference) would live in scratch.
__device__ __attribute__((always_inline)) inline void
    k7FmaSteps(const K7Args& a, Tiles& tiles, T* lds, float (&acc)[TC][TQ][ST])
{
    constexpr int TX = BN / TC;
    constexpr int HALO = TAPS - 1;
    constexpr int WQ = BMQ + VEC;
    constexpr int SMAX = ST * TAPS;
    constexpr int PADE = 16 / static_cast<int>(sizeof(T));
    constexpr int LDA = WQ + PADE;
    constexpr int CP = k7FmaPitch(SMAX, PADE);
    constexpr int LDB = BN * CP + PADE;
    constexpr int STAGE = BK * (LDA + LDB);
    const int tx = static_cast<int>(threadIdx.x) % TX, ty = static_cast<int>(threadIdx.x) / TX;
    // Phases whose last tap is inside the filter: p + ST (TAPS - 1) < S.
    const int lastPhases = a.fs - ST * HALO;

    // The FMAs of one k-step on the tiles at As (BK x LDA) and Bs (BK x LDB).
    auto multiply = [&](const T* As, const T* Bs) {
#pragma unroll 2
        for(int kk = 0; kk < BK; ++kk)
        {
            // Window values q0 + ty TQ - HALO .. q0 + ty TQ + TQ: index t is shift t - HALO.
            float av[TQ + TAPS];
#pragma unroll
            for(int t = 0; t < TQ + TAPS; ++t)
                av[t] = K1Elem<T>::load(As[kk * LDA + ty * TQ + t]);
#pragma unroll
            for(int c = 0; c < TC; ++c)
            {
                float bv[SMAX];
#pragma unroll
                for(int s = 0; s < SMAX; ++s)
                    bv[s] = K1Elem<T>::load(Bs[kk * LDB + (tx + c * TX) * CP + s]);
#pragma unroll
                for(int p = 0; p < ST; ++p)
                {
                    const int below = p < R ? 1 : 0;
#pragma unroll
                    for(int j = 0; j < HALO; ++j)
#pragma unroll
                        for(int i = 0; i < TQ; ++i)
                            acc[c][i][p] = __builtin_fmaf(
                                av[i + below - j + HALO], bv[p + j * ST], acc[c][i][p]);
                    if(FULL || p < lastPhases)
                    {
#pragma unroll
                        for(int i = 0; i < TQ; ++i)
                            acc[c][i][p]
                                = __builtin_fmaf(av[i + below], bv[p + HALO * ST], acc[c][i][p]);
                    }
                }
            }
        }
    };

    const int iters = (a.k + BK - 1) / BK;
    tiles.load(0);
    if constexpr(STAGES == 2)
    {
        tiles.template storeColumns<LDA, LDB, CP>(lds, lds + BK * LDA);
        __syncthreads();
        for(int it = 0; it < iters; ++it)
        {
            if(it + 1 < iters)
            {
                tiles.load((it + 1) * BK);
            }
            const T* cur = lds + (it & 1) * STAGE;
            multiply(cur, cur + BK * LDA);
            if(it + 1 < iters)
            {
                // The other tile was last read before the previous barrier.
                T* next = lds + ((it + 1) & 1) * STAGE;
                tiles.template storeColumns<LDA, LDB, CP>(next, next + BK * LDA);
            }
            __syncthreads();
        }
    }
    else
    {
        for(int it = 0; it < iters; ++it)
        {
            __syncthreads();
            tiles.template storeColumns<LDA, LDB, CP>(lds, lds + BK * LDA);
            __syncthreads();
            if(it + 1 < iters)
            {
                tiles.load((it + 1) * BK);
            }
            multiply(lds, lds + BK * LDA);
        }
    }
}

template <class T, int BMQ, int BN, int BK, int TQ, int TC, int VEC, int ST, int TAPS, int STAGES>
__device__ void k7DgradFma(const K7Args& a)
{
    constexpr int TX = BN / TC, TY = BMQ / TQ;
    constexpr int THREADS = TX * TY;
    constexpr int HALO = TAPS - 1;
    constexpr int WQ = BMQ + VEC;
    constexpr int SMAX = ST * TAPS;
    constexpr int PADE = 16 / static_cast<int>(sizeof(T));
    constexpr int LDA = WQ + PADE;
    constexpr int CP = k7FmaPitch(SMAX, PADE);
    constexpr int LDB = BN * CP + PADE;
    static_assert(BN % TC == 0 && BMQ % TQ == 0 && TAPS <= VEC, "K7 FMA tile");
    static_assert(STAGES == 1 || STAGES == 2, "K7 FMA stages");

    __shared__ __attribute__((aligned(16))) T lds[STAGES * BK * (LDA + LDB)];

    const K7Block b = k7Block<ST>(a, BMQ, BN);
    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;

    K7Tiles<T, VEC, 1, BK, WQ, BN, SMAX, THREADS> tiles;
    tiles.init(a, b.img, b.row, b.q0, b.c0, tid, HALO);

    float acc[TC][TQ][ST];
#pragma unroll
    for(int c = 0; c < TC; ++c)
#pragma unroll
        for(int i = 0; i < TQ; ++i)
#pragma unroll
            for(int p = 0; p < ST; ++p)
                acc[c][i][p] = 0.0f;

    // r = pad mod ST and whether the filter fills its last tap row are the same for every block:
    // one main loop per case, so the phases' shifts are compile-time.
    const bool full = a.fs == ST * TAPS;
    switch(b.r)
    {
#define K7_FMA_R(R)                                                                  \
    case R:                                                                          \
        if constexpr(R < ST)                                                         \
        {                                                                            \
            if(full)                                                                 \
                k7FmaSteps<T, BMQ, BN, BK, TQ, TC, VEC, ST, TAPS, STAGES, R, true>(  \
                    a, tiles, lds, acc);                                             \
            else                                                                     \
                k7FmaSteps<T, BMQ, BN, BK, TQ, TC, VEC, ST, TAPS, STAGES, R, false>( \
                    a, tiles, lds, acc);                                             \
        }                                                                            \
        break;
        K7_FMA_R(0)
        K7_FMA_R(1)
        K7_FMA_R(2)
        K7_FMA_R(3)
        K7_FMA_R(4)
        K7_FMA_R(5)
        K7_FMA_R(6)
        K7_FMA_R(7)
#undef K7_FMA_R
    default:
        break;
    }

    // Phase p at tile index i is dx position x0 + i ST + ((p - r) mod ST).
#pragma unroll
    for(int c = 0; c < TC; ++c)
    {
        const int ch = b.c0 + tx + c * TX;
        if(ch >= a.c)
            continue;
        T* dst = static_cast<T*>(a.dx) + static_cast<long long>(b.img) * a.dxImage
                 + (static_cast<long long>(ch) * a.rows + b.row) * a.wx;
#pragma unroll
        for(int p = 0; p < ST; ++p)
        {
            const int px = p < b.r ? ST + p - b.r : p - b.r;
#pragma unroll
            for(int i = 0; i < TQ; ++i)
            {
                const int x = b.x0 + (ty * TQ + i) * ST + px;
                if(x < a.wx)
                    dst[x] = K1Elem<T>::store(acc[c][i][p]);
            }
        }
    }
}

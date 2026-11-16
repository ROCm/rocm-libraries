// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The patch-tile kernel of the implicit-GEMM engines (K1 configs with patch 2): a patch
// convolution (filter equal to the stride, no padding) as one dedicated GEMM per launch, over the
// problem withPatchView forms (K1Args as the host reshapes it).
//
//   forward        y[m][n] = sum_k x_patch[m][k] w[n][k]    k = ((t R + r) S + s) C + c
//   backward data  dx_patch[m][n] = sum_k dy[m][k] w[k][n]  n = ((t R + r) S + s) C + c
//
// Forward rows are a patch's T R filter rows, each one contiguous run of S C channels of x
// (K1Args c is S C after the width fold), so a thread's 16-byte granule walks the runs with a
// cursor and no division. Backward data reads dy rows whole, and the filter either as staged
// [(t, r, s, c)][k] (contiguous along k) or, with WD, as the graph's K T R S C (K-major): on
// gfx950 that tile stays K-major in LDS and the B fragments come from ds_read_b64_tr_b16;
// elsewhere the loader stores it transposed. Two LDS stages, one barrier a k-step, the next step's global loads
// in flight under the current step's matrix-core work; XOR-swizzled 16-byte chunks; no split.
// Every element of a ragged k-step, row or column is loaded as zero on both operands.
#pragma once

#include "ConvIgemmCore.h"

template <class T, class Atom, int BM, int BN, int BK, int WM, int WN, int VEC, int DIR, int WD>
__device__ void k1PatchMma(const K1Args& a CONV_EPI_FN_PARAM)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WAVE;
    constexpr int DIM = Atom::DIM;
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int FM = WTM / DIM, FN = WTN / DIM;
    constexpr bool SWZ = VEC * sizeof(T) == 16;
    constexpr int LD = SWZ ? BK : BK + 16 / static_cast<int>(sizeof(T));
#if defined(__gfx950__)
    constexpr bool TR = DIR == K1_BACKWARD_DATA && WD != 0 && Atom::KPL == 8 && sizeof(T) == 2;
#else
    constexpr bool TR = false;
#endif
    // Backward data on gfx950: B stays K-major, [BK][LDT].
    constexpr int LDT = BN + 16 / static_cast<int>(sizeof(T));
    constexpr int A_TILE = BM * LD;
    constexpr int B_TILE = TR ? BK * LDT : BN * LD;
    constexpr int STAGE = A_TILE + B_TILE;
    constexpr int OLD = WTN + 16 / static_cast<int>(sizeof(T));
    constexpr int OUT = WM * WN * DIM * OLD;
    constexpr int LDS = 2 * STAGE > OUT ? 2 * STAGE : OUT;
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    typedef K1Swz<(SWZ ? BK : 16)> Swz;
    static_assert(WTM % DIM == 0 && WTN % DIM == 0 && BK % Atom::KA == 0, "tile");
    static_assert(!SWZ || BK == 16 || BK == 32 || BK == 64, "swizzled rows");
    static_assert(sizeof(T) == 2, "16-bit operands");

    // Rows (A) and B units a pass covers.
    constexpr int ROW_THREADS = BK / VEC;
    constexpr int RPP = THREADS / ROW_THREADS;
    constexpr int A_PASSES = (BM + RPP - 1) / RPP;
    // Backward data with WD reads the graph's K T R S C filter, K-major; otherwise the filter is
    // the staged [(t, r, s, c)][k] copy, read like forward's, contiguous along k.
    constexpr bool BKMAJOR = DIR == K1_BACKWARD_DATA && WD != 0;
    // K-major B: units of one k row x VEC columns; transposed stores pair two k rows.
    constexpr int KV = BKMAJOR && !TR ? 2 : 1;
    constexpr int BCOL = BN / VEC;
    constexpr int B_UNITS = BKMAJOR ? (BK / KV) * BCOL : BN * ROW_THREADS;
    constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static_assert(THREADS % ROW_THREADS == 0, "row loads");

    __shared__ __attribute__((aligned(16))) T lds[LDS];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wm = wave % WM, wn = wave / WM;
    const bool fwd = DIR == K1_FORWARD;
    // GEMM extents: rows, columns, reduction.
    const long long rows = a.m;
    const int cols = fwd ? a.k : a.c;
    const int red = fwd ? a.c * a.ft * a.fr : a.k;
    const long long m0 = (static_cast<long long>(blockIdx.x) + a.tileBase) * BM;
    const int n0 = blockIdx.y * BN;
    if(m0 >= rows)
        return;
#if defined(CONV_EPI) && CONV_EPI
    // The fused epilogue (forward only: backward data takes none, and its columns are filter
    // elements rather than channels): each output column's per-channel operands, loaded before
    // the reduction so their latency hides behind it.
    static_assert(DIR == K1_FORWARD, "the patch tile fuses forward epilogues only");
    ConvEpiGen::Chan chans[FN];
#pragma unroll
    for(int j = 0; j < FN; ++j)
    {
        const int en = n0 + wn * WTN + j * DIM + lane % DIM;
        chans[j] = epi.chan(en, en < cols);
    }
#endif

    const T* src = static_cast<const T*>(fwd ? a.x : a.y);
    const T* w = static_cast<const T*>(a.w);
    const long long srcElems
        = fwd ? static_cast<long long>(a.n) * a.di * a.hi * a.wi * a.c : rows * a.k;
    const __amdgpu_buffer_rsrc_t aRsrc = k1Rsrc(src, srcElems * 2);
    const __amdgpu_buffer_rsrc_t bRsrc
        = k1Rsrc(w, static_cast<long long>(a.k) * a.c * a.ft * a.fr * a.fs * 2);

    // A: this thread's rows and its granule kv of each k-step.
    const int kv = (tid % ROW_THREADS) * VEC;
    const int arow = tid / ROW_THREADS;
    unsigned aOff[A_PASSES];
    const long long spatial = static_cast<long long>(a.dout) * a.ho * a.wo;
#pragma unroll
    for(int p = 0; p < A_PASSES; ++p)
    {
        const long long m = m0 + arow + p * RPP;
        if(m >= rows || arow + p * RPP >= BM)
        {
            aOff[p] = K1_OUT_OF_RANGE;
            continue;
        }
        long long base;
        if(fwd)
        {
            const long long img = m / spatial;
            long long rest = m - img * spatial;
            const int ow = static_cast<int>(rest % a.wo);
            rest /= a.wo;
            const int oh = static_cast<int>(rest % a.ho);
            const int od = static_cast<int>(rest / a.ho);
            base = (((img * a.di + static_cast<long long>(od) * a.sd) * a.hi
                     + static_cast<long long>(oh) * a.sh)
                        * a.wi
                    + ow)
                   * a.c;
        }
        else
        {
            base = m * a.k;
        }
        aOff[p] = static_cast<unsigned>(base * 2);
    }
    // Forward: the granule's place in the patch, as (filter row q, channel c within its run):
    // element offset tapOff + c, rows stepped by a.wi * a.c and depth taps by a.hi * a.wi * a.c.
    const long long rowPitch = static_cast<long long>(a.wi) * a.c;
    const long long depthPitch = static_cast<long long>(a.hi) * rowPitch;
    int curC = kv, curR = 0, curT = 0;
    long long tapOff = 0;
    if(fwd)
    {
        while(curC >= a.c)
        {
            curC -= a.c;
            if(++curR == a.fr)
            {
                curR = 0;
                ++curT;
                tapOff += depthPitch - (a.fr - 1) * rowPitch;
            }
            else
            {
                tapOff += rowPitch;
            }
        }
    }

    // B: rows n (forward, contiguous along k) or units (k row, VEC columns) (backward data).
    unsigned bOff[B_PASSES];
    int bRow[B_PASSES], bCol[B_PASSES];
#pragma unroll
    for(int p = 0; p < B_PASSES; ++p)
    {
        const int u = tid + p * THREADS;
        if constexpr(BKMAJOR)
        {
            bRow[p] = (u / BCOL) * KV; // k within the step
            bCol[p] = (u % BCOL) * VEC;
            const bool ok = u < B_UNITS && n0 + bCol[p] < cols;
            bOff[p] = ok ? static_cast<unsigned>((n0 + bCol[p]) * 2) : K1_OUT_OF_RANGE;
        }
        else
        {
            bRow[p] = u / ROW_THREADS; // n within the tile
            bCol[p] = kv;
            const bool ok = u < B_UNITS && n0 + bRow[p] < cols;
            bOff[p] = ok ? static_cast<unsigned>(static_cast<long long>(n0 + bRow[p]) * red * 2)
                         : K1_OUT_OF_RANGE;
        }
    }

    Raw ra[A_PASSES], rb[B_PASSES * KV];
    const int steps = (red + BK - 1) / BK;
    auto load = [&](int it) {
        const int k0 = it * BK;
        const bool aK = k0 + kv < red;
        long long aDelta;
        if(fwd)
        {
            aDelta = tapOff + curC;
            // The cursor one k-step on.
            curC += BK;
            while(curC >= a.c)
            {
                curC -= a.c;
                if(++curR == a.fr)
                {
                    curR = 0;
                    ++curT;
                    tapOff += depthPitch - (a.fr - 1) * rowPitch;
                }
                else
                {
                    tapOff += rowPitch;
                }
            }
        }
        else
        {
            aDelta = k0 + kv;
        }
        const unsigned aStep = static_cast<unsigned>(aDelta * 2);
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            ra[p] = k1BufferLoad<Raw>(aRsrc, aK ? aOff[p] + aStep : K1_OUT_OF_RANGE);
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const int k = BKMAJOR ? k0 + bRow[p] + j : k0 + bCol[p];
                const bool ok = k < red && bOff[p] != K1_OUT_OF_RANGE;
                const unsigned off
                    = BKMAJOR
                          ? bOff[p] + static_cast<unsigned>(static_cast<long long>(k) * cols * 2)
                          : bOff[p] + static_cast<unsigned>(k * 2);
                rb[p * KV + j] = k1BufferLoad<Raw>(bRsrc, ok ? off : K1_OUT_OF_RANGE);
            }
        }
    };
    // Element offset of (row, k) in a [rows][BK] tile.
    auto at = [](int r, int k) {
        if constexpr(SWZ)
            return Swz::at(r, Swz::key(r), k & ~7) + (k & 7);
        else
            return r * LD + k;
    };
    auto store = [&](T* As, T* Bs) {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            if(arow + p * RPP < BM)
                *reinterpret_cast<Raw*>(As + at(arow + p * RPP, kv)) = ra[p];
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(tid + p * THREADS >= B_UNITS)
                continue;
            if constexpr(!BKMAJOR)
            {
                *reinterpret_cast<Raw*>(Bs + at(bRow[p], kv)) = rb[p];
            }
            else if constexpr(TR)
            {
                *reinterpret_cast<Raw*>(Bs + bRow[p] * LDT + bCol[p]) = rb[p];
            }
            else
            {
                // Transposed: column n's pair of k rows as one 4-byte store.
                const unsigned short* lo = reinterpret_cast<const unsigned short*>(&rb[p * 2]);
                const unsigned short* hi = reinterpret_cast<const unsigned short*>(&rb[p * 2 + 1]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    *reinterpret_cast<unsigned*>(Bs + at(bCol[p] + e, bRow[p]))
                        = static_cast<unsigned>(lo[e]) | (static_cast<unsigned>(hi[e]) << 16);
            }
        }
    };

    typename Atom::acc acc[FM][FN];
#pragma unroll
    for(int i = 0; i < FM; ++i)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[i][j] = typename Atom::acc{};
    const int fragRow = lane % DIM;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    typedef typename K1Raw<(Atom::KPL * 2 < 16 ? Atom::KPL * 2 : 16)>::type Piece;
    constexpr int PIECES = Atom::KPL * 2 > 16 ? Atom::KPL * 2 / 16 : 1;
    auto frag = [&](const T* tile, int r, int k) {
        Piece v[PIECES];
#pragma unroll
        for(int c = 0; c < PIECES; ++c)
            v[c] = *reinterpret_cast<const Piece*>(tile + at(r, k + c * 8));
        return __builtin_bit_cast(typename Atom::frag, v);
    };
    auto multiply = [&](const T* As, const T* Bs) {
#pragma unroll
        for(int kk = 0; kk < BK; kk += Atom::KA)
        {
            typename Atom::frag af[FM], bf[FN];
#pragma unroll
            for(int i = 0; i < FM; ++i)
                af[i] = frag(As, wm * WTM + i * DIM + fragRow, kk + kb);
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                if constexpr(TR)
                {
#if defined(__gfx950__)
                    // Column n = fragRow of the slab, k rows kk + kb .. + 7: in a 16-lane group,
                    // lane s reads 4 columns 4 (s % 4) .. at k row s / 4 (and 4 rows on).
                    typedef short s4 __attribute__((ext_vector_type(4)));
                    typedef __attribute__((address_space(3))) s4 lds_s4;
                    const int s16 = lane % 16;
                    const T* p
                        = Bs + (kk + kb + s16 / 4) * LDT + wn * WTN + j * DIM + 4 * (s16 % 4);
                    const s4 lo = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p));
                    const s4 hi = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p + 4 * LDT));
                    bf[j] = __builtin_bit_cast(
                        typename Atom::frag,
                        __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
#endif
                }
                else
                {
                    bf[j] = frag(Bs, wn * WTN + j * DIM + fragRow, kk + kb);
                }
            }
#pragma unroll
            for(int i = 0; i < FM; ++i)
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    acc[i][j] = Atom::mma(af[i], bf[j], acc[i][j]);
        }
    };

    // Two stages: step it multiplies stage it % 2 while step it + 1's loads are in flight, then
    // stores them into the other stage; one barrier a step.
    if(steps > 0)
    {
        load(0);
        store(lds, lds + A_TILE);
    }
    for(int it = 0; it < steps; ++it)
    {
        k1LdsBarrier();
        if(it + 1 < steps)
            load(it + 1);
        T* cur = lds + (it & 1) * STAGE;
        multiply(cur, cur + A_TILE);
        if(it + 1 < steps)
        {
            T* next = lds + ((it + 1) & 1) * STAGE;
            store(next, next + A_TILE);
        }
    }

    // Epilogue: per DIM-row slab, the accumulators to the wave's LDS strip, then runs of VEC
    // columns to global (forward y rows; backward data the patch scatter, runs within S C).
    k1LdsBarrier();
    T* out = static_cast<T*>(fwd ? a.y : a.x);
    T* strip = lds + wave * DIM * OLD;
    constexpr int ROW_VECS = WTN / VEC;
    K1Gather gather{};
    gather.out[0] = a.dout;
    gather.out[1] = a.ho;
    gather.out[2] = a.wo;
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
#pragma unroll
        for(int j = 0; j < FN; ++j)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
                strip[Atom::dRow(lane, e) * OLD + j * DIM + lane % DIM]
#if defined(CONV_EPI) && CONV_EPI
                    = K1Elem<T>::store(epi(acc[i][j][e], chans[j], 0, true));
#else
                    = K1Elem<T>::store(acc[i][j][e]);
#endif
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        for(int v = lane; v < DIM * ROW_VECS; v += WAVE)
        {
            const int r = v / ROW_VECS;
            const int col = (v - r * ROW_VECS) * VEC;
            const long long m = m0 + wm * WTM + i * DIM + r;
            const int nn = n0 + wn * WTN + col;
            if(m >= rows || nn >= cols)
                continue;
            const long long o
                = fwd ? m * a.k + nn : k1OutIndex<K1_BACKWARD_DATA>(a, gather, 0, 0, m, nn);
            if(nn + VEC <= cols)
                *reinterpret_cast<Raw*>(out + o)
                    = *reinterpret_cast<const Raw*>(strip + r * OLD + col);
            else
                for(int e = 0; e < VEC && nn + e < cols; ++e)
                    out[o + e] = strip[r * OLD + col + e];
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
}

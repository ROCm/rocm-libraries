// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K7: channels-first backward data as an implicit GEMM with the col2im fused, reading dy, the
// K C S filter and writing dx in place (no staging copies, no workspace).
// HIPRTC-compatible: no standard headers.
//
// The convolution runs along the width. Along depth and height a dx row reduces over every
// (dy row, filter row) pair that reaches it, one after another in the k loop, and a dx row with
// none is zero (K7Args, K7Block). With x + pad = stride * q + phase,
//   dx[c][x] = sum over pairs, k, j of w[k][c][td][th][phase + stride * j] * dy[k][row'][q - j].
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

/// The block's coordinates: image and dx row (grid z), first channel c0, and its dx positions
/// x0 .. x0 + BMQ * ST - 1 (x0 a multiple of the tile, so no tile is mostly empty). With
/// pad = ST * pq + r, x0 + pad = ST * q0 + r: phases from r take q0 + i at tile index i, phases
/// below r take q0 + 1 + i, and both land at x0 + i * ST + ((phase - r) mod ST).
///
/// Depth and height: with d + pd = sd qd0 + td0, the dx row takes dy row qd0 - jd through filter
/// row td0 + sd jd for each jd in [jdLo, jdLo + nD) (filter row inside the filter, dy row inside
/// dy), and likewise along the height: nD nH (dy row, filter row) pairs, each reduced over k in
/// turn. Pair 0's dy row is rowY and its filter row starts at element toff of each (k, c) filter.
/// A row with no pair (live false) is zero.
struct K7Block
{
    int img, row, c0, x0, q0, r, rowY, toff;
    int qd0, td0, jdLo, nD, qh0, th0, jhLo, nH;
    bool live;
};

/// The j range of one axis: filter rows t0 + s j < f, dy rows q0 - step j in [0, extent) (step:
/// the dilation, at stride 1; 1 otherwise).
__device__ inline void
    k7AxisTaps(int q0, int t0, int s, int step, int f, int extent, int& lo, int& count)
{
    const int hiFilter = t0 < f ? (f - t0 + s - 1) / s : 0; // j < hiFilter
    const int hiDy = q0 / step + 1; // q0 - step j >= 0
    const int over = q0 - extent + 1; // q0 - step j < extent: j >= over / step, rounded up
    lo = over > 0 ? (over + step - 1) / step : 0;
    const int hi = hiFilter < hiDy ? hiFilter : hiDy;
    count = hi > lo ? hi - lo : 0;
}

/// Pair pi's dy row and filter-row offset (pi below nD nH).
__device__ inline void k7PairOf(const K7Args& a, const K7Block& b, int pi, int& rowY, int& toff)
{
    const int i = pi / b.nH;
    const int jd = b.jdLo + i;
    const int jh = b.jhLo + pi - i * b.nH;
    rowY = (b.qd0 - jd * a.dd) * a.hy + b.qh0 - jh * a.dh;
    toff = ((b.td0 + a.sd * jd) * a.fh + b.th0 + a.sh * jh) * a.fs;
}

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
    // d + pd = sd qd0 + td0, h + ph = sh qh0 + th0.
    const int d = b.row / a.hx;
    const int h = b.row - d * a.hx;
    const int nd = d + a.pd, nh = h + a.ph;
    b.qd0 = nd / a.sd;
    b.qh0 = nh / a.sh;
    b.td0 = nd - b.qd0 * a.sd;
    b.th0 = nh - b.qh0 * a.sh;
    k7AxisTaps(b.qd0, b.td0, a.sd, a.dd, a.fd, a.dyDepth, b.jdLo, b.nD);
    k7AxisTaps(b.qh0, b.th0, a.sh, a.dh, a.fh, a.hy, b.jhLo, b.nH);
    b.live = b.nD * b.nH > 0;
    b.rowY = 0;
    b.toff = 0;
    if(b.live)
        k7PairOf(a, b, 0, b.rowY, b.toff);
    return b;
}

/// FLAT: the dy plane flattened. When no axis pads and dx is exactly the strides times dy on
/// every axis (Dx = sd Dy, Hx = sh Hy, Wx = ST Wy) and the width has one tap per phase, each dy
/// position (qd, qh, q) owns the dx block (sd qd + td, sh qh + th, ST q + p) for every class
/// (td, th) and phase p, so per class backward data is one GEMM over the dy positions of every
/// image (P = rowsY Wy per image, contiguous per k), N P in all. Grid z is the classes (sd sh); a
/// class past the filter (td >= T or th >= R) is zero. A block owns BMQ consecutive positions g
/// from q0 over all images (image g / P), and its outputs f = ST g + p from x0 = ST q0.
template <int ST>
__device__ inline K7Block k7BlockFlat(const K7Args& a, int bmq, int bn)
{
    K7Block b;
    b.img = 0;
    const int cls = static_cast<int>(blockIdx.z);
    const int td = cls / a.sh, th = cls - td * a.sh;
    // row carries the class: dx row offsets of class (td, th) are td hx + th from the block's.
    b.row = td * a.hx + th;
    b.c0 = static_cast<int>(blockIdx.y) * bn;
    b.q0 = static_cast<int>(blockIdx.x) * bmq;
    b.x0 = b.q0 * ST;
    b.r = 0;
    b.live = td < a.fd && th < a.fh;
    b.rowY = 0;
    b.toff = (td * a.fh + th) * a.fs;
    b.nD = b.nH = 1; // one pair, fixed
    b.qd0 = b.td0 = b.jdLo = b.qh0 = b.th0 = b.jhLo = 0;
    return b;
}

/// The dx element offset of flattened output f (over all images) of channel 0 (FLAT); channel c
/// adds c rows Wx.
__device__ inline long long k7FlatPosition(const K7Args& a, const K7Block& b, int f)
{
    const int plane = a.rowsY * a.wx; // outputs per image and channel: ST P
    const int img = f / plane;
    f -= img * plane;
    const int qrow = f / a.wx;
    const int x = f - qrow * a.wx;
    const int qd = qrow / a.hy;
    const int qh = qrow - qd * a.hy;
    const int dxRow = qd * a.sd * a.hx + qh * a.sh + b.row;
    return img * a.dxImage + static_cast<long long>(dxRow) * a.wx + x;
}

/// The dx element offset of flattened output f (over all images) of channel c (FLAT).
__device__ inline long long k7FlatOffset(const K7Args& a, const K7Block& b, int c, int f, int st)
{
    return k7FlatPosition(a, b, f) + static_cast<long long>(c) * a.rows * a.wx;
}

/// VEC consecutive elements of a filter run whose channels lie fStride elements apart in memory
/// (T R > 1): the run position's channel c (valid below cEnd) and tap s, at element offset phys
/// of the buffer. Loaded SUB elements at a time, SUB dividing S, so no load straddles channels.
template <class T, int VEC, int SUB>
__device__ inline typename K1Raw<VEC * sizeof(T)>::type k7LoadStrided(
    __amdgpu_buffer_rsrc_t rsrc, int phys, int c, int s, int cEnd, int fs, int fStride, bool ok)
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    typedef typename K1Raw<SUB * sizeof(T)>::type Sub;
    Raw v;
    Sub* part = reinterpret_cast<Sub*>(&v);
#pragma unroll
    for(int i = 0; i < VEC / SUB; ++i)
    {
        part[i] = k1BufferLoad<Sub>(
            rsrc, ok && c < cEnd ? static_cast<unsigned>(phys) * sizeof(T) : K1_OUT_OF_RANGE);
        s += SUB;
        phys += SUB;
        if(s >= fs)
        {
            s -= fs;
            ++c;
            phys += fStride - fs;
        }
    }
    return v;
}

/// A compile-time flag for generic lambdas (HIPRTC has no standard headers).
template <bool B>
struct K7Flag
{
    static constexpr bool value = B;
};

/// Byte offset @p bytes when @p ok, else one the buffer's range check zeroes: arithmetic, not a
/// select the compiler may lower to a branch per load (with its own vmcnt(0)).
__device__ inline unsigned k7Offset(unsigned bytes, bool ok)
{
    return bytes | (static_cast<unsigned>(!ok) << 31);
}

/// The block's global loads and their LDS stores. A: the dy window, WQ positions of BK
/// k rows from q0w. B: the filter run of BN channels by S taps (run = BN * S positions from
/// bStart = c0 * S) of BK k rows. A unit is KV consecutive k rows by VEC positions; positions
/// vary fastest across lanes, so a wave's loads read whole runs of each row.
///
/// The matrix-core A tile is row-major in k: position r's row starts at r * LDK plus a 16-byte
/// skew per VEC rows (aRow), so the transposing stores of a wave, whose positions sit VEC rows
/// apart, spread over the banks.
template <class T,
          int VEC,
          int KV,
          int BK,
          int WQ,
          int BN,
          int SMAX,
          int THREADS,
          bool FLAT = false>
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
    // The filter: elements per k row and per (k, c), the filter row's offset, the first channel
    // and, for T R > 1, the largest power of two (at most VEC) dividing S.
    int kStride, fStride, toff, c0, cEnd, sub;
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

    __device__ void init(const K7Args& a, const K7Block& b, int t, int halo)
    {
        tid = t;
        rows = a.rowsY;
        row = b.rowY;
        wy = a.wy;
        k = a.k;
        taps = a.fs;
        cs = a.c * a.fs;
        bStart = b.c0 * a.fs;
        q0w = b.q0 - halo;
        fStride = a.fStride;
        kStride = a.c * a.fStride;
        toff = b.toff;
        c0 = b.c0;
        cEnd = a.c;
        sub = VEC;
        while(sub > 1 && a.fs % sub != 0)
            sub /= 2;
        const T* dy = static_cast<const T*>(a.dy) + static_cast<long long>(b.img) * a.dyImage;
        aRsrc = k1Rsrc(dy, static_cast<long long>(a.n - b.img) * a.dyImage * sizeof(T));
        bRsrc = k1Rsrc(a.w, static_cast<long long>(a.k) * kStride * sizeof(T));
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

    /// The k loop over every (dy row, filter row) pair: iteration it loads pair it / kIters,
    /// k rows from (it mod kIters) BK.
    int kIters, kIt, pi;
    template <bool FAST = false>
    __device__ int start(const K7Args& a)
    {
        kIters = (a.k + BK - 1) / BK;
        kIt = 0;
        pi = 0;
        load<FAST>(0);
        return kIters;
    }
    /// Loads the iteration after the last loaded one.
    template <bool FAST = false>
    __device__ void loadNext(const K7Args& a, const K7Block& b)
    {
        if(++kIt == kIters)
        {
            kIt = 0;
            ++pi;
            k7PairOf(a, b, pi, row, toff);
        }
        load<FAST>(kIt * BK);
    }

    /// Whether the block's loads take the fast form: its dy window inside the row (or flat) and
    /// its filter run contiguous and inside the filter (every channel of the tile real), so every
    /// load is a whole vector, only the k tail and idle units read zeros (branch-free offsets).
    __device__ bool fastLoads(const K7Args& a) const
    {
        const bool aIn = FLAT || (q0w >= 0 && q0w + WQ <= wy);
        return aIn && fStride == taps && c0 + BN <= cEnd;
    }

    /// FLAT: dy read as P = rowsY Wy positions per k row and image, positions g over all
    /// images. The launcher checks VEC divides P, so a unit's VEC positions lie in one image
    /// (g / P) and the tensor's end falls between units; a unit past it reads zeros.
    int aOff[A_PASSES];
    __device__ void flatten(const K7Args& a)
    {
        const int plane = a.rowsY * a.wy;
        rows = 1;
        row = 0;
        wy = plane;
        const int total = a.n * plane;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int g = q0w + aV(p, tid) * VEC;
            const int img = g / plane;
            aOff[p] = g < total ? static_cast<int>(img * a.dyImage) + g - img * plane : -1;
        }
    }

    /// FLAT: the unit of pass p at k row kk (valid when ok).
    __device__ Raw loadFlat(int p, int kk, bool ok) const
    {
        return k1BufferLoad<Raw>(aRsrc,
                                 ok && aOff[p] >= 0
                                     ? static_cast<unsigned>(aOff[p] + kk * wy) * sizeof(T)
                                     : K1_OUT_OF_RANGE);
    }

    /// The unit of pass p's k row kk from a filter whose channels are not adjacent (T R > 1).
    __device__ Raw loadStrided(int p, int kk, bool ok) const
    {
        const int c = c0 + bC[p];
        const int phys = kk * kStride + c * fStride + toff + bS[p];
        switch(sub)
        {
        case 8:
            if constexpr(VEC >= 8)
                return k7LoadStrided<T, VEC, (VEC >= 8 ? 8 : 1)>(
                    bRsrc, phys, c, bS[p], cEnd, taps, fStride, ok);
            break;
        case 4:
            if constexpr(VEC >= 4)
                return k7LoadStrided<T, VEC, (VEC >= 4 ? 4 : 1)>(
                    bRsrc, phys, c, bS[p], cEnd, taps, fStride, ok);
            break;
        case 2:
            if constexpr(VEC >= 2)
                return k7LoadStrided<T, VEC, (VEC >= 2 ? 2 : 1)>(
                    bRsrc, phys, c, bS[p], cEnd, taps, fStride, ok);
            break;
        default:
            break;
        }
        return k7LoadStrided<T, VEC, 1>(bRsrc, phys, c, bS[p], cEnd, taps, fStride, ok);
    }

    /// Registers for the k rows from k0.
    template <bool FAST = false>
    __device__ void load(int k0)
    {
        if constexpr(FAST)
        {
#pragma unroll
            for(int p = 0; p < A_PASSES; ++p)
            {
                const bool active = aActive(p, tid);
                const int q = q0w + aV(p, tid) * VEC;
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const int kk = k0 + aK(p, tid) * KV + j;
                    const bool ok = active && kk < k;
                    unsigned bytes;
                    if constexpr(FLAT)
                        bytes = k7Offset(static_cast<unsigned>(aOff[p] + kk * wy) * sizeof(T),
                                         ok && aOff[p] >= 0);
                    else
                        bytes = k7Offset(
                            static_cast<unsigned>((kk * rows + row) * wy + q) * sizeof(T), ok);
                    ra[p * KV + j] = k1BufferLoad<Raw>(aRsrc, bytes);
                }
            }
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
                const bool active = bActive(p);
                const int pos = bStart + bPos[p];
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const int kk = k0 + bK[p] * KV + j;
                    rb[p * KV + j] = k1BufferLoad<Raw>(
                        bRsrc,
                        k7Offset(static_cast<unsigned>(kk * cs + pos) * sizeof(T),
                                 active && kk < k));
                }
            }
            return;
        }
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
                if constexpr(FLAT)
                {
                    ra[p * KV + j] = loadFlat(p, ok ? kk : 0, ok);
                }
                else
                {
                    const int off = ok ? (kk * rows + row) * wy + q : 0;
                    ra[p * KV + j] = k7LoadRun<T, VEC>(aRsrc, off, q, wy, ok);
                }
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
                rb[p * KV + j] = fStride == taps
                                     ? k7LoadRun<T, VEC>(bRsrc, ok ? kk * cs + pos : 0, pos, cs, ok)
                                     : loadStrided(p, ok ? kk : 0, ok);
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

/// A dx row no dy row reaches: the block's BN channels by BMQ * ST positions, zeros.
/// Stores VEC flattened outputs f .. f + VEC - 1 of channel c (FLAT): those past the plane
/// (f >= ST * dy positions) and channels past the last skipped. A dx row holds Wx = ST Wy outputs,
/// so a vector whose start is a multiple of VEC stays in one row when VEC divides Wx.
template <class T, int VEC, int ST>
__device__ inline void k7StoreFlat(const K7Args& a, const K7Block& b, int c, int f, const T* src)
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    if(c >= a.c)
        return;
    const int end = a.n * a.rowsY * a.wx;
    T* dst = static_cast<T*>(a.dx);
    if(a.wx % VEC == 0 && f + VEC <= end)
    {
        *reinterpret_cast<Raw*>(dst + k7FlatOffset(a, b, c, f, ST))
            = *reinterpret_cast<const Raw*>(src);
        return;
    }
#pragma unroll
    for(int e = 0; e < VEC; ++e)
        if(f + e < end)
            dst[k7FlatOffset(a, b, c, f + e, ST)] = src[e];
}

template <class T, int VEC, int ST, int FLAT>
__device__ inline void k7Store(const K7Args& a, const K7Block& b, int c, int x, const T* src)
{
    if constexpr(FLAT > 0)
        k7StoreFlat<T, VEC, ST>(a, b, c, x, src);
    else
        k7StoreRun<T, VEC>(a, b, c, x, src);
}

template <class T, int VEC, int BN, int SPAN, int THREADS, int ST, int FLAT>
__device__ inline void k7ZeroTile(const K7Args& a, const K7Block& b)
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    Raw zero;
    T* z = reinterpret_cast<T*>(&zero);
#pragma unroll
    for(int e = 0; e < VEC; ++e)
        z[e] = T{};
    constexpr int XV = SPAN / VEC;
    for(int v = static_cast<int>(threadIdx.x); v < BN * XV; v += THREADS)
    {
        const int c = v / XV;
        k7Store<T, VEC, ST, FLAT>(a, b, b.c0 + c, b.x0 + (v - c * XV) * VEC, z);
    }
}

/// The FMA filter tile's pitch per channel, in elements: the taps rounded up to an odd number of
/// 16-byte chunks, so the lanes' per-channel tap vectors (channels one pitch apart) spread over
/// the banks.
__host__ __device__ constexpr int k7FmaPitch(int smax, int pade)
{
    const int chunks = (smax + pade - 1) / pade;
    return (chunks % 2 == 0 ? chunks + 1 : chunks) * pade;
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
          int TAPS,
          int FLAT = 0,
          int HALOK = VEC,
          class TO = T>
__device__ void k7DgradMma(const K7Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WP * WAVE;
    constexpr int DIM = Atom::DIM;
    constexpr int WTQ = BMQ / WM, WTN = BN / WN;
    constexpr int FQ = WTQ / DIM, FN = WTN / DIM;
    // Phases computed: all ST, or (FLAT) the first FLAT, the rest zero.
    constexpr int NP = FLAT > 0 ? FLAT : ST;
    constexpr int PW = NP / WP;
    static_assert(NP % WP == 0 && NP <= ST, "K7 computed phases");
    static_assert(FLAT > 0 || ST % WP == 0, "K7 phase groups");
    constexpr int KV = sizeof(T) == 2 ? 2 : 1;
    constexpr int LDK = BK + 16 / static_cast<int>(sizeof(T));
    // The window: q0 - halo .. q0 + BMQ (the phases below r reach one further), whole vectors,
    // halo = (TAPS - 1) dil: HALOK past the tile (the launcher checks it covers the reach).
    constexpr int WQ = BMQ + HALOK;
    static_assert(HALOK % VEC == 0, "K7 window in whole vectors");
    constexpr int SMAX = ST * TAPS;
    typedef K7Tiles<T, VEC, KV, BK, WQ, BN, SMAX, THREADS, (FLAT > 0)> Tiles;
    // One k per lane (the fp32 atoms): the operands need no k-contiguous runs, so the tiles are
    // k-major as the FMA kernel's (As[k][q], Bs[k][c][tap]), stored as whole vectors, untransposed.
    constexpr bool KMAJOR = Atom::KPL == 1;
    constexpr int PADE = 16 / static_cast<int>(sizeof(T));
    constexpr int LDA = WQ + PADE;
    constexpr int CP = k7FmaPitch(SMAX, PADE);
    constexpr int LDB = BN * CP + PADE;
    constexpr int A_LDS = KMAJOR ? BK * LDA : Tiles::template aRow<LDK>(WQ);
    constexpr int B_LDS = KMAJOR ? BK * LDB : SMAX * BN * LDK;
    // The output tile in dx's element type TO (fp32 for three-bf16, else T), staged over the
    // operand tiles: VECO of them a 16-byte vector.
    constexpr int OLD = BMQ * ST + 16 / static_cast<int>(sizeof(TO));
    constexpr int OUT_LDS
        = (BN * OLD * static_cast<int>(sizeof(TO)) + static_cast<int>(sizeof(T)) - 1)
          / static_cast<int>(sizeof(T));
    constexpr int LDS = A_LDS + B_LDS > OUT_LDS ? A_LDS + B_LDS : OUT_LDS;
    constexpr int VECO = VEC * static_cast<int>(sizeof(T)) / static_cast<int>(sizeof(TO));
    static_assert(WTQ % DIM == 0 && WTN % DIM == 0 && BK % Atom::KA == 0, "K7 wave tile");
    static_assert((BMQ * ST) % VEC == 0, "K7 epilogue");

    __shared__ __attribute__((aligned(16))) T lds[LDS];
    T* As = lds;
    T* Bs = lds + A_LDS;

    static_assert(FLAT == 0 || TAPS == 1, "K7 flat: one tap per phase");
    const K7Block b = FLAT > 0 ? k7BlockFlat<ST>(a, BMQ, BN) : k7Block<ST>(a, BMQ, BN);
    if(!b.live)
    {
        k7ZeroTile<TO, VECO, BN, BMQ * ST, THREADS, ST, FLAT>(a, b);
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wq = wave % WM, wn = (wave / WM) % WN, wp = wave / (WM * WN);
    const int taps = a.fs;
    // The wave's phases are p0 .. p0 + PW - 1.
    const int p0 = wp * PW;

    // Tap shifts: the dilation at stride 1, unit otherwise.
    const int dil = ST == 1 ? a.dil : 1;
    const int halo = (TAPS - 1) * dil;
    Tiles tiles;
    tiles.init(a, b, tid, halo);
    if constexpr(FLAT > 0)
        tiles.flatten(a);

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
    // The lane's A row at tile index i, shift d: aRow0 + i DIM + d dil.
    const int aRow0 = wq * WTQ + fragRow + halo;
    const T* bBase
        = KMAJOR ? Bs + kb * LDB + (wn * WTN + fragRow) * CP : Bs + (wn * WTN + fragRow) * LDK + kb;

    // The k loop, in two forms: whole-vector loads with branch-free offsets for a block whose
    // window and filter run lie inside the tensors (the rule), edge-checked loads otherwise.
    auto mainLoop = [&](auto fastFlag) {
        constexpr bool FAST = decltype(fastFlag)::value;
        const int iters = tiles.template start<FAST>(a) * b.nD * b.nH;
        for(int it = 0; it < iters; ++it)
        {
            __syncthreads();
            if constexpr(KMAJOR)
                tiles.template storeColumns<LDA, LDB, CP>(As, Bs);
            else
                tiles.template storeRows<LDK>(As, Bs);
            __syncthreads();
            if(it + 1 < iters)
            {
                tiles.template loadNext<FAST>(a, b);
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
                    {
                        if constexpr(KMAJOR)
                            af[i] = As[(kk + kb) * LDA + aRow0 + i * DIM + d * dil];
                        else
                            af[i] = k1Frag<Atom>(
                                As + Tiles::template aRow<LDK>(aRow0 + i * DIM + d * dil) + kk
                                + kb);
                    }
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
                            {
                                if constexpr(KMAJOR)
                                    bf[n] = bBase[kk * LDB + n * DIM * CP + s];
                                else
                                    bf[n] = k1Frag<Atom>(bBase + (s * BN + n * DIM) * LDK + kk);
                            }
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
    };
    if(tiles.fastLoads(a))
        mainLoop(K7Flag<true>{});
    else
        mainLoop(K7Flag<false>{});

    // Epilogue: the tile into LDS as Os[c][x - x0], then each channel's contiguous dx run out in
    // vectors.
    __syncthreads();
    TO* Os = reinterpret_cast<TO*>(lds);
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
                    Os[c * OLD + q * ST + px] = K1Elem<TO>::store(acc[p][i][n][e]);
                }
    }
    if constexpr(NP < ST)
    {
        // Phases past the computed ones (FLAT, r = 0): zero.
        for(int v = tid; v < BN * BMQ * (ST - NP); v += THREADS)
        {
            const int c = v / (BMQ * (ST - NP));
            const int rest = v - c * (BMQ * (ST - NP));
            const int q = rest / (ST - NP);
            Os[c * OLD + q * ST + NP + rest - q * (ST - NP)] = TO{};
        }
    }
    __syncthreads();
    constexpr int XV = BMQ * ST / VECO;
    for(int v = tid; v < BN * XV; v += THREADS)
    {
        const int c = v / XV;
        const int xv = (v - c * XV) * VECO;
        k7Store<TO, VECO, ST, FLAT>(a, b, b.c0 + c, b.x0 + xv, Os + c * OLD + xv);
    }
}

// ---------------------------------------------------------------------------------------------
// FMA kernel: every arch and dtype (fp32 on RDNA, everything on gfx906). Threads TX (channels)
// x TY (tile indices); each computes TQ consecutive tile indices, all ST phases, of TC channels
// TX apart. Tile BMQ = TY TQ by BN = TX TC; k-step BK.

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
          int NP,
          int HALOK,
          class Tiles>
// Inlined at all 2 ST call sites: out of line, acc (passed by reference) would live in scratch.
__device__ __attribute__((always_inline)) inline void
    k7FmaSteps(const K7Args& a, const K7Block& b, Tiles& tiles, T* lds, float (&acc)[TC][TQ][NP])
{
    constexpr int TX = BN / TC;
    constexpr int HALO = TAPS - 1;
    constexpr int WQ = BMQ + HALOK;
    constexpr int SMAX = ST * TAPS;
    constexpr int PADE = 16 / static_cast<int>(sizeof(T));
    constexpr int LDA = WQ + PADE;
    constexpr int CP = k7FmaPitch(SMAX, PADE);
    constexpr int LDB = BN * CP + PADE;
    constexpr int STAGE = BK * (LDA + LDB);
    const int tx = static_cast<int>(threadIdx.x) % TX, ty = static_cast<int>(threadIdx.x) / TX;
    // Phases whose last tap is inside the filter: p + ST (TAPS - 1) < S.
    const int lastPhases = a.fs - ST * HALO;
    // HALOK past the vector (stride 1): each tap reads its own shifted window values, (TAPS - 1 -
    // j) dil into the window, so any filter width and dilation fit.
    const int dil = a.dil;

    // The FMAs of one k-step on the tiles at As (BK x LDA) and Bs (BK x LDB).
    auto multiply = [&](const T* As, const T* Bs) {
        if constexpr(HALOK > VEC)
        {
            static_assert(ST == 1 && NP == 1, "K7 FMA wide halo: stride 1");
#pragma unroll 2
            for(int kk = 0; kk < BK; ++kk)
            {
#pragma unroll
                for(int j = 0; j < TAPS; ++j)
                {
                    const T* row = As + kk * LDA + ty * TQ + (TAPS - 1 - j) * dil;
                    float av[TQ];
#pragma unroll
                    for(int i = 0; i < TQ; ++i)
                        av[i] = K1Elem<T>::load(row[i]);
#pragma unroll
                    for(int c = 0; c < TC; ++c)
                    {
                        const float bv = K1Elem<T>::load(Bs[kk * LDB + (tx + c * TX) * CP + j]);
#pragma unroll
                        for(int i = 0; i < TQ; ++i)
                            acc[c][i][0] = __builtin_fmaf(av[i], bv, acc[c][i][0]);
                    }
                }
            }
            return;
        }
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
                for(int p = 0; p < NP; ++p)
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

    // Two forms of the k loop, as the matrix-core kernel's (fast: whole-vector, branch-free loads).
    auto mainLoop = [&](auto fastFlag) {
        constexpr bool FAST = decltype(fastFlag)::value;
        const int iters = tiles.template start<FAST>(a) * b.nD * b.nH;
        if constexpr(STAGES == 2)
        {
            tiles.template storeColumns<LDA, LDB, CP>(lds, lds + BK * LDA);
            __syncthreads();
            for(int it = 0; it < iters; ++it)
            {
                if(it + 1 < iters)
                {
                    tiles.template loadNext<FAST>(a, b);
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
                    tiles.template loadNext<FAST>(a, b);
                }
                multiply(lds, lds + BK * LDA);
            }
        }
    };
    if(tiles.fastLoads(a))
        mainLoop(K7Flag<true>{});
    else
        mainLoop(K7Flag<false>{});
}

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
          int FLAT = 0,
          int HALOK = VEC>
__device__ void k7DgradFma(const K7Args& a)
{
    constexpr int TX = BN / TC, TY = BMQ / TQ;
    constexpr int THREADS = TX * TY;
    // The window: one vector past the tile (taps within the vector, unit dilation), or HALOK.
    constexpr int WQ = BMQ + HALOK;
    constexpr int SMAX = ST * TAPS;
    constexpr int PADE = 16 / static_cast<int>(sizeof(T));
    constexpr int LDA = WQ + PADE;
    constexpr int CP = k7FmaPitch(SMAX, PADE);
    constexpr int LDB = BN * CP + PADE;
    static_assert(BN % TC == 0 && BMQ % TQ == 0 && HALOK % VEC == 0, "K7 FMA tile");
    static_assert(HALOK > VEC || TAPS <= VEC, "K7 FMA: a register window within the vector");
    static_assert(STAGES == 1 || STAGES == 2, "K7 FMA stages");

    __shared__ __attribute__((aligned(16))) T lds[STAGES * BK * (LDA + LDB)];

    static_assert(FLAT == 0 || TAPS == 1, "K7 flat: one tap per phase");
    const K7Block b = FLAT > 0 ? k7BlockFlat<ST>(a, BMQ, BN) : k7Block<ST>(a, BMQ, BN);
    if(!b.live)
    {
        k7ZeroTile<T, VEC, BN, BMQ * ST, THREADS, ST, FLAT>(a, b);
        return;
    }
    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;

    K7Tiles<T, VEC, 1, BK, WQ, BN, SMAX, THREADS, (FLAT > 0)> tiles;
    tiles.init(a, b, tid, (TAPS - 1) * (ST == 1 ? a.dil : 1));
    if constexpr(FLAT > 0)
        tiles.flatten(a);

    // Phases computed: all ST, or (FLAT) the first FLAT, the rest zero.
    constexpr int NP = FLAT > 0 ? FLAT : ST;
    static_assert(NP <= ST, "K7 computed phases");
    float acc[TC][TQ][NP];
#pragma unroll
    for(int c = 0; c < TC; ++c)
#pragma unroll
        for(int i = 0; i < TQ; ++i)
#pragma unroll
            for(int p = 0; p < NP; ++p)
                acc[c][i][p] = 0.0f;

    // r = pad mod ST and whether the filter fills its last tap row are the same for every block:
    // one main loop per case, so the phases' shifts are compile-time.
    const bool full = a.fs == ST * TAPS;
    switch(b.r)
    {
#define K7_FMA_R(R)                                                                             \
    case R:                                                                                     \
        if constexpr(R < ST)                                                                    \
        {                                                                                       \
            if(full)                                                                            \
                k7FmaSteps<T, BMQ, BN, BK, TQ, TC, VEC, ST, TAPS, STAGES, R, true, NP, HALOK>(  \
                    a, b, tiles, lds, acc);                                                     \
            else                                                                                \
                k7FmaSteps<T, BMQ, BN, BK, TQ, TC, VEC, ST, TAPS, STAGES, R, false, NP, HALOK>( \
                    a, b, tiles, lds, acc);                                                     \
        }                                                                                       \
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

    if constexpr(FLAT > 0)
    {
        // Flattened outputs: phase p of position q0 + i is f = x0 + i ST + p; the ST phases of a
        // position are adjacent in its dx row (Wx = ST Wy). One position offset serves every
        // channel of the thread.
        const long long channelStride = static_cast<long long>(a.rows) * a.wx;
        const int end = a.n * a.rowsY * a.wx;
#pragma unroll
        for(int i = 0; i < TQ; ++i)
        {
            const int f0 = b.x0 + (ty * TQ + i) * ST;
            if(f0 >= end)
                continue;
            T* pos = static_cast<T*>(a.dx) + k7FlatPosition(a, b, f0);
#pragma unroll
            for(int c = 0; c < TC; ++c)
            {
                const int ch = b.c0 + tx + c * TX;
                if(ch >= a.c)
                    continue;
                T* dst = pos + ch * channelStride;
#pragma unroll
                for(int p = 0; p < ST; ++p)
                    dst[p] = p < NP ? K1Elem<T>::store(acc[c][i][p < NP ? p : 0]) : T{};
            }
        }
        return;
    }
    // Phase p at tile index i is dx position x0 + i ST + ((p - r) mod ST).
#pragma unroll
    for(int c = 0; c < TC; ++c)
    {
        const int ch = b.c0 + tx + c * TX;
        if(ch >= a.c)
            continue;
        if constexpr(FLAT > 0)
        {
            continue; // stored above, position by position
        }
        else
        {
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
}

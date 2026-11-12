// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K2: stride-1 halo-tile direct convolution, 2-D and 3-D, channels-last (NDHWC in, NDHWK out),
// for the CONV_DIRECT_FWD / CONV_DIRECT_BWD engines. HIPRTC-compatible: no standard headers.
// Reuses the implicit-GEMM core's (ConvIgemmCore.h) MMA atoms, element conversion, buffer loads
// and staging copies.
//
// A block owns an output tile: th x tw pixels of each of ds output planes (one image) x BN output
// channels. For every C-chunk of BC channels it stages the input halo (th + R - 1) x (tw + S - 1)
// x BC of one input plane in LDS once; each filter tap (r, s) is then a GEMM whose rows are the
// halo shifted by (r, s), whose reduction is the chunk, and whose columns are the tap's weights.
//
// 3-D, two modes (k2Drive):
//   RING 1 streams along depth: the block walks its input planes in order, each staged once, and
//     every plane feeds KD rotating accumulator sets (output plane l = input plane j - t gets
//     depth tap t). Set l finishes after plane l + KD - 1 and is stored then. The plane loop is
//     unrolled by KD so the set of every (plane, tap) pair is a compile-time constant.
//   RING 0 keeps one accumulator set: each output plane stages its KD input planes in turn (the
//     re-reads hit L2), so the 2-D tile and register budget carry over unchanged.
//
// Weights (MMA kernel, BL 0) are read straight from global (L1/L2-resident) in MMA fragment
// order, prepared by k2FormatWeights: wf[tap][c / KA][n][c % KA], C padded to the chunk and K to
// the block's columns with zeros, so the inner loop has no bounds checks; fragments are
// prefetched PD steps ahead in registers. LDS then holds only the halo, double buffered: the next
// chunk is loaded to registers while the current one is multiplied. BL 1 instead stages each
// tap's BN x BC weight tile in LDS (double-buffered, one barrier per tap; format ka = BC).
// The FMA kernel (fp32 on RDNA, any dtype anywhere) keeps an fp32 channel-major halo and slides
// TM + S - 1 values across the S taps; PIPE 1 double-buffers its weights in registers.
//
// Backward data at stride 1 is the same kernel: dx = conv(dy, w') with w' the filter rotated by
// 180 degrees and C/K swapped (done by k2FormatWeights), and padding F - 1 - p.
//
// Envelope: stride 1, dilation 1, groups 1, any padding, filter compile-time, per-image input
// < 2 GiB (32-bit buffer offsets).
#pragma once

#include "ConvDirectArgs.h"
#include "ConvIgemmCore.h"

// Weight fragments by buffer load, the lane's column as the vector offset and the step's
// wave-uniform offset as the scalar offset, where that measured faster than flat loads (spec
// lead bench, 2026-11-12): the matrix-core kernel on gfx950 and RDNA, the FMA kernel also on
// gfx90a. Elsewhere the flat loads stay (gfx90a's matrix-core kernel needs 20 more VGPRs in the
// buffer form and loses a wave per SIMD; gfx906/908/942 are not measured).
#if K1_HAS_WMMA11 || K1_HAS_WMMA12 || defined(__gfx950__)
#define K2_MMA_WBUF 1
#else
#define K2_MMA_WBUF 0
#endif
#if K2_MMA_WBUF || defined(__gfx90a__)
#define K2_FMA_WBUF 1
#else
#define K2_FMA_WBUF 0
#endif

/// Buffer load of 2, 4, 8 or 16 bytes at vector offset @p offset plus wave-uniform @p soffset.
template <class Raw>
__device__ inline Raw k2BufferLoadS(__amdgpu_buffer_rsrc_t r, unsigned offset, int soffset)
{
    static_assert(sizeof(Raw) == 16 || sizeof(Raw) == 8 || sizeof(Raw) == 4 || sizeof(Raw) == 2,
                  "buffer load size");
    if constexpr(sizeof(Raw) == 16)
        return __builtin_amdgcn_raw_buffer_load_b128(r, offset, soffset, 0);
    else if constexpr(sizeof(Raw) == 8)
        return __builtin_amdgcn_raw_buffer_load_b64(r, offset, soffset, 0);
    else if constexpr(sizeof(Raw) == 4)
        return __builtin_amdgcn_raw_buffer_load_b32(r, offset, soffset, 0);
    else
        return __builtin_amdgcn_raw_buffer_load_b16(r, offset, soffset, 0);
}

/// A matrix fragment by buffer loads (k2BufferLoadS), 16 bytes at a time past 16 bytes.
template <class Frag>
__device__ inline Frag k2FragLoadS(__amdgpu_buffer_rsrc_t r, unsigned offset, int soffset)
{
    if constexpr(sizeof(Frag) <= 16)
    {
        return __builtin_bit_cast(
            Frag, k2BufferLoadS<typename K1Raw<sizeof(Frag)>::type>(r, offset, soffset));
    }
    else
    {
        static_assert(sizeof(Frag) % 16 == 0, "fragment of whole 16-byte parts");
        struct Parts
        {
            typename K1Raw<16>::type p[sizeof(Frag) / 16];
        } parts;
#pragma unroll
        for(int i = 0; i < static_cast<int>(sizeof(Frag) / 16); ++i)
            parts.p[i] = k2BufferLoadS<typename K1Raw<16>::type>(r, offset + 16u * i, soffset);
        return __builtin_bit_cast(Frag, parts);
    }
}

/// Offset of filter element (forward roles) k, c, tap (t, r, s) through the graph's strides;
/// `mirror` reflects the taps.
__device__ inline long long
    k2FilterOffset(const K2FormatArgs& f, int k, int c, int tap, bool mirror)
{
    int s = tap % f.s;
    int rest = tap / f.s;
    int r = rest % f.r;
    int t = rest / f.r;
    if(mirror)
    {
        t = f.kd - 1 - t;
        r = f.r - 1 - r;
        s = f.s - 1 - s;
    }
    return k * f.wStrides[0] + c * f.wStrides[1] + t * f.wStrides[2] + r * f.wStrides[3]
           + s * f.wStrides[4];
}

/// wf[((tap * cpad / ka + c / ka) * kpad + n) * ka + c % ka]; zeros past the real channels.
template <class T>
__device__ void k2FormatWeights(const K2FormatArgs& f)
{
    const T* w = static_cast<const T*>(f.w);
    T* wf = static_cast<T*>(f.wf);
    const int taps = f.kd * f.r * f.s;
    const int cas = f.cpad / f.ka;
    // Runs of ka reduction elements (tap, chunk, column n): one filter offset per run and 32-bit
    // index math (the planner keeps the formatted weights below 2^31 elements). The pass runs on
    // every execution: per element (2-byte accesses) it took 0.105 ms for a 384 x 384 x 27 bf16
    // filter on gfx1201, a quarter of the convolution; the 16-byte form below takes 0.016 ms.
    const int runs = taps * cas * f.kpad;
    const int cred = f.backward ? f.k0 : f.c0;
    const long long step = f.backward ? f.wStrides[0] : f.wStrides[1];
    // Backward data rotates the taps, true convolution reflects them: together they cancel.
    const bool mirror = (f.backward != 0) != (f.flip != 0);
    // Channels-last forward filter with 16-byte aligned channel runs: 16-byte loads and stores.
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    typedef unsigned K2U4 __attribute__((ext_vector_type(4)));
    const bool vec = !f.backward && f.wStrides[1] == 1 && f.ka % VEC == 0 && cred % VEC == 0
                     && f.wStrides[0] % VEC == 0 && f.wStrides[2] % VEC == 0
                     && f.wStrides[3] % VEC == 0 && f.wStrides[4] % VEC == 0
                     && reinterpret_cast<unsigned long long>(w) % 16 == 0
                     && reinterpret_cast<unsigned long long>(wf) % 16 == 0;
    if(vec)
    {
        const int vpr = f.ka / VEC;
        const int units = runs * vpr;
        for(int u = blockIdx.x * blockDim.x + threadIdx.x; u < units; u += gridDim.x * blockDim.x)
        {
            const int i = u / vpr;
            const int v = u - i * vpr;
            const int n = i % f.kpad;
            const int rest = i / f.kpad;
            const int ca = rest % cas;
            const int tap = rest / cas;
            const int c = ca * f.ka + v * VEC;
            K2U4 val = {0u, 0u, 0u, 0u};
            if(n < f.k0 && c < cred)
                val = *reinterpret_cast<const K2U4*>(w + k2FilterOffset(f, n, c, tap, mirror));
            *reinterpret_cast<K2U4*>(wf + static_cast<long long>(u) * VEC) = val;
        }
        return;
    }
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < runs; i += gridDim.x * blockDim.x)
    {
        const int n = i % f.kpad;
        const int rest = i / f.kpad;
        const int ca = rest % cas;
        const int tap = rest / cas;
        const int c0 = ca * f.ka;
        const bool live = f.backward ? n < f.c0 : n < f.k0;
        const long long base = !live        ? 0
                               : f.backward ? k2FilterOffset(f, c0, n, tap, mirror)
                                            : k2FilterOffset(f, n, c0, tap, mirror);
        T* dst = wf + static_cast<long long>(i) * f.ka;
        for(int kk = 0; kk < f.ka; ++kk)
        {
            T v = K1Elem<T>::store(0.0f);
            if(live && c0 + kk < cred)
                v = w[base + kk * step];
            dst[kk] = v;
        }
    }
}

/// Split-C reduce: y[i] = sum over z in order of ws[z * count + i]. Deterministic.
template <class TO>
__device__ void k2SplitReduce(const K2ReduceArgs& a)
{
    TO* y = static_cast<TO*>(a.y);
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < a.count;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        float v = 0.0f;
        for(int z = 0; z < a.splits; ++z)
            v += a.ws[z * a.count + i];
        y[i] = K1Elem<TO>::store(v);
    }
}

// Three-bf16 fp32 (TF32-class): x * w ~= xh wh + xh wl + xl wh, run as one bf16 conv with 3C
// reduction channels, x' = [xh | xh | xl] and w' = [wh | wl | wh] along the reduction, fp32 out.
__device__ inline void k2SplitBf16(float v, unsigned short& hi, unsigned short& lo)
{
    hi = k1FloatToBf(v);
    lo = k1BfLo(v, hi);
}

/// x' from x: pixels x c fp32 -> pixels x 3c bf16.
__device__ inline void k2SplitInput(const K2SplitArgs& a)
{
    const float* x = static_cast<const float*>(a.x);
    unsigned short* xs = static_cast<unsigned short*>(a.xs);
    const long long total = a.pixels * a.c;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const long long p = i / a.c;
        const int ch = static_cast<int>(i - p * a.c);
        unsigned short hi, lo;
        k2SplitBf16(x[i], hi, lo);
        unsigned short* d = xs + p * 3 * a.c + ch;
        d[0] = hi;
        d[a.c] = k1BfFinite(hi); // pairs with the weights' lo
        d[2 * a.c] = lo;
    }
}

/// Formatted w' (bf16) from the fp32 filter; the reduction is 3x the real one.
__device__ inline void k2FormatWeightsSplit(const K2FormatArgs& f)
{
    const float* w = static_cast<const float*>(f.w);
    unsigned short* wf = static_cast<unsigned short*>(f.wf);
    const int taps = f.kd * f.r * f.s;
    const long long total = static_cast<long long>(taps) * f.cpad * f.kpad;
    const int cas = f.cpad / f.ka;
    const int cred = f.backward ? f.k0 : f.c0;
    const bool mirror = (f.backward != 0) != (f.flip != 0);
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int kk = static_cast<int>(i % f.ka);
        long long rest = i / f.ka;
        const int n = static_cast<int>(rest % f.kpad);
        rest /= f.kpad;
        const int ca = static_cast<int>(rest % cas);
        const int tap = static_cast<int>(rest / cas);
        const int cs = ca * f.ka + kk;
        unsigned short out = 0;
        if(cs < 3 * cred)
        {
            const int part = cs / cred, c = cs - part * cred;
            const bool ok = f.backward ? n < f.c0 : n < f.k0;
            if(ok)
            {
                const float v = f.backward ? w[k2FilterOffset(f, c, n, tap, mirror)]
                                           : w[k2FilterOffset(f, n, c, tap, mirror)];
                unsigned short hi, lo;
                k2SplitBf16(v, hi, lo);
                out = part == 1 ? lo : part == 2 ? k1BfFinite(hi) : hi; // part 2 pairs with x lo
            }
        }
        wf[i] = out;
    }
}

// Block decode shared by both kernels.
struct K2Block
{
    int img, oh0, ow0, od0, dsEff, hw, hp;
    __device__ void init(const K2Args& a, int R, int S)
    {
        int bx = blockIdx.x;
        const int tWi = bx % a.tilesW;
        bx /= a.tilesW;
        const int tHi = bx % a.tilesH;
        bx /= a.tilesH;
        const int tDi = bx % a.tilesD;
        img = bx / a.tilesD;
        oh0 = tHi * a.th;
        ow0 = tWi * a.tw;
        od0 = tDi * a.ds;
        dsEff = a.dout - od0 < a.ds ? a.dout - od0 : a.ds;
        hw = a.tw + S - 1;
        hp = (a.th + R - 1) * hw;
    }
};

// Halo loader: VEC channels of PASSES halo pixels per thread, zero outside the input.
template <class T, int BC, int VEC, int THREADS, int HALO>
struct K2HaloLoad
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int PT = BC / VEC;
    static constexpr int PPP = THREADS / PT;
    static constexpr int PASSES = (HALO + PPP - 1) / PPP;
    static_assert(BC % VEC == 0 && THREADS % PT == 0, "halo load");

    __amdgpu_buffer_rsrc_t rsrc;
    int kv, prow, hp, planeElems, cIn;
    int hOff[PASSES];
    Raw r[PASSES];

    __device__ void init(const K2Args& a, const K2Block& b, int tid)
    {
        kv = (tid % PT) * VEC;
        prow = tid / PT;
        hp = b.hp;
        cIn = a.c;
        planeElems = a.hi * a.wi * a.c;
        const T* img
            = static_cast<const T*>(a.x) + static_cast<long long>(b.img) * a.di * planeElems;
        rsrc = k1Rsrc(
            img, static_cast<long long>(a.di) * planeElems * static_cast<long long>(sizeof(T)));
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int pix = prow + p * PPP;
            const int hh = pix / b.hw;
            const int ww = pix - hh * b.hw;
            const int ih = b.oh0 - a.ph + hh, iw = b.ow0 - a.pw + ww;
            hOff[p] = pix < b.hp && static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                              && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)
                          ? (ih * a.wi + iw) * a.c
                          : -1;
        }
    }
    // Input plane id (in range), channel chunk c0.
    __device__ void load(int id, int c0)
    {
        const int c = c0 + kv;
        const bool cOk = c < cIn;
        const int base = id * planeElems + c;
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const unsigned off
                = cOk && hOff[p] >= 0
                      ? static_cast<unsigned>((base + hOff[p]) * static_cast<int>(sizeof(T)))
                      : K1_OUT_OF_RANGE;
            r[p] = k1BufferLoad<Raw>(rsrc, off);
        }
    }
    // Pixel-major: buf[pix * LDC + c].
    template <int LDC>
    __device__ void storePixels(T* buf) const
    {
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int pix = prow + p * PPP;
            if(pix < hp)
                *reinterpret_cast<Raw*>(buf + pix * LDC + kv) = r[p];
        }
    }
    // Channel-major fp32: buf[c * LDH + pix].
    template <int LDH>
    __device__ void storeChannels(float* buf) const
    {
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int pix = prow + p * PPP;
            if(pix < hp)
            {
                const T* v = reinterpret_cast<const T*>(&r[p]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    buf[(kv + e) * LDH + pix] = K1Elem<T>::load(v[e]);
            }
        }
    }
};

// Walks the block's (input plane, channel chunk) steps in order, skipping planes outside the
// input. planes = dsEff + KD - 1, plane j is input plane od0 - pd + j.
struct K2Cursor
{
    int j, cc, planes, nCh, first, di;
    __device__ bool ok(int jj) const
    {
        return static_cast<unsigned>(first + jj) < static_cast<unsigned>(di);
    }
    __device__ void init(int planes_, int nCh_, int first_, int di_)
    {
        planes = planes_;
        nCh = nCh_;
        first = first_;
        di = di_;
        cc = 0;
        j = 0;
        while(j < planes && !ok(j))
            ++j;
    }
    __device__ bool more() const
    {
        return j < planes;
    }
    __device__ int plane() const
    {
        return first + j;
    }
    __device__ void advance()
    {
        if(++cc == nCh)
        {
            cc = 0;
            do
            {
                ++j;
            } while(j < planes && !ok(j));
        }
    }
};

// Non-ring order: output plane l, depth tap t (input plane od0 - pd + l + t), channel chunk cc.
struct K2CursorTaps
{
    int l, t, cc, ds, kd, nCh, first, di;
    __device__ bool ok(int ll, int tt) const
    {
        return static_cast<unsigned>(first + ll + tt) < static_cast<unsigned>(di);
    }
    __device__ void skip()
    {
        while(l < ds && !ok(l, t))
        {
            if(++t == kd)
            {
                t = 0;
                ++l;
            }
        }
    }
    __device__ void init(int ds_, int kd_, int nCh_, int first_, int di_)
    {
        ds = ds_;
        kd = kd_;
        nCh = nCh_;
        first = first_;
        di = di_;
        l = t = cc = 0;
        skip();
    }
    __device__ bool more() const
    {
        return l < ds;
    }
    __device__ int plane() const
    {
        return first + l + t;
    }
    __device__ void advance()
    {
        if(++cc == nCh)
        {
            cc = 0;
            if(++t == kd)
            {
                t = 0;
                ++l;
            }
            skip();
        }
    }
};

template <int V>
struct K2Int
{
    static constexpr int value = V;
};
template <int I, int N, class F>
__device__ inline void k2StaticFor(F&& f)
{
    if constexpr(I < N)
    {
        f(K2Int<I>{});
        k2StaticFor<I + 1, N>(f);
    }
}

// The block's main loop, shared by both kernels. Double-buffered halo: each step waits on one
// barrier, loads the next step's halo to registers (pf, called by the multiply where it suits
// the vmcnt order), multiplies the current stage, then stores the registers to the other stage.
//   load(plane, cc)                     next halo chunk to registers
//   commit(stage)                       registers to LDS stage
//   ring(stage, cc, jj, K2Int<u>, pf, nx)  RING: multiply the stage into the sets of plane jj (u = jj % KD)
//   tap(stage, cc, t, pf, nx)              !RING: multiply the stage with depth tap t into set 0
//   (nx(t, cc) gives the next step's depth tap and chunk, false at the end.)
//   store(set, od, stage)           epilogue of a finished output plane
//
// RING 1: input planes stream once through KD rotating accumulator sets (output plane l gets
// input plane l + t with depth tap t, in set l % KD; the plane loop is unrolled by KD so every set
// index is a compile-time constant). RING 0: one set; each output plane re-stages its KD input
// planes (L2 hits), so the 2-D tiles and register budget apply unchanged.
template <int KD, int RING, class Load, class Commit, class Ring, class Tap, class Store>
__device__ inline void k2Drive(const K2Block& b,
                               int pd,
                               int di,
                               int nCh,
                               Load&& load,
                               Commit&& commit,
                               Ring&& ring,
                               Tap&& tap,
                               Store&& store)
{
    const int first = b.od0 - pd;
    int stage = 0;
    if constexpr(RING)
    {
        const int planes = b.dsEff + KD - 1;
        K2Cursor cur;
        cur.init(planes, nCh, first, di);
        if(cur.more())
        {
            load(cur.plane(), 0);
            commit(0);
            cur.advance();
        }
        for(int j0 = 0; j0 < planes; j0 += KD)
        {
            k2StaticFor<0, KD>([&](auto U) {
                constexpr int u = decltype(U)::value;
                const int jj = j0 + u;
                if(jj >= planes)
                    return;
                if(cur.ok(jj))
                {
                    for(int cc = 0; cc < nCh; ++cc)
                    {
                        __syncthreads();
                        const bool more = cur.more();
                        auto pf = [&]() {
                            if(more)
                                load(cur.plane(), cur.cc);
                        };
                        auto nx = [&](int& t, int& c) {
                            t = 0;
                            c = cur.cc;
                            return more;
                        };
                        ring(stage, cc, jj, U, pf, nx);
                        if(more)
                        {
                            commit(stage ^ 1);
                            cur.advance();
                        }
                        stage ^= 1;
                    }
                }
                if(jj >= KD - 1)
                    store((u + 1) % KD, b.od0 + jj - (KD - 1), stage);
            });
        }
    }
    else
    {
        K2CursorTaps cur;
        cur.init(b.dsEff, KD, nCh, first, di);
        if(cur.more())
        {
            load(cur.plane(), 0);
            commit(0);
            cur.advance();
        }
        for(int l = 0; l < b.dsEff; ++l)
        {
            for(int t = 0; t < KD; ++t)
            {
                if(!cur.ok(l, t))
                    continue;
                for(int cc = 0; cc < nCh; ++cc)
                {
                    __syncthreads();
                    const bool more = cur.more();
                    auto pf = [&]() {
                        if(more)
                            load(cur.plane(), cur.cc);
                    };
                    auto nx = [&](int& tn, int& c) {
                        tn = cur.t;
                        c = cur.cc;
                        return more;
                    };
                    tap(stage, cc, t, pf, nx);
                    if(more)
                    {
                        commit(stage ^ 1);
                        cur.advance();
                    }
                    stage ^= 1;
                }
            }
            store(0, b.od0 + l, stage);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Matrix-core kernel. WM x WN waves; block tile BM pixel rows (th * tw <= BM) x BN columns.

// BL 0: weight fragments straight from global (L1/L2), PD steps ahead in registers; weights
// formatted with ka = Atom::KA. BL 1: per in-plane tap, the block stages the tap's BN x BC weight
// tile (one per depth set) in LDS, double-buffered across taps with one barrier per tap, like the
// halo; weights formatted with ka = BC, so each (tap, chunk) tile is contiguous in global.
// TO: output type (T, or fp32 for the three-bf16 form whose operands are bf16).
// SPLIT 1: split-C config; the block reduces chunks [z cps, (z+1) cps) and the epilogue writes fp32
// partials to ws[z] (k2SplitReduce finishes). SPLIT 0 writes TO to y and needs splits = 1.
template <class T,
          class Atom,
          int KD,
          int R,
          int S,
          int BM,
          int BN,
          int BC,
          int WM,
          int WN,
          int VEC,
          int RING = 1,
          int BL = 0,
          class TO = T,
          int SPLIT = 0,
          int HALO = k2HaloMax(BM, R, S)>
__device__ void k2ConvMma(const K2Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WAVE;
    constexpr int DIM = Atom::DIM;
    constexpr int KA = Atom::KA;
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int FM = WTM / DIM, FN = WTN / DIM;
    constexpr int LDC = BC + 16 / static_cast<int>(sizeof(T));
    constexpr int STAGE = HALO * LDC;
    typedef
        typename K1If<SPLIT != 0, float, TO>::type TS; // epilogue element: fp32 partials or output
    constexpr int VO = 16 / static_cast<int>(sizeof(TS));
    constexpr int OLD = WTN + VO;
    constexpr int STRIP = DIM * OLD; // in TS
    constexpr int SETS = RING ? KD : 1;
    static_assert(WTM % DIM == 0 && WTN % DIM == 0 && BC % KA == 0 && WTN % VO == 0, "tile");
    typedef K2HaloLoad<T, BC, VEC, THREADS, HALO> Loader;
    typedef typename Atom::frag Frag;

    // Two halo stages; the epilogue strips reuse the stage just consumed (after a barrier).
    constexpr int STRIPS
        = (WM * WN * STRIP * static_cast<int>(sizeof(TS)) + static_cast<int>(sizeof(T)) - 1)
          / static_cast<int>(sizeof(T)); // in T
    constexpr int SLOT = STAGE > STRIPS ? STAGE : STRIPS;
    // BL: two weight stages of NBT tiles [BN][BLD].
    constexpr int NBT = RING ? KD : 1;
    constexpr int BLD = BC + 16 / static_cast<int>(sizeof(T));
    constexpr int BTILE = BN * BLD;
    constexpr int BSTAGE = BL ? NBT * BTILE : 0;
    __shared__ __attribute__((aligned(16))) T lds[2 * SLOT + 2 * BSTAGE];
    T* const bls = lds + 2 * SLOT;

    K2Block b;
    b.init(a, R, S);
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wm = wave % WM, wn = wave / WM;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    const int tilePix = a.th * a.tw;

    Loader ld;
    ld.init(a, b, tid);
    // Split-C: this block's channel chunks [cb, cb + nChL).
    const int nChAll = (a.c + BC - 1) / BC;
    const int cb = SPLIT ? blockIdx.z * a.cps : 0;
    const int nChL = SPLIT ? (nChAll - cb < a.cps ? nChAll - cb : a.cps) : nChAll;

    // Each lane's fragment rows: halo pixel of tap (0, 0), times LDC, plus its k base.
    int aOff[FM];
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
        const int m = wm * WTM + i * DIM + lane % DIM;
        const int ohl = m / a.tw, owl = m - ohl * a.tw;
        aOff[i] = (m < tilePix ? ohl * b.hw + owl : 0) * LDC + kb;
    }
    int bCol[FN];
#pragma unroll
    for(int j = 0; j < FN; ++j)
        bCol[j] = (n0 + wn * WTN + j * DIM + lane % DIM) * KA + kb;
    const T* wf = static_cast<const T*>(a.w);
    const int rowStep = b.hw * LDC;
    const int tapStride = a.ca * a.kpad * KA; // elements per tap in wf
#if K2_MMA_WBUF
    // Register-path weights (BL 0) through a buffer: the lane's column bytes are the vector
    // offset, fixed for the whole kernel; the step's (chunk, k-step, tap) offset is wave-uniform
    // and goes in the scalar offset, so a fragment load costs no VALU address arithmetic. The
    // planner keeps the formatted weights within the buffer's 31-bit range.
    const __amdgpu_buffer_rsrc_t wRsrc = k1Rsrc(
        wf, static_cast<long long>(KD) * R * S * tapStride * static_cast<long long>(sizeof(T)));
    unsigned bByte[FN];
#pragma unroll
    for(int j = 0; j < FN; ++j)
        bByte[j] = static_cast<unsigned>(bCol[j]) * static_cast<unsigned>(sizeof(T));
#endif

    typename Atom::acc acc[SETS][FM][FN];
#pragma unroll
    for(int t = 0; t < SETS; ++t)
#pragma unroll
        for(int i = 0; i < FM; ++i)
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[t][i][j] = typename Atom::acc{};

    // One chunk, as NQ = (BC / KA) * R * S steps (k-step, in-plane tap). The weight fragments are
    // prefetched PD steps ahead in registers; the next chunk's halo loads go out after the first
    // step's weights, so the in-order vmcnt never makes the first weights wait on the halo, and
    // the halo has PD steps of MMAs to land. NB depth taps per step: KD sets (ring) or one (tap mode).
    // tapOf(t) gives the depth tap feeding set t (or -1 to skip it).
    auto chunk = [&](int stage, int cc, auto&& pf, auto NBc, auto&& tapOf, auto&& setOf) {
        constexpr int NB = decltype(NBc)::value;
        constexpr int NQ = (BC / KA) * R * S;
        const T* halo = lds + stage * SLOT;
        int dt[NB];
#pragma unroll
        for(int t = 0; t < NB; ++t)
            dt[t] = tapOf(t);
        auto loadB = [&](int q, Frag(&bf)[NB][FN]) {
            const int kk = q / (R * S), rs = q % (R * S);
#if K2_MMA_WBUF
            const unsigned wq = static_cast<unsigned>(((cb + cc) * (BC / KA) + kk) * a.kpad * KA
                                                      + rs * tapStride);
#pragma unroll
            for(int t = 0; t < NB; ++t)
            {
                const unsigned so
                    = (wq + static_cast<unsigned>((dt[t] < 0 ? 0 : dt[t]) * R * S * tapStride))
                      * static_cast<unsigned>(sizeof(T));
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    bf[t][j] = k2FragLoadS<Frag>(wRsrc, bByte[j], static_cast<int>(so));
            }
#else
            const T* wq = wf + ((cb + cc) * (BC / KA) + kk) * a.kpad * KA + rs * tapStride;
#pragma unroll
            for(int t = 0; t < NB; ++t)
            {
                const T* wt = wq + (dt[t] < 0 ? 0 : dt[t]) * R * S * tapStride;
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    bf[t][j] = *reinterpret_cast<const Frag*>(wt + bCol[j]);
            }
#endif
        };
        // Weight fragments PD steps ahead, in a ring indexed by compile-time step numbers. PD from
        // a ~16-VGPR budget: deep for fp32 (one VGPR per fragment), one step for big fragments.
        constexpr int FRAG_VGPRS = (sizeof(Frag) + 3) / 4;
        constexpr int PD0 = 16 / (NB * FN * FRAG_VGPRS);
        constexpr int PD = PD0 < 1 ? 1 : (PD0 > 4 ? 4 : PD0);
        Frag bq[PD + 1][NB][FN];
#pragma unroll
        for(int q = 0; q < PD && q < NQ; ++q)
        {
            loadB(q, bq[q]);
            if(q == 0)
                pf();
        }
#pragma unroll
        for(int q = 0; q < NQ; ++q)
        {
            if(q + PD < NQ)
                loadB(q + PD, bq[(q + PD) % (PD + 1)]);
            const int kk = (q / (R * S)) * KA, r = (q % (R * S)) / S, s = q % S;
            Frag af[FM];
#pragma unroll
            for(int i = 0; i < FM; ++i)
                af[i] = k1Frag<Atom>(halo + aOff[i] + r * rowStep + s * LDC + kk);
#pragma unroll
            for(int t = 0; t < NB; ++t)
            {
                if(dt[t] < 0)
                    continue;
                auto& ac = acc[setOf(t)];
#pragma unroll
                for(int i = 0; i < FM; ++i)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        ac[i][j] = Atom::mma(af[i], bq[q % (PD + 1)][t][j], ac[i][j]);
            }
        }
    };
    // BL: weight tiles through LDS. Per thread LOADS vectors of VB elements per tile.
    constexpr int VB = 16 / static_cast<int>(sizeof(T));
    constexpr int LOADS = BL ? (BN * BC) / (VB * THREADS) : 1;
    static_assert(!BL || (BN * BC) % (VB * THREADS) == 0, "weight tile loads");
    typedef typename K1Raw<16>::type RawB;
    RawB rb[NBT][LOADS];
    int bstage = 0;
    // Tile of (depth tap dt, in-plane tap rs, chunk c): wf[((dt R S + rs) ca + c) kpad + n0][BC].
    auto loadBT = [&](const int(&dts)[NBT], int rs, int c) {
#pragma unroll
        for(int t = 0; t < NBT; ++t)
        {
            const T* src
                = wf
                  + ((static_cast<long long>(dts[t] * R * S + rs) * a.ca + cb + c) * a.kpad + n0)
                        * BC;
#pragma unroll
            for(int l = 0; l < LOADS; ++l)
                rb[t][l] = *reinterpret_cast<const RawB*>(src + (l * THREADS + tid) * VB);
        }
    };
    auto commitBT = [&](int st) {
#pragma unroll
        for(int t = 0; t < NBT; ++t)
#pragma unroll
            for(int l = 0; l < LOADS; ++l)
            {
                const int idx = (l * THREADS + tid) * VB;
                *reinterpret_cast<RawB*>(bls + st * BSTAGE + t * BTILE + (idx / BC) * BLD
                                         + idx % BC)
                    = rb[t][l];
            }
    };
    int bRow[FN];
#pragma unroll
    for(int j = 0; j < FN; ++j)
        bRow[j] = (wn * WTN + j * DIM + lane % DIM) * BLD + kb;
    // One chunk, per in-plane tap: barrier (the driver's for the first), the next tap's tiles to
    // registers (or the next chunk's first tap: nx), then the MMAs, then the tiles to the other
    // stage. The halo prefetch goes out after the first tap's weight loads.
    auto chunkL = [&](int stage,
                      int cc,
                      auto&& pf,
                      auto&& nx,
                      const int(&dts)[NBT],
                      auto&& active,
                      auto&& setOf) {
        const T* halo = lds + stage * SLOT;
#pragma unroll
        for(int rs = 0; rs < R * S; ++rs)
        {
            if(rs > 0)
                __syncthreads();
            bool next = true;
            if(rs + 1 < R * S)
            {
                loadBT(dts, rs + 1, cc);
            }
            else
            {
                int tn, cn;
                next = nx(tn, cn);
                if(next)
                {
                    int dn[NBT];
#pragma unroll
                    for(int t = 0; t < NBT; ++t)
                        dn[t] = RING ? t : tn;
                    loadBT(dn, 0, cn);
                }
            }
            if(rs == 0)
                pf();
            const int r = rs / S, s = rs % S;
            const T* bt = bls + bstage * BSTAGE;
#pragma unroll
            for(int kk = 0; kk < BC; kk += KA)
            {
                Frag af[FM];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    af[i] = k1Frag<Atom>(halo + aOff[i] + r * rowStep + s * LDC + kk);
#pragma unroll
                for(int t = 0; t < NBT; ++t)
                {
                    if(!active(t))
                        continue;
                    Frag bf[FN];
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        bf[j] = k1Frag<Atom>(bt + t * BTILE + bRow[j] + kk);
                    auto& ac = acc[setOf(t)];
#pragma unroll
                    for(int i = 0; i < FM; ++i)
#pragma unroll
                        for(int j = 0; j < FN; ++j)
                            ac[i][j] = Atom::mma(af[i], bf[j], ac[i][j]);
                }
            }
            if(next)
                commitBT(bstage ^ 1);
            bstage ^= 1;
        }
    };
    if constexpr(BL)
    {
        // Prologue: the first step's tap-0 tiles (the driver's first barrier orders them).
        int d0[NBT];
        if constexpr(RING)
        {
#pragma unroll
            for(int t = 0; t < NBT; ++t)
                d0[t] = t;
        }
        else
        {
            K2CursorTaps c0;
            c0.init(b.dsEff, KD, (a.c + BC - 1) / BC, b.od0 - a.pd, a.di);
            d0[0] = c0.more() ? c0.t : 0;
        }
        loadBT(d0, 0, 0);
        commitBT(0);
    }

    auto ring = [&](int stage, int cc, int jj, auto U, auto&& pf, auto&& nx) {
        constexpr int u = decltype(U)::value;
        if constexpr(BL)
        {
            int dts[NBT];
#pragma unroll
            for(int t = 0; t < NBT; ++t)
                dts[t] = t;
            chunkL(
                stage,
                cc,
                pf,
                nx,
                dts,
                [&](int t) { return jj - t >= 0 && jj - t < b.dsEff; },
                [&](int t) { return (u - t + KD) % KD; });
        }
        else
        {
            chunk(
                stage,
                cc,
                pf,
                K2Int<KD>{},
                [&](int t) { return jj - t >= 0 && jj - t < b.dsEff ? t : -1; },
                [&](int t) { return (u - t + KD) % KD; });
        }
    };
    auto tap = [&](int stage, int cc, int t0, auto&& pf, auto&& nx) {
        if constexpr(BL)
        {
            const int dts[1] = {t0};
            chunkL(stage, cc, pf, nx, dts, [&](int) { return true; }, [&](int) { return 0; });
        }
        else
        {
            chunk(stage, cc, pf, K2Int<1>{}, [&](int) { return t0; }, [&](int) { return 0; });
        }
    };

    TS* out
        = SPLIT ? reinterpret_cast<TS*>(
                      a.ws + static_cast<long long>(blockIdx.z) * a.n * a.dout * a.ho * a.wo * a.k)
                : reinterpret_cast<TS*>(a.y);
    // Epilogue: per DIM-row slab, accumulators to the wave's LDS strip, then 16-byte rows to
    // global. The strips use the halo stage not holding the next chunk: all waves are done with it
    // after the barrier, and the next chunk's barrier orders the strips before its reuse.
    auto store = [&](int set, int od, int stage) {
        constexpr int RV = WTN / VO;
        typedef typename K1Raw<16>::type Raw;
        __syncthreads();
        TS* strip = reinterpret_cast<TS*>(lds + (stage ^ 1) * SLOT) + wave * STRIP;
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
#pragma unroll
            for(int j = 0; j < FN; ++j)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                    strip[Atom::dRow(lane, e) * OLD + j * DIM + lane % DIM]
                        = K1Elem<TS>::store(acc[set][i][j][e]);
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            for(int v = lane; v < DIM * RV; v += WAVE)
            {
                const int rr = v / RV;
                const int col = (v - rr * RV) * VO;
                const int m = wm * WTM + i * DIM + rr;
                const int nn = n0 + wn * WTN + col;
                if(m >= tilePix || nn >= a.k)
                    continue;
                const int ohl = m / a.tw, owl = m - ohl * a.tw;
                const int oh = b.oh0 + ohl, ow = b.ow0 + owl;
                if(oh >= a.ho || ow >= a.wo)
                    continue;
                const long long o
                    = (((static_cast<long long>(b.img) * a.dout + od) * a.ho + oh) * a.wo + ow)
                          * a.k
                      + nn;
                if(nn + VO <= a.k && a.k % VO == 0)
                    *reinterpret_cast<Raw*>(out + o)
                        = *reinterpret_cast<const Raw*>(strip + rr * OLD + col);
                else
                    for(int e = 0; e < VO && nn + e < a.k; ++e)
                        out[o + e] = strip[rr * OLD + col + e];
            }
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[set][i][j] = typename Atom::acc{};
        }
    };

    k2Drive<KD, RING>(
        b,
        a.pd,
        a.di,
        nChL,
        [&](int plane, int cc) { ld.load(plane, (cb + cc) * BC); },
        [&](int stage) { ld.template storePixels<LDC>(lds + stage * SLOT); },
        ring,
        tap,
        store);
}

// ---------------------------------------------------------------------------------------------
// FMA kernel (fp32 on RDNA; every dtype anywhere). Each thread owns TM consecutive output pixels
// of one row and TN consecutive output channels; per (channel, filter row) it reads the TM + S - 1
// halo values once and slides them across the S taps. The halo is fp32, channel-major in LDS;
// weights come from global in [tap][c][kpad] order (k2FormatWeights with ka = 1).

// PIPE 1: the single-depth-tap path double-buffers the weights one (channel, filter row) step
// ahead in registers; PIPE 0 loads them at use (fewer VGPRs; better for big filters).
template <class T,
          int KD,
          int R,
          int S,
          int BC,
          int TM,
          int TN,
          int NT,
          int THREADS,
          int VEC,
          int RING = 1,
          int PIPE = 0,
          int HALO = k2HaloMax((THREADS / NT) * TM, R, S)>
__device__ void k2ConvFma(const K2Args& a)
{
    constexpr int BN = NT * TN;
    constexpr int LDH = HALO + 1;
    constexpr int STAGE = BC * LDH;
    constexpr int SV
        = TN < 16 / static_cast<int>(sizeof(T)) ? TN : 16 / static_cast<int>(sizeof(T));
    constexpr int SETS = RING ? KD : 1;
    static_assert(TN % SV == 0, "TN");
    typedef K2HaloLoad<T, BC, VEC, THREADS, HALO> Loader;
    typedef typename K1Raw<SV * sizeof(T)>::type Raw;

    __shared__ __attribute__((aligned(16))) float lds[2 * STAGE];

    K2Block b;
    b.init(a, R, S);
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int tn = tid % NT, tq = tid / NT;
    const int runs = a.tw / TM;
    const int ohl = tq / runs, owl = (tq - ohl * runs) * TM;
    const bool rowOk = ohl < a.th;
    const int aBase = rowOk ? ohl * b.hw + owl : 0;

    Loader ld;
    ld.init(a, b, tid);
#if K2_FMA_WBUF
    // Weights through a buffer: the thread's TN columns are the vector offset, fixed for the
    // whole kernel; a step's (tap, channel) position is a wave-uniform element offset of
    // [tap][c][kpad] and goes in the scalar offset (no per-load address arithmetic).
    typedef int WPos;
    const WPos wf = 0;
    const int tapStride = a.ca * a.kpad;
    const __amdgpu_buffer_rsrc_t wRsrc = k1Rsrc(
        a.w, static_cast<long long>(KD) * R * S * tapStride * static_cast<long long>(sizeof(T)));
    const unsigned wLane = static_cast<unsigned>(n0 + tn * TN) * static_cast<unsigned>(sizeof(T));
    auto wLoad = [&](WPos e, int q) {
        return k2BufferLoadS<Raw>(wRsrc,
                                  wLane + static_cast<unsigned>(q * SV * sizeof(T)),
                                  static_cast<int>(static_cast<unsigned>(e) * sizeof(T)));
    };
#else
    // Weights by flat loads: a step's position is a pointer to the thread's TN columns.
    typedef const T* WPos;
    WPos wf = static_cast<const T*>(a.w) + n0 + tn * TN;
    const int tapStride = a.ca * a.kpad;
    auto wLoad = [&](WPos p, int q) { return *reinterpret_cast<const Raw*>(p + q * SV); };
#endif

    float acc[SETS][TM][TN];
#pragma unroll
    for(int t = 0; t < SETS; ++t)
#pragma unroll
        for(int i = 0; i < TM; ++i)
#pragma unroll
            for(int j = 0; j < TN; ++j)
                acc[t][i][j] = 0.0f;

    // One chunk: per channel and filter row the TM + S - 1 halo values, then per in-plane tap
    // taps(av, s, weights of depth tap 0).
    auto chunk = [&](int stage, int cc, auto&& taps) {
        const float* halo = lds + stage * STAGE + aBase;
#pragma unroll 2
        for(int c = 0; c < BC; ++c)
        {
            const WPos wc = wf + (cc * BC + c) * a.kpad;
#pragma unroll
            for(int r = 0; r < R; ++r)
            {
                float av[TM + S - 1];
#pragma unroll
                for(int q = 0; q < TM + S - 1; ++q)
                    av[q] = halo[c * LDH + r * b.hw + q];
#pragma unroll
                for(int s = 0; s < S; ++s)
                    taps(av, s, wc + (r * S + s) * tapStride);
            }
        }
    };
    auto fmaInto = [&](float(&ac)[TM][TN], const float(&av)[TM + S - 1], int s, WPos wt) {
        float bv[TN];
#pragma unroll
        for(int q = 0; q < TN / SV; ++q)
        {
            const Raw raw = wLoad(wt, q);
            const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
            for(int v = 0; v < SV; ++v)
                bv[q * SV + v] = K1Elem<T>::load(e[v]);
        }
#pragma unroll
        for(int i = 0; i < TM; ++i)
#pragma unroll
            for(int j = 0; j < TN; ++j)
                ac[i][j] = __builtin_fmaf(av[i + s], bv[j], ac[i][j]);
    };
    // One depth tap t0 into set 0 (2-D, and 3-D without the ring). Steps are (channel, filter
    // row); the S weight vectors of the next step are loaded while the current one is multiplied
    // (two buffers, static parity: an odd R takes two channels per iteration), and the next
    // chunk's halo loads go out after the first step's weights (in-order vmcnt).
    auto chunk1 = [&](int stage, int cc, auto&& pf, int t0) {
        constexpr int CPI = R % 2 ? 2 : 1;
        static_assert(BC % CPI == 0, "BC");
        const float* halo = lds + stage * STAGE + aBase;
        const WPos wt = wf + t0 * R * S * tapStride + cc * BC * a.kpad;
        Raw wr[2][S][TN / SV];
        auto loadW = [&](int c, int r, Raw(&w)[S][TN / SV]) {
            const WPos wc = wt + c * a.kpad + r * S * tapStride;
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int q = 0; q < TN / SV; ++q)
                    w[s][q] = wLoad(wc + s * tapStride, q);
        };
        loadW(0, 0, wr[0]);
        pf();
#pragma unroll 1
        for(int c0 = 0; c0 < BC; c0 += CPI)
        {
#pragma unroll
            for(int k = 0; k < CPI * R; ++k)
            {
                const int c = c0 + k / R, r = k % R;
                const int kn = k + 1, cn = c0 + kn / R, rn = kn % R;
                if(cn < BC)
                    loadW(cn, rn, wr[kn % 2]);
                float av[TM + S - 1];
#pragma unroll
                for(int q = 0; q < TM + S - 1; ++q)
                    av[q] = halo[c * LDH + r * b.hw + q];
#pragma unroll
                for(int s = 0; s < S; ++s)
                {
                    float bv[TN];
#pragma unroll
                    for(int q = 0; q < TN / SV; ++q)
                    {
                        const T* e = reinterpret_cast<const T*>(&wr[k % 2][s][q]);
#pragma unroll
                        for(int v = 0; v < SV; ++v)
                            bv[q * SV + v] = K1Elem<T>::load(e[v]);
                    }
#pragma unroll
                    for(int i = 0; i < TM; ++i)
#pragma unroll
                        for(int j = 0; j < TN; ++j)
                            acc[0][i][j] = __builtin_fmaf(av[i + s], bv[j], acc[0][i][j]);
                }
            }
        }
    };
    auto ring = [&](int stage, int cc, int jj, auto U, auto&& pf, auto&&) {
        constexpr int u = decltype(U)::value;
        if constexpr(KD == 1 && PIPE)
        {
            chunk1(stage, cc, pf, 0);
            return;
        }
        pf();
        bool act[KD];
#pragma unroll
        for(int t = 0; t < KD; ++t)
            act[t] = jj - t >= 0 && jj - t < b.dsEff;
        chunk(stage, cc, [&](const float(&av)[TM + S - 1], int s, WPos w0) {
#pragma unroll
            for(int t = 0; t < KD; ++t)
                if(act[t])
                    fmaInto(acc[(u - t + KD) % KD], av, s, w0 + t * R * S * tapStride);
        });
    };
    auto tap = [&](int stage, int cc, int t, auto&& pf, auto&&) {
        if constexpr(PIPE)
        {
            chunk1(stage, cc, pf, t);
        }
        else
        {
            pf();
            const int off = t * R * S * tapStride;
            chunk(stage, cc, [&](const float(&av)[TM + S - 1], int s, WPos w0) {
                fmaInto(acc[0], av, s, w0 + off);
            });
        }
    };

    T* out = static_cast<T*>(a.y);
    auto store = [&](int set, int od, int) {
        const int nb = n0 + tn * TN;
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            const int oh = b.oh0 + ohl, ow = b.ow0 + owl + i;
            if(rowOk && oh < a.ho && ow < a.wo && nb < a.k)
            {
                const long long o
                    = (((static_cast<long long>(b.img) * a.dout + od) * a.ho + oh) * a.wo + ow)
                          * a.k
                      + nb;
                if(nb + TN <= a.k && a.k % SV == 0)
                {
#pragma unroll
                    for(int q = 0; q < TN / SV; ++q)
                    {
                        Raw raw;
                        T* e = reinterpret_cast<T*>(&raw);
#pragma unroll
                        for(int v = 0; v < SV; ++v)
                            e[v] = K1Elem<T>::store(acc[set][i][q * SV + v]);
                        *reinterpret_cast<Raw*>(out + o + q * SV) = raw;
                    }
                }
                else
                {
#pragma unroll
                    for(int j = 0; j < TN; ++j)
                        if(nb + j < a.k)
                            out[o + j] = K1Elem<T>::store(acc[set][i][j]);
                }
            }
#pragma unroll
            for(int j = 0; j < TN; ++j)
                acc[set][i][j] = 0.0f;
        }
    };

    k2Drive<KD, RING>(
        b,
        a.pd,
        a.di,
        (a.c + BC - 1) / BC,
        [&](int plane, int cc) { ld.load(plane, cc * BC); },
        [&](int stage) { ld.template storeChannels<LDH>(lds + stage * STAGE); },
        ring,
        tap,
        store);
}

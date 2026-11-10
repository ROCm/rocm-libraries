// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K8: forward convolution for small input-channel counts (stems: C <= CMAX), any compile-time
// filter, stride and dilation, runtime padding and extents, any activation layout (element
// strides). HIPRTC-compatible: no standard headers. Reuses ConvIgemmCore.h's 16x16 MMA atoms,
// element conversion and buffer loads.
//
// GEMM view: D[pixel][n] = sum_k A[pixel][k] B[k][n] over the flattened reduction
// k = rho * SP + s, with rho = (c, t, r) a filter row (c < CMAX, depth tap t, row r) and s < SP a
// column tap (SP = S rounded up to a power of two; the extra taps have zero weights). With the
// window staged in LDS as [c][t][row][col], the A element of (pixel, k) sits at the pixel's
// window base plus rowOff(rho) + s DW, so a lane's KPL consecutive k are runs of adjacent window
// columns: whole filter rows when KPL >= SP, part of one otherwise. For atoms whose lanes all hold
// the same k (gfx11 WMMA) every offset is a compile-time constant from the lane's pixel base;
// with an even column stride the runs are 4-byte aligned and read by dwords.
//
// Block: WM x WQ x WN waves. Output tile TH = WM FM rows x TW = 16 WQ columns of one output
// plane x BN = 16 FN WN output channels; wave (wm, wq, wn) owns FM rows x 16 columns x 16 FN
// channels. The block stages its input window (zero outside the image and for c >= C) and its
// BN x KP weights in LDS once; the reduction is short (KP = CMAX KD R SP rounded to whole MMA
// steps), so there is no channel loop. Each wave then writes its tile row by row through a
// private LDS slot, in the output's layout.
#pragma once

#include "ConvSmallcArgs.h"

/// Smallest power of two >= V.
template <int V>
struct SmallcPow2
{
    static constexpr int value = V <= 1 ? 1 : 2 * SmallcPow2<(V + 1) / 2>::value;
};

template <class T>
__device__ void smallcFormatWeights(const SmallcFormatArgs& f)
{
    const T* w = static_cast<const T*>(f.w);
    T* wf = static_cast<T*>(f.wf);
    const long long total = static_cast<long long>(f.kpad) * f.kpl;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int kp = static_cast<int>(i % f.kpl);
        const int n = static_cast<int>(i / f.kpl);
        const int s = kp % f.sp;
        int rho = kp / f.sp;
        const int r = rho % f.r;
        rho /= f.r;
        const int t = rho % f.kd;
        const int c = rho / f.kd;
        T v = K1Elem<T>::store(0.0f);
        if(n < f.k && c < f.c && s < f.s)
        {
            const int tt = f.flip ? f.kd - 1 - t : t;
            const int rr = f.flip ? f.r - 1 - r : r;
            const int ss = f.flip ? f.s - 1 - s : s;
            v = w[n * f.wStrides[0] + c * f.wStrides[1] + tt * f.wStrides[2] + rr * f.wStrides[3]
                  + ss * f.wStrides[4]];
        }
        wf[i] = v;
    }
}

template <class T,
          class Atom,
          class TO,
          int CMAX,
          int KD,
          int R,
          int S,
          int SD,
          int SH,
          int SW,
          int DD,
          int DH,
          int DW,
          int FM,
          int FN,
          int WM,
          int WQ,
          int WN>
__device__ __attribute__((always_inline)) void smallcConv(const SmallcArgs& a)
{
    constexpr int WAVE = Atom::WAVE, DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int SP = SmallcPow2<S>::value;
    static_assert(DIM == 16, "16x16 atoms");
    static_assert(KPL >= SP ? KPL % SP == 0 : SP % KPL == 0, "lane runs align with filter rows");
    constexpr int ROWS = CMAX * KD * R; // filter rows rho
    constexpr int NS = (ROWS * SP + KA - 1) / KA; // MMA steps
    constexpr int KP = NS * KA;
    constexpr int TH = WM * FM, TW = WQ * DIM, BN = WN * FN * DIM;
    constexpr int THREADS = WM * WQ * WN * WAVE;
    constexpr int WROWS = (TH - 1) * SH + (R - 1) * DH + 1;
    constexpr int WCOLS = ((TW - 1) * SW + (SP - 1) * DW + 1 + 1) / 2 * 2;
    constexpr int TS = WROWS * WCOLS; // per depth tap
    constexpr int CS = KD * TS; // per channel
    constexpr int WIN = CMAX * CS;
    constexpr int KPS = KP + 16 / static_cast<int>(sizeof(T)); // padded weight row in LDS
    constexpr int SLOT = DIM * FN * DIM; // per-wave epilogue slot, in TO
    constexpr int WIN_BYTES = (WIN * static_cast<int>(sizeof(T)) + 15) / 16 * 16;
    constexpr int W_BYTES = BN * KPS * static_cast<int>(sizeof(T));
    // A lane's taps start 4-byte aligned when the column stride is even (or the element is 4
    // bytes) and the taps are adjacent: gather by dwords.
    constexpr bool VEC32
        = DW == 1 && (sizeof(T) == 4 || SW % 2 == 0) && KPL >= SP && SP * sizeof(T) >= 4;
    static_assert(KP % (16 / static_cast<int>(sizeof(T))) == 0, "weight rows move by 16 bytes");
    typedef typename Atom::frag Frag;
    typedef typename K1Raw<KPL * sizeof(T) >= 16 ? 16 : KPL * sizeof(T)>::type RawB;
    constexpr int BCH = KPL * static_cast<int>(sizeof(T)) / static_cast<int>(sizeof(RawB));

    __shared__ __attribute__((
        aligned(16))) char lds[WIN_BYTES + W_BYTES + WM * WQ * WN * SLOT * sizeof(TO)];
    T* win = reinterpret_cast<T*>(lds);
    T* wl = reinterpret_cast<T*>(lds + WIN_BYTES);

    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wm = wave % WM, wq = (wave / WM) % WQ, wn = wave / (WM * WQ);

    int bx = blockIdx.x;
    const int tq = bx % a.tilesW;
    bx /= a.tilesW;
    const int tp = bx % a.tilesH;
    bx /= a.tilesH;
    const int od = bx % a.dout;
    const int img = bx / a.dout;
    const int oh0 = tp * TH, ow0 = tq * TW;
    const int nb = blockIdx.y * BN;

    // Stage the window: element (c, t, row, col) <- x(img, c, od SD - pd + t DD,
    // oh0 SH - ph + row, ow0 SW - pw + col), zero outside (one buffer over the image, 32-bit
    // offsets; the planner keeps an image below 2 GiB).
    {
        const T* x = static_cast<const T*>(a.x) + img * a.xs[0];
        const long long span = (a.c - 1) * a.xs[1] + (a.di - 1) * a.xs[2] + (a.hi - 1) * a.xs[3]
                               + (a.wi - 1) * a.xs[4] + 1;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(x, span * static_cast<long long>(sizeof(T)));
        const int s1 = static_cast<int>(a.xs[1]), s2 = static_cast<int>(a.xs[2]);
        const int s3 = static_cast<int>(a.xs[3]), s4 = static_cast<int>(a.xs[4]);
        const int id0 = od * SD - a.pd, ih0 = oh0 * SH - a.ph, iw0 = ow0 * SW - a.pw;
        typedef typename K1Raw<sizeof(T)>::type RawE;
        constexpr int LSTEP = THREADS / WCOLS, CSTEP = THREADS % WCOLS;
        int col = tid % WCOLS, line = tid / WCOLS;
        for(int e = tid; e < WIN; e += THREADS)
        {
            const int row = line % WROWS;
            const int ct = line / WROWS;
            const int t = ct % KD;
            const int c = ct / KD;
            const int id = id0 + t * DD, ih = ih0 + row, iw = iw0 + col;
            const bool ok = (c < a.c) & (static_cast<unsigned>(id) < static_cast<unsigned>(a.di))
                            & (static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi));
            const unsigned off = ok ? static_cast<unsigned>((c * s1 + id * s2 + ih * s3 + iw * s4)
                                                            * static_cast<int>(sizeof(T)))
                                    : K1_OUT_OF_RANGE;
            win[e] = __builtin_bit_cast(T, k1BufferLoad<RawE>(xr, off));
            col += CSTEP;
            line += LSTEP;
            if(col >= WCOLS)
            {
                col -= WCOLS;
                ++line;
            }
        }
        // Weights: BN formatted rows of KP.
        typedef typename K1Raw<16>::type Raw16;
        constexpr int V = 16 / static_cast<int>(sizeof(T));
        const T* wf = static_cast<const T*>(a.w) + static_cast<long long>(nb) * KP;
        for(int e = tid; e < BN * (KP / V); e += THREADS)
        {
            const int n = e / (KP / V), kv = (e % (KP / V)) * V;
            *reinterpret_cast<Raw16*>(wl + n * KPS + kv)
                = *reinterpret_cast<const Raw16*>(wf + n * KP + kv);
        }
    }
    __syncthreads();

    typename Atom::acc acc[FM][FN];
#pragma unroll
    for(int i = 0; i < FM; ++i)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[i][j] = typename Atom::acc{};

    const int kb = Atom::template kBase<KPL>(lane);
    // Window base of fragment row (lane % 16): output row wm FM of the tile, column
    // wq 16 + lane % 16.
    const int pbase = (wm * FM * SH) * WCOLS + (wq * DIM + lane % DIM) * SW;
    // Window offset of filter row rho (padding rows: zero weights, any in-window element).
    auto rowOff = [](int rho) {
        if(rho >= ROWS)
            return 0;
        const int r = rho % R;
        const int ct = rho / R;
        const int t = ct % KD;
        const int c = ct / KD;
        return c * CS + t * TS + r * DH * WCOLS;
    };
#pragma unroll
    for(int st = 0; st < NS; ++st)
    {
        // A fresh (opaque) pixel base per step: the window offsets are compile-time constants, so
        // the compiler would otherwise merge the steps' identical fragment loads and keep them
        // live across the whole unrolled reduction (spilling).
        int pb = pbase;
        asm volatile("" : "+v"(pb));
        __builtin_amdgcn_sched_barrier(0);
        // B fragments: column n, KPL consecutive k from st KA + kb.
        Frag bf[FN];
#pragma unroll
        for(int j = 0; j < FN; ++j)
        {
            const T* p = wl + (wn * FN * DIM + j * DIM + lane % DIM) * KPS + st * KA + kb;
            RawB raw[BCH];
#pragma unroll
            for(int q = 0; q < BCH; ++q)
                raw[q] = reinterpret_cast<const RawB*>(p)[q];
            bf[j] = __builtin_bit_cast(Frag, raw);
        }
        const int k0 = st * KA + kb;
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
            Frag af;
            if constexpr(KPL >= SP && VEC32)
            {
                // KPL / SP whole filter rows, each SP adjacent columns read as dwords straight
                // into the fragment's registers.
                constexpr int DPR = SP * static_cast<int>(sizeof(T)) / 4; // dwords per row
                unsigned dv[KPL / SP * DPR];
#pragma unroll
                for(int jr = 0; jr < KPL / SP; ++jr)
                {
                    const unsigned* src = reinterpret_cast<const unsigned*>(
                        win + pb + i * SH * WCOLS + rowOff(k0 / SP + jr));
#pragma unroll
                    for(int q = 0; q < DPR; ++q)
                        dv[jr * DPR + q] = src[q];
                }
                af = __builtin_bit_cast(Frag, dv);
            }
            else
            {
                T av[KPL];
                if constexpr(KPL >= SP)
                {
#pragma unroll
                    for(int jr = 0; jr < KPL / SP; ++jr)
                    {
                        const T* src = win + pb + i * SH * WCOLS + rowOff(k0 / SP + jr);
#pragma unroll
                        for(int s = 0; s < SP; ++s)
                            av[jr * SP + s] = src[s * DW];
                    }
                }
                else
                {
                    // Part of one filter row: taps k0 % SP .. + KPL.
                    const T* src = win + pb + i * SH * WCOLS + rowOff(k0 / SP) + (k0 % SP) * DW;
#pragma unroll
                    for(int e = 0; e < KPL; ++e)
                        av[e] = src[e * DW];
                }
                af = __builtin_bit_cast(Frag, av);
            }
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[i][j] = Atom::mma(af, bf[j], acc[i][j]);
            // One A fragment live at a time.
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    // Epilogue, per wave: each output row of the wave's fragments goes through the wave's own
    // LDS slot as [n][16 columns] in TO; lanes then store it in the output's layout (columns
    // fastest when the output's width stride is 1, channels fastest otherwise).
    TO* slot = reinterpret_cast<TO*>(lds + WIN_BYTES + W_BYTES) + wave * SLOT;
    TO* y = static_cast<TO*>(a.y) + img * a.ys[0] + od * a.ys[2];
    const bool colFast = a.ys[4] == 1;
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
#pragma unroll
        for(int j = 0; j < FN; ++j)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
                slot[(j * DIM + lane % DIM) * DIM + Atom::dRow(lane, e)]
                    = K1Elem<TO>::store(acc[i][j][e]);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        const int oh = oh0 + wm * FM + i;
#pragma unroll
        for(int it = 0; it < SLOT / WAVE; ++it)
        {
            const int e = lane + it * WAVE;
            int n, px;
            if(colFast)
            {
                px = e % DIM;
                n = e / DIM;
            }
            else
            {
                n = e % (FN * DIM);
                px = e / (FN * DIM);
            }
            const int ow = ow0 + wq * DIM + px, kk = nb + wn * FN * DIM + n;
            if(oh < a.ho && ow < a.wo && kk < a.k)
                y[kk * a.ys[1] + oh * a.ys[3] + ow * a.ys[4]] = slot[n * DIM + px];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
}

// ---------------------------------------------------------------------------------------------
// FMA kernel (fp32 where no fp32 MMA exists: gfx11/gfx12, and any dtype anywhere). 256 threads;
// output channels in NG = BN / FN groups of FN, pixels in runs of FM adjacent columns: thread t
// owns channel group t % NG and pixel run t / NG of a TH = WM row x TW = 16 WQ column tile. The
// window is staged as in the MMA kernel; the weights go to LDS transposed ([kp][n]) so a thread
// reads its FN weights of one k as vectors. Accumulation in fp32.
template <class T,
          class TO,
          int CMAX,
          int KD,
          int R,
          int S,
          int SD,
          int SH,
          int SW,
          int DD,
          int DH,
          int DW,
          int FM,
          int FN,
          int WM,
          int WQ,
          int WN>
__device__ __attribute__((always_inline)) void smallcConvFma(const SmallcArgs& a)
{
    constexpr int THREADS = 256;
    constexpr int SP = SmallcPow2<S>::value;
    constexpr int ROWS = CMAX * KD * R;
    constexpr int KP = ROWS * SP; // the format's row length (ka 1)
    constexpr int TH = WM, TW = WQ * 16, BN = WN * 16;
    constexpr int NG = BN / FN;
    constexpr int RUNS = THREADS / NG; // pixel runs
    static_assert(BN % FN == 0 && THREADS % NG == 0 && RUNS * FM == TH * TW && TW % FM == 0,
                  "the threads cover the tile exactly");
    constexpr int WROWS = (TH - 1) * SH + (R - 1) * DH + 1;
    constexpr int WCOLS = ((TW - 1) * SW + (SP - 1) * DW + 1 + 1) / 2 * 2;
    constexpr int TS = WROWS * WCOLS;
    constexpr int CS = KD * TS;
    constexpr int WIN = CMAX * CS;
    constexpr int WIN_BYTES = (WIN * static_cast<int>(sizeof(T)) + 15) / 16 * 16;
    constexpr int BNS = BN + 4; // padded weight row ([kp][n]), in float
    __shared__ __attribute__((aligned(16))) char lds[WIN_BYTES + KP * BNS * sizeof(float)];
    T* win = reinterpret_cast<T*>(lds);
    float* wl = reinterpret_cast<float*>(lds + WIN_BYTES);

    const int tid = threadIdx.x;
    int bx = blockIdx.x;
    const int tq = bx % a.tilesW;
    bx /= a.tilesW;
    const int tp = bx % a.tilesH;
    bx /= a.tilesH;
    const int od = bx % a.dout;
    const int img = bx / a.dout;
    const int oh0 = tp * TH, ow0 = tq * TW;
    const int nb = blockIdx.y * BN;
    {
        const T* x = static_cast<const T*>(a.x) + img * a.xs[0];
        const long long span = (a.c - 1) * a.xs[1] + (a.di - 1) * a.xs[2] + (a.hi - 1) * a.xs[3]
                               + (a.wi - 1) * a.xs[4] + 1;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(x, span * static_cast<long long>(sizeof(T)));
        const int s1 = static_cast<int>(a.xs[1]), s2 = static_cast<int>(a.xs[2]);
        const int s3 = static_cast<int>(a.xs[3]), s4 = static_cast<int>(a.xs[4]);
        const int id0 = od * SD - a.pd, ih0 = oh0 * SH - a.ph, iw0 = ow0 * SW - a.pw;
        typedef typename K1Raw<sizeof(T)>::type RawE;
        constexpr int LSTEP = THREADS / WCOLS, CSTEP = THREADS % WCOLS;
        int col = tid % WCOLS, line = tid / WCOLS;
        for(int e = tid; e < WIN; e += THREADS)
        {
            const int row = line % WROWS;
            const int ct = line / WROWS;
            const int t = ct % KD;
            const int c = ct / KD;
            const int id = id0 + t * DD, ih = ih0 + row, iw = iw0 + col;
            const bool ok = (c < a.c) & (static_cast<unsigned>(id) < static_cast<unsigned>(a.di))
                            & (static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi));
            const unsigned off = ok ? static_cast<unsigned>((c * s1 + id * s2 + ih * s3 + iw * s4)
                                                            * static_cast<int>(sizeof(T)))
                                    : K1_OUT_OF_RANGE;
            win[e] = __builtin_bit_cast(T, k1BufferLoad<RawE>(xr, off));
            col += CSTEP;
            line += LSTEP;
            if(col >= WCOLS)
            {
                col -= WCOLS;
                ++line;
            }
        }
        const T* wf = static_cast<const T*>(a.w) + static_cast<long long>(nb) * KP;
        for(int e = tid; e < BN * KP; e += THREADS)
        {
            const int n = e / KP, kp = e % KP;
            wl[kp * BNS + n] = K1Elem<T>::load(wf[e]);
        }
    }
    __syncthreads();

    const int g = tid % NG, run = tid / NG;
    const int prow = run / (TW / FM), pcol = (run % (TW / FM)) * FM;
    float acc[FM][FN];
#pragma unroll
    for(int p = 0; p < FM; ++p)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[p][j] = 0.0f;
    const T* xb = win + prow * SH * WCOLS + pcol * SW;
    const float* wb = wl + g * FN;
    for(int rho = 0; rho < ROWS; ++rho)
    {
        const int r = rho % R, ct = rho / R;
        const int t = ct % KD, c = ct / KD;
        const T* xr = xb + c * CS + t * TS + r * DH * WCOLS;
#pragma unroll
        for(int s = 0; s < S; ++s)
        {
            float wv[FN];
            const float* wp = wb + (rho * SP + s) * BNS;
#pragma unroll
            for(int j = 0; j < FN; ++j)
                wv[j] = wp[j];
            float xv[FM];
#pragma unroll
            for(int p = 0; p < FM; ++p)
                xv[p] = K1Elem<T>::load(xr[p * SW + s * DW]);
#pragma unroll
            for(int p = 0; p < FM; ++p)
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    acc[p][j] = __builtin_fmaf(xv[p], wv[j], acc[p][j]);
        }
    }

    TO* y = static_cast<TO*>(a.y) + img * a.ys[0] + od * a.ys[2];
    const int oh = oh0 + prow;
    if(oh >= a.ho)
        return;
#pragma unroll
    for(int j = 0; j < FN; ++j)
    {
        const int kk = nb + g * FN + j;
        if(kk >= a.k)
            continue;
#pragma unroll
        for(int p = 0; p < FM; ++p)
        {
            const int ow = ow0 + pcol + p;
            if(ow < a.wo)
                y[kk * a.ys[1] + oh * a.ys[3] + ow * a.ys[4]] = K1Elem<TO>::store(acc[p][j]);
        }
    }
}

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
// private LDS slot, in the output's layout (EP 0), or straight from registers (EP 1: the MMA
// computes the transposed tile, so a lane holds adjacent output channels of one pixel; for
// channels-fastest outputs, where the stores are 16-byte channel runs).
//
// Backward data at unit stride (CONV_SMALLC_BWD) runs on the same kernel: the forward
// convolution over dy with the filter transposed and reflected by the weight pre-pass.
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
        int kp, n;
        if(f.bn == 0)
        {
            kp = static_cast<int>(i % f.kpl);
            n = static_cast<int>(i / f.kpl);
        }
        else
        {
            const int nl = static_cast<int>(i % f.bn);
            const long long rest = i / f.bn;
            kp = static_cast<int>(rest % f.kpl);
            n = static_cast<int>(rest / f.kpl) * f.bn + nl;
        }
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
          int WN,
          int EP = 0>
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
    // Register epilogue (EP 1): a lane's accumulators are channels dRow(lane, e) of fragment j's
    // B rows. Fragment rows read the weight rows (output channels) permuted so that a lane's
    // stored run is 8 adjacent channels (one 16-byte store of 16-bit output): 16-bit MFMA pairs
    // fragments (2 x 4 rows); gfx11 WMMA un-weaves its interleaved rows (2e + half -> 8 half +
    // e; its 2-byte scalar stores ran 2-4x slower). gfx12 WMMA rows and fp32 MFMA rows already
    // are such runs.
    constexpr bool W11 = __is_base_of(K1Wmma11, Atom);
    constexpr bool PAIRS = EP != 0 && !W11 && Atom::ACC == 4 && sizeof(TO) == 2 && FN % 2 == 0;
    constexpr bool UNWEAVE = EP != 0 && W11;
    auto bRow = [](int j, int m) {
        if constexpr(PAIRS)
            return (j / 2) * 32 + 8 * (m >> 2) + 4 * (j & 1) + (m & 3);
        else if constexpr(UNWEAVE)
            return j * 16 + 8 * (m & 1) + (m >> 1);
        else
            return j * 16 + m;
    };
    constexpr int BCH = KPL * static_cast<int>(sizeof(T)) / static_cast<int>(sizeof(RawB));

    // EP 1 stores from registers: no epilogue slots.
    __shared__ __attribute__((aligned(
        16))) char lds[WIN_BYTES + W_BYTES + (EP == 0 ? WM * WQ * WN * SLOT * sizeof(TO) : 0)];
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
            const T* p = wl + (wn * FN * DIM + bRow(j, lane % DIM)) * KPS + st * KA + kb;
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
            if constexpr(EP == 0)
            {
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    acc[i][j] = Atom::mma(af, bf[j], acc[i][j]);
            }
            else
            {
                // Transposed tile: D[channel][pixel], so a lane's accumulators are output
                // channels of one pixel (the operands share the lane layout).
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    acc[i][j] = Atom::mma(bf[j], af, acc[i][j]);
            }
            // One A fragment live at a time.
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    if constexpr(EP != 0)
    {
        // Straight from registers: lane holds channels bRow(j, dRow(lane, e)) of pixel
        // lane % 16. Where a store group's channels are adjacent (VW of them over G fragments)
        // and the output is channels-fastest and aligned, one vector store per group.
        constexpr int ACC = Atom::ACC;
        constexpr int G = PAIRS ? 2 : 1;
        constexpr int VW = PAIRS || UNWEAVE ? 8 : ACC;
        constexpr bool RUN = !W11 || UNWEAVE;
        typedef TO VO __attribute__((ext_vector_type(VW)));
        TO* y = static_cast<TO*>(a.y) + img * a.ys[0] + od * a.ys[2];
        const bool vec = RUN && a.ys[1] == 1 && a.ys[3] % VW == 0 && a.ys[4] % VW == 0
                         && (reinterpret_cast<unsigned long long>(y) % (VW * sizeof(TO))) == 0;
        const int ow = ow0 + wq * DIM + lane % DIM;
        const int kw = nb + wn * FN * DIM;
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
            const int oh = oh0 + wm * FM + i;
            if(oh >= a.ho || ow >= a.wo)
                continue;
            TO* yp = y + oh * a.ys[3] + ow * a.ys[4];
#pragma unroll
            for(int gs = 0; gs < FN / G; ++gs)
            {
                const int k0 = kw + bRow(gs * G, Atom::dRow(lane, 0));
                if(vec && k0 + VW <= a.k)
                {
                    VO v;
#pragma unroll
                    for(int t = 0; t < VW; ++t)
                        v[t] = K1Elem<TO>::store(acc[i][gs * G + t / ACC][t % ACC]);
                    *reinterpret_cast<VO*>(yp + k0) = v;
                }
                else
                {
#pragma unroll
                    for(int t = 0; t < G * ACC; ++t)
                    {
                        const int j = gs * G + t / ACC;
                        const int kk = kw + bRow(j, Atom::dRow(lane, t % ACC));
                        if(kk < a.k)
                            yp[kk * a.ys[1]] = K1Elem<TO>::store(acc[i][j][t % ACC]);
                    }
                }
            }
        }
        return;
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
        // The pre-pass wrote this block's rows by k ([kp][BN]): copy them as they lie.
        const T* wf = static_cast<const T*>(a.w) + static_cast<long long>(nb) * KP;
        if constexpr(sizeof(T) == 4)
        {
            typedef typename K1Raw<16>::type Raw16;
            for(int e = tid; e < BN * KP / 4; e += THREADS)
            {
                const int kp = e / (BN / 4), n = (e % (BN / 4)) * 4;
                *reinterpret_cast<Raw16*>(wl + kp * BNS + n)
                    = reinterpret_cast<const Raw16*>(wf)[e];
            }
        }
        else
        {
            for(int e = tid; e < BN * KP; e += THREADS)
            {
                const int kp = e / BN, n = e % BN;
                wl[kp * BNS + n] = K1Elem<T>::load(wf[e]);
            }
        }
    }
    __syncthreads();

    // Channel group per run of RUNS threads (a whole wave or more when RUNS >= the wave), so a
    // wave's weight reads are one address (an LDS broadcast).
    const int g = tid / RUNS, run = tid % RUNS;
    const int prow = run / (TW / FM), pcol = (run % (TW / FM)) * FM;
    float acc[FM][FN];
#pragma unroll
    for(int p = 0; p < FM; ++p)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[p][j] = 0.0f;
    const T* xb = win + prow * SH * WCOLS + pcol * SW;
    const float* wb = wl + g * FN;
    // A filter row's taps for FM adjacent outputs span XS window columns: read them once.
    constexpr int XS = (FM - 1) * SW + (S - 1) * DW + 1;
    for(int rho = 0; rho < ROWS; ++rho)
    {
        const int r = rho % R, ct = rho / R;
        const int t = ct % KD, c = ct / KD;
        const T* xr = xb + c * CS + t * TS + r * DH * WCOLS;
        float xv[XS];
#pragma unroll
        for(int q = 0; q < XS; ++q)
            xv[q] = K1Elem<T>::load(xr[q]);
#pragma unroll
        for(int s = 0; s < S; ++s)
        {
            float wv[FN];
            const float* wp = wb + (rho * SP + s) * BNS;
#pragma unroll
            for(int j = 0; j < FN; ++j)
                wv[j] = wp[j];
#pragma unroll
            for(int p = 0; p < FM; ++p)
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    acc[p][j] = __builtin_fmaf(xv[p * SW + s * DW], wv[j], acc[p][j]);
        }
    }

    TO* y = static_cast<TO*>(a.y) + img * a.ys[0] + od * a.ys[2];
    const int oh = oh0 + prow;
    if(oh >= a.ho)
        return;
    const int k0 = nb + g * FN, ow0t = ow0 + pcol;
    TO* yr = y + oh * a.ys[3];
    // Vector stores where the layout makes a thread's outputs adjacent: its FN channels of one
    // pixel (channels fastest) or its FM pixels of one channel (columns fastest).
    if(a.ys[1] == 1 && k0 + FN <= a.k && (a.k % FN) == 0 && ((a.ys[4] % FN) == 0))
    {
        typedef TO VF __attribute__((ext_vector_type(FN)));
#pragma unroll
        for(int p = 0; p < FM; ++p)
        {
            const int ow = ow0t + p;
            if(ow < a.wo)
            {
                VF v;
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    v[j] = K1Elem<TO>::store(acc[p][j]);
                *reinterpret_cast<VF*>(yr + ow * a.ys[4] + k0) = v;
            }
        }
        return;
    }
    if(a.ys[4] == 1 && ow0t + FM <= a.wo && ((a.ys[3] % FM) == 0) && ((a.ys[1] % FM) == 0)
       && ((ow0t % FM) == 0))
    {
        typedef TO VM __attribute__((ext_vector_type(FM)));
#pragma unroll
        for(int j = 0; j < FN; ++j)
        {
            const int kk = k0 + j;
            if(kk < a.k)
            {
                VM v;
#pragma unroll
                for(int p = 0; p < FM; ++p)
                    v[p] = K1Elem<TO>::store(acc[p][j]);
                *reinterpret_cast<VM*>(yr + kk * a.ys[1] + ow0t) = v;
            }
        }
        return;
    }
#pragma unroll
    for(int j = 0; j < FN; ++j)
    {
        const int kk = k0 + j;
        if(kk >= a.k)
            continue;
#pragma unroll
        for(int p = 0; p < FM; ++p)
        {
            const int ow = ow0t + p;
            if(ow < a.wo)
                yr[kk * a.ys[1] + ow * a.ys[4]] = K1Elem<TO>::store(acc[p][j]);
        }
    }
}

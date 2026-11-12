// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K2 "rows" kernel: stride-1 2-D and 3-D direct convolution with filter-row reuse, channels-last, for
// the CDNA 16x16 matrix cores (gfx90a to gfx950), the halo filled by LDS DMA: 16 bytes a lane on
// gfx950, a dword a lane elsewhere (K2R_DMA16). Included by ConvDirect.cpp after ConvDirectCore.h
// (atoms, K2Args, k2FilterOffset, k2SplitBf16, k2StaticFor). HIPRTC-compatible.
//
// D = W x X^T per 16x16 MFMA atom: the weights are the M operand (output channels) and the
// input the N operand (16 output columns of one row). A wave owns PW output rows x 16 columns x
// FN * 16 output channels. One step is one filter column s and one KA slice of the channel
// chunk: it loads the R x FN weight fragments (pre-formatted, read through L1), the PW + R - 1
// input row fragments from the LDS halo, and runs R x FN x PW MMAs. Input row p + r feeds output
// row p through filter row r, so every input fragment serves R filter rows.
//
// Block: WP x WQ x WK waves, BP = WP PW rows x BQ = 16 WQ columns x BN = 16 FN WK channels.
// Halo stage: (BP + R - 1) x (BQ + S - 1) pixels x BC channels in 16-byte granules; granule g of
// column x sits at slot x G + (g + x) mod G of its row (conflict-free 16-byte reads). Filled by
// LDS DMA (buffer_load ... lds, 16 B per lane): DMA instruction (h, xg) writes 64 consecutive
// granules of row h, each lane fetching the global granule its slot holds. Without 16-byte DMA
// the same 64 granules take K2R_DMA_SUB = 4 dword instructions, the j-th writing granules
// 16 j .. 16 j + 15 (lane l: granule 16 j + l / 4, dword l % 4) from the same fetch addresses.
// Two stages, separate LDS objects, so the compiler tracks each DMA against its own stage.
//
// Tiles: blocks walk tiles (channel block fastest, then column tile, row tile, image) in an
// XCD-contiguous order. With fewer blocks than tiles (the planner launches one block per CU),
// the last chunk of a tile prefetches chunk 0 of the block's next tile.
//
// PP 0: one barrier per chunk; the two waves per SIMD interleave by themselves. PP 1: the WK = 2
// channel halves are two wave groups one barrier apart; each step is "memory phase | barrier |
// MMAs at priority 1 | barrier". DMA goes out in steps 0..NQ-2; each wave drains it (vmcnt 0) in
// step NQ-1's memory phase, before the barrier after which anyone may read the stage.
// EPI 1 (16-bit output, PP 0): the epilogue transposes through LDS so 8 lanes store one pixel's
// 16 FN channels as 16-byte pieces; EPI 0 stores each lane's 4 channels directly.
//
// 3-D (KD > 1): a tile is one output plane's rows x columns x channels; its chunks walk the depth
// taps whose input plane lies in the image (each staged as a 2-D halo, re-read through L2 by the
// neighbouring planes), each tap's R filter rows as in 2-D.
//
// Envelope: stride 1, dilation 1, groups 1, KD 1 (2-D) or KD x R x S (3-D), any padding;
// reduction channels a multiple of 16 / sizeof(T); per-image input < 2 GiB; formatted weights
// < 2 GiB.
#pragma once

// 16-byte LDS DMA (buffer_load_dwordx4 ... lds) is a CDNA4 feature; gfx950 is CDNA4's only
// member, so its chip macro is the family test. The other MFMA families (gfx90a, gfx94x, and a
// gfx9-4-generic build) have dword LDS DMA only and take the K2R_DMA_SUB = 4 fill.
#if defined(__gfx950__)
#define K2R_DMA16 1
#else
#define K2R_DMA16 0
#endif
#define K2R_DMA_SUB (K2R_DMA16 ? 1 : 4) // DMA instructions per 64-granule run

#define K2R_VMCNT0 0x0F70 // s_waitcnt vmcnt(0), lgkmcnt and expcnt untouched (gfx9 encoding)
#define K2R_LGKMCNT0 0xC07F // s_waitcnt lgkmcnt(0), vmcnt and expcnt untouched

template <class Raw>
__device__ inline Raw k2rBufferLoad(__amdgpu_buffer_rsrc_t r, unsigned offset, int soffset)
{
    if constexpr(sizeof(Raw) == 16)
        return __builtin_amdgcn_raw_buffer_load_b128(r, offset, soffset, 0);
    else if constexpr(sizeof(Raw) == 8)
        return __builtin_amdgcn_raw_buffer_load_b64(r, offset, soffset, 0);
    else
        return __builtin_amdgcn_raw_buffer_load_b32(r, offset, soffset, 0);
}

/// s_barrier without the fences of __syncthreads: the in-flight LDS DMA of the next stage must
/// stay in flight. Memory clobbers keep the compiler from moving LDS accesses across it.
__device__ inline void k2rBarrier()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

/// Four fp32 accumulators as four TO. gfx950 rounds bf16 pairs in hardware (v_cvt_pk_bf16_f32):
/// bit-identical to k1FloatToBf for every non-NaN input (checked exhaustively), and NaNs stay
/// quiet NaNs; one instruction per pair instead of the software round's six or so per value.
template <class TO, class V>
__device__ __attribute__((always_inline)) inline TO __attribute__((ext_vector_type(4)))
k2rPack4(const V& v)
{
    typedef TO v4 __attribute__((ext_vector_type(4)));
#if defined(__gfx950__)
    if constexpr(__is_same(TO, k1_bf16))
    {
        typedef float f2 __attribute__((ext_vector_type(2)));
        typedef __bf16 b2 __attribute__((ext_vector_type(2)));
        typedef unsigned u2 __attribute__((ext_vector_type(2)));
        const b2 lo = __builtin_convertvector((f2){v[0], v[1]}, b2);
        const b2 hi = __builtin_convertvector((f2){v[2], v[3]}, b2);
        return __builtin_bit_cast(
            v4, (u2){__builtin_bit_cast(unsigned, lo), __builtin_bit_cast(unsigned, hi)});
    }
#endif
    v4 o;
#pragma unroll
    for(int i = 0; i < 4; ++i)
        o[i] = K1Elem<TO>::store(v[i]);
    return o;
}

/// Weight pre-pass of the rows kernel. 16 x KA fragments are stored lane-major for the 16x16 MFMA
/// lane map (lane l: column l % 16, slice offset (l / 16) KPL), ordered so one wave's step (FN
/// column tiles x R filter rows of one filter column s and one KA slice kc) is one contiguous run:
/// wf[(((((g ca + kc) S + s) R + r) FN + j) 64 + l) KPL + e], output-column group g = n / (16 FN),
/// tile j = (n / 16) % FN, kc over cpad / ka. Roles (backward data, flip) as k2FormatWeights;
/// MATH 1 writes the three-bf16 form ([hi | lo | hi] of each weight along a 3x reduction).
template <class T, int KPL, int FN, int MATH>
__device__ void k2rFormatWeightsElementwise(const K2FormatArgs& f)
{
    T* wf = static_cast<T*>(f.wf);
    const long long total = static_cast<long long>(f.kd) * f.r * f.s * f.cpad * f.kpad;
    const int cas = f.cpad / f.ka;
    const int cred = f.backward ? f.k0 : f.c0;
    const bool mirror = (f.backward != 0) != (f.flip != 0);
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int e = static_cast<int>(i % KPL);
        long long rest = i / KPL;
        const int l = static_cast<int>(rest % 64);
        rest /= 64;
        const int j = static_cast<int>(rest % FN);
        rest /= FN;
        const int r = static_cast<int>(rest % (f.kd * f.r));
        rest /= f.kd * f.r;
        const int s = static_cast<int>(rest % f.s);
        rest /= f.s;
        const int kc = static_cast<int>(rest % cas);
        const int g = static_cast<int>(rest / cas);
        const int tap = r * f.s + s;
        const int n = (g * FN + j) * 16 + l % 16;
        const int c = kc * f.ka + (l / 16) * KPL + e;
        if constexpr(MATH)
        {
            unsigned short out = 0;
            if(c < 3 * cred)
            {
                const int part = c / cred, cc = c - part * cred;
                const bool ok = f.backward ? n < f.c0 : n < f.k0;
                if(ok)
                {
                    const float* w = static_cast<const float*>(f.w);
                    const float v = f.backward ? w[k2FilterOffset(f, cc, n, tap, mirror)]
                                               : w[k2FilterOffset(f, n, cc, tap, mirror)];
                    unsigned short hi, lo;
                    k2SplitBf16(v, hi, lo);
                    out = part == 1 ? lo : part == 2 ? k1BfFinite(hi) : hi; // part 2: with x lo
                }
            }
            reinterpret_cast<unsigned short*>(wf)[i] = out;
        }
        else
        {
            const T* w = static_cast<const T*>(f.w);
            T v = K1Elem<T>::store(0.0f);
            if(!f.backward)
            {
                if(n < f.k0 && c < f.c0)
                    v = w[k2FilterOffset(f, n, c, tap, mirror)];
            }
            else
            {
                if(n < f.c0 && c < f.k0)
                    v = w[k2FilterOffset(f, c, n, tap, mirror)];
            }
            wf[i] = v;
        }
    }
}

/// The rows weight pre-pass, one 16 FN x KA tile (output columns x reduction slice) of one tap
/// per step: every (g, kc, s, r) run of the formatted layout is one such tile, contiguous. The
/// block gathers the tile from the filter with its unit-stride channel fastest across lanes (the
/// forward filter's c for the forward pass, its k for backward data, where the reduction and
/// output roles swap), transposes it through LDS and stores the run contiguously. Offsets are
/// per-tile bases plus multiplies: no per-element index decode. Layouts other than the 64-lane
/// 16 x 16 fragment (ka != 4 KPL) and the three-bf16 form take the element-wise loop.
template <class T, int KPL, int FN, int MATH>
__device__ void k2rFormatWeights(const K2FormatArgs& f)
{
    constexpr int KA = 4 * KPL; // 64 lanes: 16 columns x 4 slices of KPL
    constexpr int TN = 16 * FN;
    constexpr int TILE = TN * KA;
    if constexpr(MATH != 0)
    {
        k2rFormatWeightsElementwise<T, KPL, FN, MATH>(f);
        return;
    }
    else
    {
        if(f.ka != KA)
        {
            k2rFormatWeightsElementwise<T, KPL, FN, MATH>(f);
            return;
        }
        // Rows padded by one KPL run: the store phase reads whole KPL runs (aligned vectors).
        __shared__ __attribute__((aligned(16))) T tile[TN][KA + KPL];
        typedef typename K1Raw<KPL * sizeof(T)>::type Run;
        T* wf = static_cast<T*>(f.wf);
        const T* w = static_cast<const T*>(f.w);
        const int rk = f.kd * f.r;
        const int cas = f.cpad / KA;
        const int groups = f.kpad / TN;
        const int tiles = groups * cas * f.s * rk;
        const bool mirror = (f.backward != 0) != (f.flip != 0);
        // Output column n and reduction channel c of the forward filter K C T R S: forward
        // n = k, c = c; backward data n = c, c = k.
        const long long strideN = f.backward ? f.wStrides[1] : f.wStrides[0];
        const long long strideC = f.backward ? f.wStrides[0] : f.wStrides[1];
        const int limN = f.backward ? f.c0 : f.k0;
        const int limC = f.backward ? f.k0 : f.c0;
        // Gather order: lanes along the smaller-stride channel; VL-element vectors along it where
        // it is unit-stride, in range and aligned.
        const bool nFast = strideN < strideC;
        constexpr int VL = 16 / static_cast<int>(sizeof(T));
        typedef typename K1Raw<16>::type Vec;
        const long long fastStride = nFast ? strideN : strideC;
        const int tid = static_cast<int>(threadIdx.x);
        const int threads = static_cast<int>(blockDim.x);
        for(int t = static_cast<int>(blockIdx.x); t < tiles; t += static_cast<int>(gridDim.x))
        {
            int rest = t;
            const int r = rest % rk;
            rest /= rk;
            const int sc = rest % f.s;
            rest /= f.s;
            const int kc = rest % cas;
            const int g = rest / cas;
            // Tap (t, r, s) of r = t R + r', reflected for backward data or a flipped filter.
            int tt = r / f.r, rr = r - tt * f.r, ss = sc;
            if(mirror)
            {
                tt = f.kd - 1 - tt;
                rr = f.r - 1 - rr;
                ss = f.s - 1 - ss;
            }
            const long long tapOff = tt * f.wStrides[2] + rr * f.wStrides[3] + ss * f.wStrides[4];
            const int n0 = g * TN, c0 = kc * KA;
            static_assert(TN % VL == 0 && KA % VL == 0, "rows pre-pass vectors");
            for(int u = tid; u < TILE / VL; u += threads)
            {
                // VL consecutive elements along the fast channel from (nl, cl).
                const int nl = nFast ? (u % (TN / VL)) * VL : u / (KA / VL);
                const int cl = nFast ? u / (TN / VL) : (u % (KA / VL)) * VL;
                const int n = n0 + nl, c = c0 + cl;
                const long long off = n * strideN + c * strideC + tapOff;
                T v[VL];
                const bool whole
                    = fastStride == 1 && off % VL == 0
                      && (nFast ? n + VL <= limN && c < limC : c + VL <= limC && n < limN);
                if(whole)
                {
                    *reinterpret_cast<Vec*>(v) = *reinterpret_cast<const Vec*>(w + off);
                }
                else
                {
#pragma unroll
                    for(int e = 0; e < VL; ++e)
                    {
                        const int ne = nFast ? n + e : n, ce = nFast ? c : c + e;
                        v[e] = ne < limN && ce < limC ? w[ne * strideN + ce * strideC + tapOff]
                                                      : K1Elem<T>::store(0.0f);
                    }
                }
#pragma unroll
                for(int e = 0; e < VL; ++e)
                    tile[nFast ? nl + e : nl][nFast ? cl : cl + e] = v[e];
            }
            __syncthreads();
            // The run: index (j 64 + l) KPL + e holds column j 16 + l % 16, channel (l / 16) KPL + e;
            // each (j, l) is KPL consecutive channels of one LDS row, stored as one vector.
            Run* dst = reinterpret_cast<Run*>(wf + static_cast<long long>(t) * TILE);
            for(int u = tid; u < TILE / KPL; u += threads)
            {
                const int l = u % 64;
                const int j = u / 64;
                dst[u] = *reinterpret_cast<const Run*>(&tile[j * 16 + l % 16][(l / 16) * KPL]);
            }
            __syncthreads();
        }
    }
}

template <class T,
          class Atom,
          int R,
          int S,
          int PW,
          int FN,
          int WP,
          int WQ,
          int WK,
          int BC,
          int PP,
          int EPI,
          class TO = T>
__device__ __attribute__((always_inline)) void k2ConvRows(const K2Args& a)
{
    constexpr int WAVE = 64;
    constexpr int DIM = 16;
    constexpr int KA = Atom::KA, KPL = Atom::KPL;
    constexpr int BP = WP * PW, BQ = WQ * DIM, BN = WK * FN * DIM;
    constexpr int NW = WP * WQ * WK;
    constexpr int HH = BP + R - 1, HW = BQ + S - 1;
    constexpr int GE = 16 / static_cast<int>(sizeof(T)); // elements per granule
    constexpr int G = BC / GE; // granules per pixel per chunk
    constexpr int ROWG = HW * G;
    constexpr int NG = HH * ROWG; // granules per stage
    constexpr int KK = BC / KA;
    constexpr int NQ = S * KK; // steps per chunk
    constexpr int NR = PW + R - 1; // input fragments per step
    constexpr int FB = KPL * static_cast<int>(sizeof(T)); // fragment bytes per lane
    constexpr int FRAG = WAVE * FB; // bytes per weight fragment
    static_assert(Atom::WAVE == WAVE && Atom::DIM == DIM, "16x16 MFMA atoms only");
    static_assert(BC % GE == 0 && BC % KA == 0 && G >= 1 && (G & (G - 1)) == 0, "chunk");
    static_assert(WAVE % G == 0 && G <= 8, "DMA lane map");
    static_assert(PP == 0 || (PP == 1 && WK == 2), "ping-pong needs two channel halves");
    constexpr int XPI = WAVE / G; // columns per DMA instruction
    constexpr int XG = (HW + XPI - 1) / XPI; // instructions per halo row
    constexpr int NI = HH * XG; // instructions per stage
    constexpr int IPW = (NI + NW - 1) / NW; // per wave (the last one only for some waves)
    // DMA per step over steps 0..NQ-2 (step 0 alone when a chunk is one step).
    constexpr int DSTEPS = NQ > 1 ? NQ - 1 : 1;
    constexpr int DPS = (IPW + DSTEPS - 1) / DSTEPS;
    typedef typename K1Raw<16>::type Raw16;
    typedef typename K1Raw<FB>::type RawF;
    typedef typename Atom::frag Frag;

    __shared__ Raw16 st0[NG];
    __shared__ Raw16 st1[NG];

    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int wq = wave % WQ, wp = (wave / WQ) % WP, wk = wave / (WQ * WP);
    // Lane-derived values are recomputed from an opaque lane id where they are used, so the
    // compiler does not hoist dozens of per-step invariants into live registers.
    auto olane = [&]() __attribute__((always_inline))
    {
        // Unsigned and masked: the compiler then knows the lane is in [0, WAVE) and keeps the
        // divisions and remainders below as shifts and masks (a signed remainder is six VALU).
        unsigned l = static_cast<unsigned>(tid);
        asm volatile("" : "+v"(l));
        return static_cast<int>(l & (WAVE - 1));
    };

    // Tile walk: XCD x of K1_XCDS holds blocks b = x mod X (dispatch is round-robin) and owns a
    // share of the logical tiles proportional to its block count; its blocks take them
    // round-robin, so concurrent blocks of one XCD work on neighbouring tiles (shared L2).
    const int kBlocks = (a.k + BN - 1) / BN;
    const long long total = static_cast<long long>(a.tilesW) * a.tilesH * a.n * kBlocks;
    const int gsz = static_cast<int>(gridDim.x);
    const int xcd = static_cast<int>(blockIdx.x) % K1_XCDS;
    const int local = static_cast<int>(blockIdx.x) / K1_XCDS;
    auto cum = [&](int x) { return x * (gsz / K1_XCDS) + (x < gsz % K1_XCDS ? x : gsz % K1_XCDS); };
    const int bpx = cum(xcd + 1) - cum(xcd);
    const int tEnd = static_cast<int>(total * cum(xcd + 1) / gsz);
    const int tFirst = static_cast<int>(total * cum(xcd) / gsz) + local;

    struct Tile
    {
        int kb, oh0, ow0, img;
    };
    auto decode = [&](int t) __attribute__((always_inline))
    {
        Tile c;
        c.kb = t % kBlocks;
        t /= kBlocks;
        const int tq = t % a.tilesW;
        t /= a.tilesW;
        const int tp = t % a.tilesH;
        c.img = t / a.tilesH;
        c.oh0 = tp * BP;
        c.ow0 = tq * BQ;
        return c;
    };
    const long long imgElems = static_cast<long long>(a.hi) * a.wi * a.c;
    const long long imgBytes = imgElems * static_cast<long long>(sizeof(T));

    // This wave's DMA instructions [i0, i1) of chunk cc of tile tc into stage st. Lane l takes
    // column x = XPI xg + l / G and slot l % G, so its granule g = (l % G - x) mod G is the same
    // for every instruction. Out-of-image pixels and channels past c read as zero.
#if K2R_DMA16
    auto dma
        = [&](Raw16 * st, const Tile& tc, int cc, int i0, int i1) __attribute__((always_inline))
    {
        const T* xi = static_cast<const T*>(a.x) + tc.img * imgElems;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(xi, imgBytes);
        const int cbytes = cc * BC * static_cast<int>(sizeof(T));
        const int ln = olane();
        const int lx = ln / G;
        const int gofs = static_cast<int>(static_cast<unsigned>(ln % G - lx) & (G - 1)) * GE;
        const int iwl = tc.ow0 - a.pw + lx;
        const bool cok = gofs < a.c - cc * BC;
        int wv = wave;
        asm volatile("" : "+s"(wv)); // per-call scalar decode, not hoisted
#pragma unroll
        for(int it = 0; it < IPW; ++it)
        {
            if(it < i0 || it >= i1)
                continue;
            const int i = it * NW + wv; // wave-uniform
            if(i >= NI)
                continue;
            const int h = i / XG, xg = i - h * XG;
            const int ih = tc.oh0 - a.ph + h;
            const int iw = iwl + xg * XPI;
            const bool ok = (static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)) & cok;
            const unsigned off = ok ? static_cast<unsigned>(((ih * a.wi + iw) * a.c + gofs)
                                                            * static_cast<int>(sizeof(T)))
                                    : K1_OUT_OF_RANGE;
            if(HW % XPI == 0 || xg * XPI + lx < HW)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    xr,
                    (__attribute__((address_space(3))) void*)(st + h * ROWG + xg * WAVE),
                    16,
                    off,
                    cbytes,
                    0,
                    0);
        }
    };
#else
    // Dword DMA: sub-instruction j of run (h, xg) fills granules 16 j .. 16 j + 15 of the run,
    // lane l granule 16 j + l / 4 (column x = that / G, slot that % G) and its dword l % 4. The
    // fetch address is the rotation's inverse, as above, plus the dword.
    auto dma
        = [&](Raw16 * st, const Tile& tc, int cc, int i0, int i1) __attribute__((always_inline))
    {
        constexpr int SUB = K2R_DMA_SUB;
        constexpr int SPAN = WAVE / SUB; // granules per instruction
        const T* xi = static_cast<const T*>(a.x) + tc.img * imgElems;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(xi, imgBytes);
        const int cbytes = cc * BC * static_cast<int>(sizeof(T));
        const int ln = olane();
        const int dword = (ln % SUB) * (4 / static_cast<int>(sizeof(T)));
        // Per call (one channel chunk): each sub-instruction's lane column within the run, its
        // lane byte offset from the run's first pixel (column, granule, dword), and whether its
        // granule is past the channels. The run's pixel goes into the scalar offset below.
        int lxj[SUB], lofj[SUB];
        bool cokj[SUB];
#pragma unroll
        for(int j = 0; j < SUB; ++j)
        {
            const int q = j * SPAN + ln / SUB; // granule within the run
            const int lx = q / G;
            const int gran = static_cast<int>(static_cast<unsigned>(q % G - lx) & (G - 1));
            lxj[j] = lx;
            lofj[j] = (lx * a.c + gran * GE + dword) * static_cast<int>(sizeof(T));
            cokj[j] = gran * GE < a.c - cc * BC;
        }
        int wv = wave;
        asm volatile("" : "+s"(wv)); // per-call scalar decode, not hoisted
#pragma unroll
        for(int it = 0; it < IPW; ++it)
        {
            if(it < i0 || it >= i1)
                continue;
            const int i = it * NW + wv; // wave-uniform
            if(i >= NI)
                continue;
            const int h = i / XG, xg = i - h * XG;
            const int ih = tc.oh0 - a.ph + h;
            const bool hok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi);
            const int col0 = tc.ow0 - a.pw + xg * XPI; // the run's first column
            // The run's first pixel in bytes, uniform: into soffset when it is in the image (a
            // run left of the image keeps the negative part in the lane offset; a row outside
            // the image adds nothing, so its out-of-range lanes stay out of range).
            const int base = (ih * a.wi + col0) * a.c * static_cast<int>(sizeof(T));
            const int sbase = hok && base > 0 ? base : 0;
            const int vbase = hok ? base - sbase : 0;
#pragma unroll
            for(int j = 0; j < SUB; ++j)
            {
                const int lx = lxj[j];
                const int iw = col0 + lx;
                const bool ok
                    = hok & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)) & cokj[j];
                const unsigned off = ok ? static_cast<unsigned>(lofj[j] + vbase) : K1_OUT_OF_RANGE;
                if(HW % XPI == 0 || xg * XPI + lx < HW)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        xr,
                        (__attribute__((address_space(3))) void*)(st + h * ROWG + xg * WAVE
                                                                  + j * SPAN),
                        4,
                        off,
                        cbytes + sbase,
                        0,
                        0);
            }
        }
    };
#endif

    // Weights (k2rFormatWeights): this wave's run of FN fragments for (kc, s, r) is contiguous.
    const __amdgpu_buffer_rsrc_t wr = k1Rsrc(
        a.w, static_cast<long long>(a.ca) * a.kpad * KA * R * S * static_cast<int>(sizeof(T)));
    auto loadW = [&](Frag(&w)[FN], int r, int nb, int cc, int qq) __attribute__((always_inline))
    {
        const unsigned wl = static_cast<unsigned>(olane() * FB);
        const int so
            = ((((nb / (FN * DIM)) * a.ca + cc * KK + qq % KK) * S + qq / KK) * R + r) * FN * FRAG;
#pragma unroll
        for(int j = 0; j < FN; ++j)
            w[j] = __builtin_bit_cast(Frag, k2rBufferLoad<RawF>(wr, wl + j * FRAG, so));
    };

    typename Atom::acc acc[PW][FN];
    const int nCh = (a.c + BC - 1) / BC;

    // One chunk cc of the current tile from stage CUR; prefetches (dmaTile, dmaCc) into the other
    // stage when pf.
    auto chunk = [&](auto CUR, int nb, int cc, const Tile& dmaTile, int dmaCc, bool pf)
        __attribute__((always_inline))
    {
        constexpr int cs = decltype(CUR)::value;
        const Raw16* cur = cs ? st1 : st0;
        Raw16* nxt = cs ? st0 : st1;
        k2StaticFor<0, NQ>([&](auto Q) __attribute__((always_inline)) {
            constexpr int q = decltype(Q)::value;
            constexpr int s = q / KK, kk = (q % KK) * KA;
            // Memory phase: weights first (in-order vmcnt), then the DMA, then
            // the input fragments.
            Frag wfr[R][FN];
#pragma unroll
            for(int r = 0; r < R; ++r)
                loadW(wfr[r], r, nb, cc, q);
            if constexpr(q < DSTEPS)
            {
                if(pf)
                    dma(nxt, dmaTile, dmaCc, q * DPS, (q + 1) * DPS);
            }
            Frag xin[NR];
            {
                const int l = olane();
                const int x = wq * DIM + l % DIM + s;
                const int ch = kk + Atom::template kBase<KPL>(l);
                const int slot = x * G + ((ch / GE + x) % G);
                const T* base = reinterpret_cast<const T*>(cur) + slot * GE + ch % GE;
#pragma unroll
                for(int rr = 0; rr < NR; ++rr)
                    xin[rr] = __builtin_bit_cast(
                        Frag, *reinterpret_cast<const RawF*>(base + (wp * PW + rr) * ROWG * GE));
            }
            if constexpr(q == NQ - 1)
                __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
            else if constexpr(PP == 1)
            {
                // Weights landed; this step's DMA (issued after them) may stay
                // in flight. Count what every wave issues (a wave with one more
                // waits a little longer).
                constexpr int i0 = q * DPS;
                constexpr int i1 = (q + 1) * DPS < IPW ? (q + 1) * DPS : IPW;
                constexpr int full = NI / NW;
                constexpr int dmin = i1 <= full ? i1 - i0 : (i0 < full ? full - i0 : 0);
                // K2R_DMA_SUB instructions per issue slot; vmcnt holds at most 63.
                constexpr int dmi = (dmin > 0 ? dmin : 0) * K2R_DMA_SUB;
                constexpr int dm = dmi < 63 ? dmi : 63;
                if(pf)
                    __builtin_amdgcn_s_waitcnt(0x0F70 | (dm & 0xF) | ((dm >> 4) << 14));
                else
                    __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
            }
            if constexpr(PP == 1)
            {
                __builtin_amdgcn_s_waitcnt(K2R_LGKMCNT0);
                k2rBarrier();
                __builtin_amdgcn_s_setprio(1);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int r = 0; r < R; ++r)
#pragma unroll
                for(int j = 0; j < FN; ++j)
#pragma unroll
                    for(int p = 0; p < PW; ++p)
                        acc[p][j] = Atom::mma(wfr[r][j], xin[p + r], acc[p][j]);
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(PP == 1)
            {
                __builtin_amdgcn_s_setprio(0);
                k2rBarrier();
            }
        });
        if constexpr(PP == 0)
            k2rBarrier();
    };

    // Direct epilogue: the lane holds channels n .. n + 3 of column ow for each row p.
    auto store = [&](const Tile& tc, int nb) __attribute__((always_inline))
    {
        TO* y = static_cast<TO*>(a.y);
        const int l = olane();
        const int ow = tc.ow0 + wq * DIM + l % DIM;
        const bool vec = (a.k & 3) == 0;
#pragma unroll
        for(int p = 0; p < PW; ++p)
        {
            const int oh = tc.oh0 + wp * PW + p;
            if(oh >= a.ho || ow >= a.wo)
                continue;
            TO* yp = y + ((static_cast<long long>(tc.img) * a.ho + oh) * a.wo + ow) * a.k;
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int n = nb + j * DIM + (l / DIM) * 4;
                if(vec && n + 3 < a.k)
                {
                    typedef TO v4 __attribute__((ext_vector_type(4)));
                    *reinterpret_cast<v4*>(yp + n) = k2rPack4<TO>(acc[p][j]);
                }
                else
                {
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        if(n + i < a.k)
                            yp[n + i] = K1Elem<TO>::store(acc[p][j][i]);
                }
            }
        }
    };

    // Staged epilogue (EPI 1): each wave transposes its tile through a private slot of the stage
    // the last chunk consumed. A write instruction's 16-lane pass (one 4-channel group, pixels
    // x = 0..15) must hit 16 distinct 8-byte slots of a 128-byte bank row: granule gi of pixel x
    // sits at gi ^ key(x), key(x) = (x / XR) mod GP with XR = 8 / GP pixels per bank row, and the
    // pixels x >= 8 swap the two 8-byte halves of each granule (the reader swaps them back).
    // Granule-only keying left 8 slots for 16 lanes: a two-way conflict on every write. A barrier
    // after it keeps the next tile's DMA out of the slots.
    constexpr int GP = FN * DIM * static_cast<int>(sizeof(TO)) / 16; // granules per pixel
    constexpr bool STAGED = EPI == 1 && sizeof(TO) == 2 && PP == 0 && (GP & (GP - 1)) == 0
                            && GP <= 8 && WAVE / GP <= DIM && DIM * GP * NW <= NG;
    auto storeStaged = [&](Raw16 * slotBase, const Tile& tc, int nb) __attribute__((always_inline))
    {
        constexpr int XR = GP < 8 ? 8 / GP : 1;
        Raw16* slot = slotBase + wave * DIM * GP;
        TO* y = static_cast<TO*>(a.y);
        const int l = olane();
        const int x = l % DIM, g = l / DIM;
        const int px = l / GP, c8 = l % GP; // reader: pixel, granule
        constexpr int PPI = WAVE / GP; // pixels per store instruction
#pragma unroll
        for(int p = 0; p < PW; ++p)
        {
            const int oh = tc.oh0 + wp * PW + p;
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                typedef TO v4 __attribute__((ext_vector_type(4)));
                const v4 o = k2rPack4<TO>(acc[p][j]);
                const int gi = (j * DIM + g * 4) * static_cast<int>(sizeof(TO)) / 16;
                const int half = ((g * 4 * static_cast<int>(sizeof(TO))) % 16) ^ ((x >> 3) * 8);
                *reinterpret_cast<v4*>(
                    reinterpret_cast<char*>(slot + x * GP + (gi ^ ((x / XR) & (GP - 1)))) + half)
                    = o;
            }
#pragma unroll
            for(int k = 0; k < DIM / PPI; ++k)
            {
                const int xx = px + k * PPI;
                Raw16 v = slot[xx * GP + (c8 ^ ((xx / XR) & (GP - 1)))];
                if((xx >> 3) & 1)
                    v = __builtin_shufflevector(v, v, 2, 3, 0, 1);
                const int ow = tc.ow0 + wq * DIM + xx;
                const int n = nb + c8 * (16 / static_cast<int>(sizeof(TO)));
                if(oh < a.ho && ow < a.wo && n < a.k)
                    *reinterpret_cast<Raw16*>(
                        y + ((static_cast<long long>(tc.img) * a.ho + oh) * a.wo + ow) * a.k + n)
                        = v;
            }
        }
    };

    // The block's chunks as one stream over its tiles, two per iteration (stage 0, then stage
    // 1), so the chunk body exists twice whatever the parity of the chunk count. A tile's last
    // chunk prefetches the first chunk of the block's next tile, then the epilogue writes it.
    if(tFirst >= tEnd)
        return;
    Tile tc = decode(tFirst);
    int t = tFirst, cc = 0;
    int nb = tc.kb * BN + wk * FN * DIM;
    auto zero = [&]() __attribute__((always_inline))
    {
#pragma unroll
        for(int p = 0; p < PW; ++p)
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[p][j] = typename Atom::acc{};
    };
    zero();
    dma(st0, tc, 0, 0, IPW);
    __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
    k2rBarrier();
    if constexpr(PP == 1)
    {
        if(wk == 1)
            k2rBarrier();
    }
    // One chunk from stage SG; returns false when the block has no chunk left.
    auto step = [&](auto SG) __attribute__((always_inline))
    {
        constexpr int sc = decltype(SG)::value;
        const bool last = cc + 1 >= nCh;
        const bool hasNext = t + bpx < tEnd;
        // The chunk to prefetch: the next of this tile, or the first of the next tile.
        Tile dt = tc;
        if(last && hasNext)
            dt = decode(t + bpx);
        chunk(SG, nb, cc, dt, last ? 0 : cc + 1, !last || hasNext);
        if(!last)
        {
            ++cc;
            return true;
        }
        if constexpr(STAGED)
        {
            // 16-byte stores need whole granules of output channels. The chunk just read
            // stage sc and its end barrier passed, so that stage is free.
            if((a.k & (16 / static_cast<int>(sizeof(TO)) - 1)) == 0)
            {
                storeStaged(sc ? st1 : st0, tc, nb);
                k2rBarrier();
            }
            else
                store(tc, nb);
        }
        else
            store(tc, nb);
        if(!hasNext)
            return false;
        t += bpx;
        tc = dt;
        cc = 0;
        nb = tc.kb * BN + wk * FN * DIM;
        zero();
        return true;
    };
    while(step(K2Int<0>{}) && step(K2Int<1>{}))
    {
    }
    if constexpr(PP == 1)
    {
        if(wk == 0)
            k2rBarrier();
    }
}

// The 3-D form (KD > 1): the 2-D kernel above plus the plane walk (a tile is one output plane; its
// chunks run over the depth taps whose input plane lies in the image). A separate function, so
// the 2-D kernel's code is unchanged by it.
template <class T,
          class Atom,
          int R,
          int S,
          int PW,
          int FN,
          int WP,
          int WQ,
          int WK,
          int BC,
          int PP,
          int EPI,
          class TO,
          int KD>
__device__ __attribute__((always_inline)) void k2ConvRows3d(const K2Args& a)
{
    static_assert(KD > 1, "k2ConvRows3d is the 3-D form; 2-D configs run k2ConvRows");
    constexpr int WAVE = 64;
    constexpr int DIM = 16;
    constexpr int KA = Atom::KA, KPL = Atom::KPL;
    constexpr int BP = WP * PW, BQ = WQ * DIM, BN = WK * FN * DIM;
    constexpr int NW = WP * WQ * WK;
    constexpr int HH = BP + R - 1, HW = BQ + S - 1;
    constexpr int GE = 16 / static_cast<int>(sizeof(T)); // elements per granule
    constexpr int G = BC / GE; // granules per pixel per chunk
    constexpr int ROWG = HW * G;
    constexpr int NG = HH * ROWG; // granules per stage
    constexpr int KK = BC / KA;
    constexpr int NQ = S * KK; // steps per chunk
    constexpr int NR = PW + R - 1; // input fragments per step
    constexpr int FB = KPL * static_cast<int>(sizeof(T)); // fragment bytes per lane
    constexpr int FRAG = WAVE * FB; // bytes per weight fragment
    static_assert(Atom::WAVE == WAVE && Atom::DIM == DIM, "16x16 MFMA atoms only");
    static_assert(BC % GE == 0 && BC % KA == 0 && G >= 1 && (G & (G - 1)) == 0, "chunk");
    static_assert(WAVE % G == 0 && G <= 8, "DMA lane map");
    static_assert(PP == 0 || (PP == 1 && WK == 2), "ping-pong needs two channel halves");
    constexpr int XPI = WAVE / G; // columns per DMA instruction
    constexpr int XG = (HW + XPI - 1) / XPI; // instructions per halo row
    constexpr int NI = HH * XG; // instructions per stage
    constexpr int IPW = (NI + NW - 1) / NW; // per wave (the last one only for some waves)
    // DMA per step over steps 0..NQ-2 (step 0 alone when a chunk is one step).
    constexpr int DSTEPS = NQ > 1 ? NQ - 1 : 1;
    constexpr int DPS = (IPW + DSTEPS - 1) / DSTEPS;
    typedef typename K1Raw<16>::type Raw16;
    typedef typename K1Raw<FB>::type RawF;
    typedef typename Atom::frag Frag;

    __shared__ Raw16 st0[NG];
    __shared__ Raw16 st1[NG];

    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int wq = wave % WQ, wp = (wave / WQ) % WP, wk = wave / (WQ * WP);
    // Lane-derived values are recomputed from an opaque lane id where they are used, so the
    // compiler does not hoist dozens of per-step invariants into live registers.
    auto olane = [&]() __attribute__((always_inline))
    {
        // Unsigned and masked: the compiler then knows the lane is in [0, WAVE) and keeps the
        // divisions and remainders below as shifts and masks (a signed remainder is six VALU).
        unsigned l = static_cast<unsigned>(tid);
        asm volatile("" : "+v"(l));
        return static_cast<int>(l & (WAVE - 1));
    };

    // Tile walk: XCD x of K1_XCDS holds blocks b = x mod X (dispatch is round-robin) and owns a
    // share of the logical tiles proportional to its block count; its blocks take them
    // round-robin, so concurrent blocks of one XCD work on neighbouring tiles (shared L2).
    const int kBlocks = (a.k + BN - 1) / BN;
    long long total = static_cast<long long>(a.tilesW) * a.tilesH * a.n * kBlocks;
    if constexpr(KD > 1)
        total *= a.dout;
    const int gsz = static_cast<int>(gridDim.x);
    const int xcd = static_cast<int>(blockIdx.x) % K1_XCDS;
    const int local = static_cast<int>(blockIdx.x) / K1_XCDS;
    auto cum = [&](int x) { return x * (gsz / K1_XCDS) + (x < gsz % K1_XCDS ? x : gsz % K1_XCDS); };
    const int bpx = cum(xcd + 1) - cum(xcd);
    const int tEnd = static_cast<int>(total * cum(xcd + 1) / gsz);
    const int tFirst = static_cast<int>(total * cum(xcd) / gsz) + local;

    // 3-D (KD > 1): a tile is also one output plane od. Its chunks run over the depth taps whose
    // input plane is in the image, [t0, t0 + nt), times the channel chunks: chunk v is tap
    // t0 + v / nCh, channel chunk v % nCh. An output plane with no such tap runs one tap of an
    // out-of-image plane (zero input), so it still writes zeros.
    const int nCh = (a.c + BC - 1) / BC;
    struct Tile
    {
        int kb, oh0, ow0, img;
        int od, t0, nv; // 3-D only
    };
    auto decode = [&](int t) __attribute__((always_inline))
    {
        Tile c;
        c.kb = t % kBlocks;
        t /= kBlocks;
        const int tq = t % a.tilesW;
        t /= a.tilesW;
        const int tp = t % a.tilesH;
        if constexpr(KD > 1)
        {
            t /= a.tilesH;
            c.od = t % a.dout;
            c.img = t / a.dout;
            const int lo = a.pd - c.od, hi = a.di + a.pd - c.od; // taps [lo, hi) are in the image
            const int t0 = lo > 0 ? lo : 0, t1 = hi < KD ? hi : KD;
            c.t0 = t1 > t0 ? t0 : 0;
            c.nv = (t1 > t0 ? t1 - t0 : 1) * nCh;
        }
        else
        {
            c.img = t / a.tilesH;
            c.od = 0;
            c.t0 = 0;
            c.nv = nCh;
        }
        c.oh0 = tp * BP;
        c.ow0 = tq * BQ;
        return c;
    };
    const long long imgElems = KD > 1 ? static_cast<long long>(a.di) * a.hi * a.wi * a.c
                                      : static_cast<long long>(a.hi) * a.wi * a.c;
    const long long imgBytes = imgElems * static_cast<long long>(sizeof(T));

    // Chunk v of tile tc: its channel chunk and (3-D) its input plane.
    auto chunkOf = [&](const Tile& tc, int v, int& cc, int& plane) __attribute__((always_inline))
    {
        if constexpr(KD > 1)
        {
            const int tap = v / nCh;
            cc = v - tap * nCh;
            plane = tc.od - a.pd + tc.t0 + tap;
        }
        else
        {
            cc = v;
            plane = 0;
        }
    };

    // This wave's DMA instructions [i0, i1) of chunk cc of tile tc into stage st. Lane l takes
    // column x = XPI xg + l / G and slot l % G, so its granule g = (l % G - x) mod G is the same
    // for every instruction. Out-of-image pixels and channels past c read as zero.
#if K2R_DMA16
    auto dma = [&](Raw16 * st, const Tile& tc, int v, int i0, int i1) __attribute__((always_inline))
    {
        int cc, plane;
        chunkOf(tc, v, cc, plane);
        const bool pok = KD > 1 ? static_cast<unsigned>(plane) < static_cast<unsigned>(a.di) : true;
        const int prow = KD > 1 ? plane * a.hi : 0; // the plane's first row in the image
        const T* xi = static_cast<const T*>(a.x) + tc.img * imgElems;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(xi, imgBytes);
        const int cbytes = cc * BC * static_cast<int>(sizeof(T));
        const int ln = olane();
        const int lx = ln / G;
        const int gofs = static_cast<int>(static_cast<unsigned>(ln % G - lx) & (G - 1)) * GE;
        const int iwl = tc.ow0 - a.pw + lx;
        const bool cok = gofs < a.c - cc * BC;
        int wv = wave;
        asm volatile("" : "+s"(wv)); // per-call scalar decode, not hoisted
#pragma unroll
        for(int it = 0; it < IPW; ++it)
        {
            if(it < i0 || it >= i1)
                continue;
            const int i = it * NW + wv; // wave-uniform
            if(i >= NI)
                continue;
            const int h = i / XG, xg = i - h * XG;
            const int ih = tc.oh0 - a.ph + h;
            const int iw = iwl + xg * XPI;
            const bool ok = (static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)) & cok & pok;
            const unsigned off = ok ? static_cast<unsigned>((((prow + ih) * a.wi + iw) * a.c + gofs)
                                                            * static_cast<int>(sizeof(T)))
                                    : K1_OUT_OF_RANGE;
            if(HW % XPI == 0 || xg * XPI + lx < HW)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    xr,
                    (__attribute__((address_space(3))) void*)(st + h * ROWG + xg * WAVE),
                    16,
                    off,
                    cbytes,
                    0,
                    0);
        }
    };
#else
    // Dword DMA: sub-instruction j of run (h, xg) fills granules 16 j .. 16 j + 15 of the run,
    // lane l granule 16 j + l / 4 (column x = that / G, slot that % G) and its dword l % 4. The
    // fetch address is the rotation's inverse, as above, plus the dword.
    auto dma = [&](Raw16 * st, const Tile& tc, int v, int i0, int i1) __attribute__((always_inline))
    {
        int cc, plane;
        chunkOf(tc, v, cc, plane);
        const bool pok = KD > 1 ? static_cast<unsigned>(plane) < static_cast<unsigned>(a.di) : true;
        const int prow = KD > 1 ? plane * a.hi : 0; // the plane's first row in the image
        constexpr int SUB = K2R_DMA_SUB;
        constexpr int SPAN = WAVE / SUB; // granules per instruction
        const T* xi = static_cast<const T*>(a.x) + tc.img * imgElems;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(xi, imgBytes);
        const int cbytes = cc * BC * static_cast<int>(sizeof(T));
        const int ln = olane();
        const int dword = (ln % SUB) * (4 / static_cast<int>(sizeof(T)));
        // Per call (one channel chunk): each sub-instruction's lane column within the run, its
        // lane byte offset from the run's first pixel (column, granule, dword), and whether its
        // granule is past the channels. The run's pixel goes into the scalar offset below.
        int lxj[SUB], lofj[SUB];
        bool cokj[SUB];
#pragma unroll
        for(int j = 0; j < SUB; ++j)
        {
            const int q = j * SPAN + ln / SUB; // granule within the run
            const int lx = q / G;
            const int gran = static_cast<int>(static_cast<unsigned>(q % G - lx) & (G - 1));
            lxj[j] = lx;
            lofj[j] = (lx * a.c + gran * GE + dword) * static_cast<int>(sizeof(T));
            cokj[j] = gran * GE < a.c - cc * BC;
        }
        int wv = wave;
        asm volatile("" : "+s"(wv)); // per-call scalar decode, not hoisted
#pragma unroll
        for(int it = 0; it < IPW; ++it)
        {
            if(it < i0 || it >= i1)
                continue;
            const int i = it * NW + wv; // wave-uniform
            if(i >= NI)
                continue;
            const int h = i / XG, xg = i - h * XG;
            const int ih = tc.oh0 - a.ph + h;
            const bool hok = (static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)) & pok;
            const int col0 = tc.ow0 - a.pw + xg * XPI; // the run's first column
            // The run's first pixel in bytes, uniform: into soffset when it is in the image (a
            // run left of the image keeps the negative part in the lane offset; a row outside
            // the image adds nothing, so its out-of-range lanes stay out of range).
            const int base = ((prow + ih) * a.wi + col0) * a.c * static_cast<int>(sizeof(T));
            const int sbase = hok && base > 0 ? base : 0;
            const int vbase = hok ? base - sbase : 0;
#pragma unroll
            for(int j = 0; j < SUB; ++j)
            {
                const int lx = lxj[j];
                const int iw = col0 + lx;
                const bool ok
                    = hok & (static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)) & cokj[j];
                const unsigned off = ok ? static_cast<unsigned>(lofj[j] + vbase) : K1_OUT_OF_RANGE;
                if(HW % XPI == 0 || xg * XPI + lx < HW)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        xr,
                        (__attribute__((address_space(3))) void*)(st + h * ROWG + xg * WAVE
                                                                  + j * SPAN),
                        4,
                        off,
                        cbytes + sbase,
                        0,
                        0);
            }
        }
    };
#endif

    // Weights (k2rFormatWeights): this wave's run of FN fragments for (kc, s, t r) is contiguous;
    // the pre-pass orders a depth tap's filter rows as KD R rows (tap t, row r: t R + r).
    const __amdgpu_buffer_rsrc_t wr = k1Rsrc(
        a.w, static_cast<long long>(a.ca) * a.kpad * KA * KD * R * S * static_cast<int>(sizeof(T)));
    auto loadW
        = [&](Frag(&w)[FN], int r, int nb, int cc, int qq, int tap) __attribute__((always_inline))
    {
        const unsigned wl = static_cast<unsigned>(olane() * FB);
        const int so = ((((nb / (FN * DIM)) * a.ca + cc * KK + qq % KK) * S + qq / KK) * (KD * R)
                        + (KD > 1 ? tap * R : 0) + r)
                       * FN * FRAG;
#pragma unroll
        for(int j = 0; j < FN; ++j)
            w[j] = __builtin_bit_cast(Frag, k2rBufferLoad<RawF>(wr, wl + j * FRAG, so));
    };

    typename Atom::acc acc[PW][FN];

    // One chunk of the current tile (channel chunk cc, depth tap) from stage CUR; prefetches
    // chunk dmaCc of dmaTile into the other stage when pf.
    auto chunk = [&](auto CUR, int nb, int cc, int tap, const Tile& dmaTile, int dmaCc, bool pf)
        __attribute__((always_inline))
    {
        constexpr int cs = decltype(CUR)::value;
        const Raw16* cur = cs ? st1 : st0;
        Raw16* nxt = cs ? st0 : st1;
        k2StaticFor<0, NQ>([&](auto Q) __attribute__((always_inline)) {
            constexpr int q = decltype(Q)::value;
            constexpr int s = q / KK, kk = (q % KK) * KA;
            // Memory phase: weights first (in-order vmcnt), then the DMA, then
            // the input fragments.
            Frag wfr[R][FN];
#pragma unroll
            for(int r = 0; r < R; ++r)
                loadW(wfr[r], r, nb, cc, q, tap);
            if constexpr(q < DSTEPS)
            {
                if(pf)
                    dma(nxt, dmaTile, dmaCc, q * DPS, (q + 1) * DPS);
            }
            Frag xin[NR];
            {
                const int l = olane();
                const int x = wq * DIM + l % DIM + s;
                const int ch = kk + Atom::template kBase<KPL>(l);
                const int slot = x * G + ((ch / GE + x) % G);
                const T* base = reinterpret_cast<const T*>(cur) + slot * GE + ch % GE;
#pragma unroll
                for(int rr = 0; rr < NR; ++rr)
                    xin[rr] = __builtin_bit_cast(
                        Frag, *reinterpret_cast<const RawF*>(base + (wp * PW + rr) * ROWG * GE));
            }
            if constexpr(q == NQ - 1)
                __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
            else if constexpr(PP == 1)
            {
                // Weights landed; this step's DMA (issued after them) may stay
                // in flight. Count what every wave issues (a wave with one more
                // waits a little longer).
                constexpr int i0 = q * DPS;
                constexpr int i1 = (q + 1) * DPS < IPW ? (q + 1) * DPS : IPW;
                constexpr int full = NI / NW;
                constexpr int dmin = i1 <= full ? i1 - i0 : (i0 < full ? full - i0 : 0);
                // K2R_DMA_SUB instructions per issue slot; vmcnt holds at most 63.
                constexpr int dmi = (dmin > 0 ? dmin : 0) * K2R_DMA_SUB;
                constexpr int dm = dmi < 63 ? dmi : 63;
                if(pf)
                    __builtin_amdgcn_s_waitcnt(0x0F70 | (dm & 0xF) | ((dm >> 4) << 14));
                else
                    __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
            }
            if constexpr(PP == 1)
            {
                __builtin_amdgcn_s_waitcnt(K2R_LGKMCNT0);
                k2rBarrier();
                __builtin_amdgcn_s_setprio(1);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int r = 0; r < R; ++r)
#pragma unroll
                for(int j = 0; j < FN; ++j)
#pragma unroll
                    for(int p = 0; p < PW; ++p)
                        acc[p][j] = Atom::mma(wfr[r][j], xin[p + r], acc[p][j]);
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(PP == 1)
            {
                __builtin_amdgcn_s_setprio(0);
                k2rBarrier();
            }
        });
        if constexpr(PP == 0)
            k2rBarrier();
    };

    // The output plane of tile tc: the image (2-D), or the image's plane od (3-D).
    auto outPlane = [&](const Tile& tc) __attribute__((always_inline))
    {
        return KD > 1 ? static_cast<long long>(tc.img) * a.dout + tc.od
                      : static_cast<long long>(tc.img);
    };

    // Direct epilogue: the lane holds channels n .. n + 3 of column ow for each row p.
    auto store = [&](const Tile& tc, int nb) __attribute__((always_inline))
    {
        TO* y = static_cast<TO*>(a.y);
        const int l = olane();
        const int ow = tc.ow0 + wq * DIM + l % DIM;
        const bool vec = (a.k & 3) == 0;
#pragma unroll
        for(int p = 0; p < PW; ++p)
        {
            const int oh = tc.oh0 + wp * PW + p;
            if(oh >= a.ho || ow >= a.wo)
                continue;
            TO* yp = y + ((outPlane(tc) * a.ho + oh) * a.wo + ow) * a.k;
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int n = nb + j * DIM + (l / DIM) * 4;
                if(vec && n + 3 < a.k)
                {
                    typedef TO v4 __attribute__((ext_vector_type(4)));
                    *reinterpret_cast<v4*>(yp + n) = k2rPack4<TO>(acc[p][j]);
                }
                else
                {
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        if(n + i < a.k)
                            yp[n + i] = K1Elem<TO>::store(acc[p][j][i]);
                }
            }
        }
    };

    // Staged epilogue (EPI 1): each wave transposes its tile through a private slot of the stage
    // the last chunk consumed. A write instruction's 16-lane pass (one 4-channel group, pixels
    // x = 0..15) must hit 16 distinct 8-byte slots of a 128-byte bank row: granule gi of pixel x
    // sits at gi ^ key(x), key(x) = (x / XR) mod GP with XR = 8 / GP pixels per bank row, and the
    // pixels x >= 8 swap the two 8-byte halves of each granule (the reader swaps them back).
    // Granule-only keying left 8 slots for 16 lanes: a two-way conflict on every write. A barrier
    // after it keeps the next tile's DMA out of the slots.
    constexpr int GP = FN * DIM * static_cast<int>(sizeof(TO)) / 16; // granules per pixel
    constexpr bool STAGED = EPI == 1 && sizeof(TO) == 2 && PP == 0 && (GP & (GP - 1)) == 0
                            && GP <= 8 && WAVE / GP <= DIM && DIM * GP * NW <= NG;
    auto storeStaged = [&](Raw16 * slotBase, const Tile& tc, int nb) __attribute__((always_inline))
    {
        constexpr int XR = GP < 8 ? 8 / GP : 1;
        Raw16* slot = slotBase + wave * DIM * GP;
        TO* y = static_cast<TO*>(a.y);
        const int l = olane();
        const int x = l % DIM, g = l / DIM;
        const int px = l / GP, c8 = l % GP; // reader: pixel, granule
        constexpr int PPI = WAVE / GP; // pixels per store instruction
#pragma unroll
        for(int p = 0; p < PW; ++p)
        {
            const int oh = tc.oh0 + wp * PW + p;
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                typedef TO v4 __attribute__((ext_vector_type(4)));
                const v4 o = k2rPack4<TO>(acc[p][j]);
                const int gi = (j * DIM + g * 4) * static_cast<int>(sizeof(TO)) / 16;
                const int half = ((g * 4 * static_cast<int>(sizeof(TO))) % 16) ^ ((x >> 3) * 8);
                *reinterpret_cast<v4*>(
                    reinterpret_cast<char*>(slot + x * GP + (gi ^ ((x / XR) & (GP - 1)))) + half)
                    = o;
            }
#pragma unroll
            for(int k = 0; k < DIM / PPI; ++k)
            {
                const int xx = px + k * PPI;
                Raw16 v = slot[xx * GP + (c8 ^ ((xx / XR) & (GP - 1)))];
                if((xx >> 3) & 1)
                    v = __builtin_shufflevector(v, v, 2, 3, 0, 1);
                const int ow = tc.ow0 + wq * DIM + xx;
                const int n = nb + c8 * (16 / static_cast<int>(sizeof(TO)));
                if(oh < a.ho && ow < a.wo && n < a.k)
                    *reinterpret_cast<Raw16*>(y + ((outPlane(tc) * a.ho + oh) * a.wo + ow) * a.k
                                              + n)
                        = v;
            }
        }
    };

    // The block's chunks as one stream over its tiles, two per iteration (stage 0, then stage
    // 1), so the chunk body exists twice whatever the parity of the chunk count. A tile's last
    // chunk prefetches the first chunk of the block's next tile, then the epilogue writes it.
    if(tFirst >= tEnd)
        return;
    Tile tc = decode(tFirst);
    int t = tFirst, cc = 0;
    int nb = tc.kb * BN + wk * FN * DIM;
    auto zero = [&]() __attribute__((always_inline))
    {
#pragma unroll
        for(int p = 0; p < PW; ++p)
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[p][j] = typename Atom::acc{};
    };
    zero();
    dma(st0, tc, 0, 0, IPW);
    __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
    k2rBarrier();
    if constexpr(PP == 1)
    {
        if(wk == 1)
            k2rBarrier();
    }
    // One chunk from stage SG; returns false when the block has no chunk left.
    auto step = [&](auto SG) __attribute__((always_inline))
    {
        constexpr int sc = decltype(SG)::value;
        const bool last = cc + 1 >= (KD > 1 ? tc.nv : nCh);
        const bool hasNext = t + bpx < tEnd;
        // The chunk to prefetch: the next of this tile, or the first of the next tile.
        Tile dt = tc;
        if(last && hasNext)
            dt = decode(t + bpx);
        int wcc = cc, tap = 0; // chunk cc's channel chunk and depth tap (3-D)
        if constexpr(KD > 1)
        {
            const int tv = cc / nCh;
            wcc = cc - tv * nCh;
            tap = tc.t0 + tv;
        }
        chunk(SG, nb, wcc, tap, dt, last ? 0 : cc + 1, !last || hasNext);
        if(!last)
        {
            ++cc;
            return true;
        }
        if constexpr(STAGED)
        {
            // 16-byte stores need whole granules of output channels. The chunk just read
            // stage sc and its end barrier passed, so that stage is free.
            if((a.k & (16 / static_cast<int>(sizeof(TO)) - 1)) == 0)
            {
                storeStaged(sc ? st1 : st0, tc, nb);
                k2rBarrier();
            }
            else
                store(tc, nb);
        }
        else
            store(tc, nb);
        if(!hasNext)
            return false;
        t += bpx;
        tc = dt;
        cc = 0;
        nb = tc.kb * BN + wk * FN * DIM;
        zero();
        return true;
    };
    while(step(K2Int<0>{}) && step(K2Int<1>{}))
    {
    }
    if constexpr(PP == 1)
    {
        if(wk == 0)
            k2rBarrier();
    }
}

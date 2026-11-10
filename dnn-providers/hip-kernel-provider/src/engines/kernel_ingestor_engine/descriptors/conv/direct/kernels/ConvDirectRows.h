// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K2 "rows" kernel: stride-1 2-D direct convolution with filter-row reuse, channels-last, for
// the gfx950 matrix cores (16-byte LDS DMA). Included by ConvDirect.cpp after ConvDirectCore.h
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
// granules of row h, each lane fetching the global granule its slot holds. Two stages, separate
// LDS objects, so the compiler tracks each DMA against its own stage.
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
// Envelope: stride 1, dilation 1, groups 1, KD 1, any padding; reduction channels a multiple of
// 16 / sizeof(T); per-image input < 2 GiB; formatted weights < 2 GiB.
#pragma once

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

/// Weight pre-pass of the rows kernel. 16 x KA fragments are stored lane-major for the 16x16 MFMA
/// lane map (lane l: column l % 16, slice offset (l / 16) KPL), ordered so one wave's step (FN
/// column tiles x R filter rows of one filter column s and one KA slice kc) is one contiguous run:
/// wf[(((((g ca + kc) S + s) R + r) FN + j) 64 + l) KPL + e], output-column group g = n / (16 FN),
/// tile j = (n / 16) % FN, kc over cpad / ka. Roles (backward data, flip) as k2FormatWeights;
/// MATH 1 writes the three-bf16 form ([hi | lo | hi] of each weight along a 3x reduction).
template <class T, int KPL, int FN, int MATH>
__device__ void k2rFormatWeights(const K2FormatArgs& f)
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
                    out = part == 1 ? lo : hi;
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
__device__ void k2ConvRows(const K2Args& a)
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
        int l = tid;
        asm volatile("" : "+v"(l));
        return l % WAVE;
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
                constexpr int dm = dmin > 0 ? dmin : 0;
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
                    v4 o;
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        o[i] = K1Elem<TO>::store(acc[p][j][i]);
                    *reinterpret_cast<v4*>(yp + n) = o;
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
    // the last chunk consumed; granule gi of pixel x sits at gi ^ (x & (GP - 1)). A barrier
    // after it keeps the next tile's DMA out of the slots.
    constexpr int GP = FN * DIM * static_cast<int>(sizeof(TO)) / 16; // granules per pixel
    constexpr bool STAGED = EPI == 1 && sizeof(TO) == 2 && PP == 0 && (GP & (GP - 1)) == 0
                            && GP <= 8 && DIM * GP * NW <= NG;
    auto storeStaged = [&](Raw16 * slotBase, const Tile& tc, int nb) __attribute__((always_inline))
    {
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
                v4 o;
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    o[i] = K1Elem<TO>::store(acc[p][j][i]);
                const int gi = (j * DIM + g * 4) * static_cast<int>(sizeof(TO)) / 16;
                const int half = (g * 4 * static_cast<int>(sizeof(TO))) % 16;
                *reinterpret_cast<v4*>(
                    reinterpret_cast<char*>(slot + x * GP + (gi ^ (x & (GP - 1)))) + half)
                    = o;
            }
#pragma unroll
            for(int k = 0; k < DIM / PPI; ++k)
            {
                const int xx = px + k * PPI;
                const Raw16 v = slot[xx * GP + (c8 ^ (xx & (GP - 1)))];
                const int ow = tc.ow0 + wq * DIM + xx;
                const int n = nb + c8 * (16 / static_cast<int>(sizeof(TO)));
                if(oh < a.ho && ow < a.wo && n < a.k)
                    *reinterpret_cast<Raw16*>(
                        y + ((static_cast<long long>(tc.img) * a.ho + oh) * a.wo + ow) * a.k + n)
                        = v;
            }
        }
    };

    // All chunks of tile tc starting in stage S0; prefetches chunk 0 of tn (when hasNext) during
    // the last chunk.
    auto tileRun
        = [&](auto S0, const Tile& tc, const Tile& tn, bool hasNext) __attribute__((always_inline))
    {
        constexpr int s0 = decltype(S0)::value;
        const int nb = tc.kb * BN + wk * FN * DIM;
#pragma unroll
        for(int p = 0; p < PW; ++p)
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[p][j] = typename Atom::acc{};
        for(int cc = 0; cc < nCh; cc += 2)
        {
            {
                const bool last = cc + 1 >= nCh;
                chunk(K2Int<s0>{}, nb, cc, last ? tn : tc, last ? 0 : cc + 1, !last || hasNext);
            }
            if(cc + 1 < nCh)
            {
                const bool last = cc + 2 >= nCh;
                chunk(K2Int<1 - s0>{},
                      nb,
                      cc + 1,
                      last ? tn : tc,
                      last ? 0 : cc + 2,
                      !last || hasNext);
            }
        }
        if constexpr(STAGED)
        {
            // 16-byte stores need whole granules of output channels.
            if((a.k & (16 / static_cast<int>(sizeof(TO)) - 1)) == 0)
            {
                // The last chunk read stage (s0 + nCh - 1) % 2 and its end
                // barrier passed, so that stage is free.
                if(((s0 + nCh - 1) & 1) == 0)
                    storeStaged(st0, tc, nb);
                else
                    storeStaged(st1, tc, nb);
                k2rBarrier();
            }
            else
                store(tc, nb);
        }
        else
            store(tc, nb);
    };

    if(tFirst >= tEnd)
        return;
    Tile tc = decode(tFirst);
    dma(st0, tc, 0, 0, IPW);
    __builtin_amdgcn_s_waitcnt(K2R_VMCNT0);
    k2rBarrier();
    if constexpr(PP == 1)
    {
        if(wk == 1)
            k2rBarrier();
    }
    int parity = 0;
    for(int t = tFirst; t < tEnd; t += bpx)
    {
        const bool hasNext = t + bpx < tEnd;
        const Tile tn = decode(hasNext ? t + bpx : t);
        if(parity == 0)
            tileRun(K2Int<0>{}, tc, tn, hasNext);
        else
            tileRun(K2Int<1>{}, tc, tn, hasNext);
        parity ^= nCh & 1;
        tc = tn;
    }
    if constexpr(PP == 1)
    {
        if(wk == 0)
            k2rBarrier();
    }
}

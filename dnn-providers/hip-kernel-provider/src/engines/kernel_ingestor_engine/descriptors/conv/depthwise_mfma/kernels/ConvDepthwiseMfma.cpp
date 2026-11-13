// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// 2-D depthwise convolution at stride 1 or 2 on CDNA matrix cores, 16-bit. Channels-last runs
// the channel-diagonal core below; channels-first the channel-pair core (depthwiseMfmaNchw).
//
// Channels-last: a wave owns 8 channels and 32 output columns and streams down the output rows, holding one
// 16 x 16 fp32 accumulator per filter row as a ring (the accumulator of output row y collects
// the input rows y - pad .. y - pad + kh - 1, then leaves). One MFMA multiplies
//
//   A (16 x KA) = the taps of one filter row as a one-hot diagonal: row m = (q = m / 8, channel
//                 m % 8), k = 8 * p + channel (input position p of the pass);
//                 A[m][8p + c] = (c == m % 8) w[c][r][P * pass + p - q]
//   B (KA x 16) = input: column j, k = 8 * p + c holds channel c at input column 2j + p + P * pass
//
// so D[m][j] is output column 2j + q of channel m % 8: one MFMA covers 8 channels x 32 columns
// for P = KA / 8 taps (the off-diagonal products are zeros). A filter row takes
// ceil((kw + 1) / P) passes.
//
// The block's waves own adjacent channel groups (8 * WAVES channels, 16 B each per pixel). The
// block stages every input row once: coalesced 16-byte loads (DEPTH rows in flight in registers)
// into a double-buffered LDS row the waves read their B fragments from; finished output rows go
// back through a double-buffered LDS row and leave as coalesced 16-byte stores. One LDS barrier
// per row. Padding and out-of-range rows and columns read zero through out-of-range buffer
// offsets, and stores past the image go to an out-of-range offset, so the streaming body has no
// branches. FOLD images share the 32 columns (32 / FOLD each, with their own halo) for narrow
// images.
//
//   CONV_DWM_TYPE    _Float16 or __bf16
//   CONV_DWM_LAYOUT  0: channels-last (NHWC), 1: channels-first (NCHW, depthwiseMfmaNchw)
//   CONV_DWM_ATOM    32: v_mfma_f32_16x16x32 f16/bf16 (gfx950); 16: v_mfma_f32_16x16x16 f16 /
//                    bf16_1k (gfx908 f16, gfx90a, gfx94x, gfx950)
//   CONV_DWM_KH/KW   filter extent
//   CONV_DWM_STRIDE  the stride along both axes (1 or 2; backward data runs at 1)
//   CONV_DWM_WAVES   waves per block (8 channels each; channels-first: a channel pair each)
//   CONV_DWM_DEPTH   input rows in flight
//   CONV_DWM_FOLD    images sharing the output columns (32 channels-last: 1, 2 or 4; 128
//                    channels-first: 1, 2, 4 or 8)
//   CONV_DWM_FLIP    1 reads the filter reflected

// Embedded headers are keyed by bare file name (src/cmake/KernelEmbedding.cmake).
#include "ConvDepthwiseMfmaArgs.h"

#ifndef CONV_DWM_TYPE
#error "CONV_DWM_TYPE must name the element type"
#endif
#ifndef CONV_DWM_ATOM
#error "CONV_DWM_ATOM must select the MFMA"
#endif
#ifndef CONV_DWM_KH
#error "CONV_DWM_KH must give the filter height"
#endif
#ifndef CONV_DWM_KW
#error "CONV_DWM_KW must give the filter width"
#endif
#ifndef CONV_DWM_LAYOUT
#define CONV_DWM_LAYOUT 0
#endif
#ifndef CONV_DWM_STRIDE
#define CONV_DWM_STRIDE 1
#endif
#ifndef CONV_DWM_WAVES
#define CONV_DWM_WAVES 8
#endif
#ifndef CONV_DWM_DEPTH
#define CONV_DWM_DEPTH 2
#endif
#ifndef CONV_DWM_FOLD
#define CONV_DWM_FOLD 1
#endif
#ifndef CONV_DWM_FLIP
#define CONV_DWM_FLIP 0
#endif

// Buffer resource word 3 of the CDNA targets (gfx9): 32-bit data format.
#define CONV_DWM_RSRC_FLAGS 0x00020000
#define CONV_DWM_OOB 0x80000000u

namespace conv_depthwise_mfma
{

typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

constexpr int gcd(int x, int y)
{
    return y ? gcd(y, x % y) : x;
}
constexpr int lcm(int x, int y)
{
    return x / gcd(x, y) * y;
}

template <class T, int KA>
struct Atom
{
    static constexpr int KPL = KA / 4; // elements per lane
    static constexpr int P = KA / 8; // input positions per pass
    typedef T frag __attribute__((ext_vector_type(KPL)));
    __device__ static f4 mma(frag a, frag b, f4 c)
    {
        if constexpr(KA == 32)
        {
#if defined(__gfx950__)
            if constexpr(__is_same(T, _Float16))
                return __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
            else
                return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
#endif
        }
        else
        {
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) \
    || defined(__gfx942__) || defined(__gfx950__)
            if constexpr(__is_same(T, _Float16))
                return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#if !defined(__gfx908__)
            else
            {
                typedef short s4 __attribute__((ext_vector_type(4)));
                return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    __builtin_bit_cast(s4, a), __builtin_bit_cast(s4, b), c, 0, 0, 0);
            }
#endif
#endif
        }
        __builtin_trap();
        return c;
    }
};

// gfx94x has no dense 16x16x32 MFMA but a 2:4-sparse one (v_smfmac_f32_16x16x32): the
// channel-diagonal taps hold at most one nonzero per 4 channels, so they compress exactly.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define CONV_DWM_SPARSE32 1
#else
#define CONV_DWM_SPARSE32 0
#endif

/// D += A B with A 2:4-compressed (4 values a lane, 2-bit positions in idx), B 16 x ... dense.
template <class T>
__device__ inline f4 smfmac32(T __attribute__((ext_vector_type(4))) a,
                              T __attribute__((ext_vector_type(8))) b,
                              f4 c,
                              int idx)
{
#if CONV_DWM_SPARSE32
    if constexpr(__is_same(T, _Float16))
        return __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c, idx, 0, 0);
    else
    {
        typedef short s4 __attribute__((ext_vector_type(4)));
        typedef short s8 __attribute__((ext_vector_type(8)));
        return __builtin_amdgcn_smfmac_f32_16x16x32_bf16(
            __builtin_bit_cast(s4, a), __builtin_bit_cast(s8, b), c, idx, 0, 0);
    }
#endif
    __builtin_trap();
    return c;
}

/// Waits for this block's LDS traffic only (no vmcnt drain), then barriers.
__device__ inline void ldsBarrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <class T, int KA, int KH, int KW, int S, int WB, int D, int F, bool FLIP>
__device__ void depthwiseMfma(const ConvDepthwiseMfmaArgs& a)
{
    using Am = Atom<T, KA>;
    typedef typename Am::frag frag;
    typedef T v4 __attribute__((ext_vector_type(4)));
    typedef T v8 __attribute__((ext_vector_type(8)));
    constexpr int P = Am::P, KPL = Am::KPL;
    constexpr int KWP = (KW + S + P - 1) / P; // passes per filter row: ceil((kw + S) / P)
    constexpr int SW = 32 / F; // output columns per image segment
    constexpr int LWS = S * SW + P * KWP; // staged input columns per segment
    constexpr int LW = F * LWS;
    constexpr int NT = 64 * WB;
    constexpr int NL = (LW * WB + NT - 1) / NT; // staged granules per thread per row
    constexpr int NO = (32 * WB + NT - 1) / NT; // stored granules per thread per row
    constexpr int KM = WB < 8 ? WB - 1 : 7; // LDS swizzle key mask (granules per bank row)
    constexpr int RING = (KH + S - 1) / S; // live output rows
    // Unroll: accumulator ring (S * RING input rows), register ring (D), LDS input row (2) and
    // output row (2 completions, 2 S input rows) parities all static.
    constexpr int U = lcm(lcm(S * RING, D), 2 * S);
    // Added to step offsets so their divisions and remainders stay non-negative.
    constexpr int BIAS = 2 * S * RING * KH;
    __shared__ __attribute__((aligned(16))) T inL[2][LW * WB * 8];
    __shared__ __attribute__((aligned(16))) T outL[2][32 * WB * 8];

    const T* x = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.filter);
    T* y = static_cast<T*>(a.out);
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int cbs = a.c / (8 * WB), wTiles = (a.wo + SW - 1) / SW;
    int b = blockIdx.x;
    const int cb = b % cbs;
    b /= cbs;
    const int tx = b % wTiles;
    b /= wTiles;
    const int chk = b % a.chunks;
    const int gi = b / a.chunks;
    const int cB = cb * WB * 8, c0 = cB + wave * 8, x0 = tx * SW;
    const int oy0 = chk * a.rows, oy1 = min(a.ho, oy0 + a.rows);
    const int j = lane & 15, q = j >> 3, kc = j & 7, g = lane >> 4;
    const int pos = (g * KPL) >> 3, chOff = (g * KPL) & 7;

    // Taps: one one-hot diagonal fragment per filter row and pass.
    // Sparse (gfx94x, 16x16x32): the lane's 8 channels are two groups of 4 with the tap in
    // group kc / 4 at t = kc % 4; each group compresses to 2 values at ascending 2-bit
    // positions: (t, t + 1) with the tap first, or (2, 3) with it second for t = 3; the other
    // group holds zeros at (0, 1).
    constexpr bool SPARSE = KA == 32 && CONV_DWM_SPARSE32;
    typedef T c4 __attribute__((ext_vector_type(4)));
    frag A[SPARSE ? 1 : KH][SPARSE ? 1 : KWP];
    c4 As[SPARSE ? KH : 1][SPARSE ? KWP : 1];
    const int tg = kc & 3, lo = tg == 3 ? 2 : tg;
    const int tapPair = lo | (lo + 1) << 2, zeroPair = 0 | 1 << 2;
    const int Ai = kc < 4 ? (tapPair | zeroPair << 4) : (zeroPair | tapPair << 4);
    const int slotInGroup = tg == 3 ? 1 : 0;
#pragma unroll
    for(int r = 0; r < KH; ++r)
#pragma unroll
        for(int ph = 0; ph < KWP; ++ph)
        {
            const int s = P * ph + pos - S * q;
            T v = (T)0.0f;
            if(s >= 0 && s < KW && kc >= chOff && kc < chOff + KPL)
                v = w[((long)(c0 + kc) * KH + (FLIP ? KH - 1 - r : r)) * KW
                      + (FLIP ? KW - 1 - s : s)];
            if constexpr(SPARSE)
            {
                const T z = (T)0.0f;
                c4 a4 = c4{z, z, z, z};
                a4[(kc < 4 ? 0 : 2) + slotInGroup] = v;
                As[r][ph] = a4;
            }
            else
            {
#pragma unroll
                for(int e = 0; e < KPL; ++e)
                    A[r][ph][e] = chOff + e == kc ? v : (T)0.0f;
            }
        }

    // One buffer resource over the block's FOLD images (the host bounds their bytes by 2^31 - 1,
    // so CONV_DWM_OOB is past every one).
    const long imgE = (long)a.hi * a.wi * a.c, oimgE = (long)a.ho * a.wo * a.c;
    const int nimg = min(F, a.n - gi * F);
    const __amdgpu_buffer_rsrc_t xr
        = __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(x + gi * F * imgE),
                                            (short)0,
                                            (int)(nimg * imgE * (long)sizeof(T)),
                                            CONV_DWM_RSRC_FLAGS);
    const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
        y + gi * F * oimgE, (short)0, (int)(nimg * oimgE * (long)sizeof(T)), CONV_DWM_RSRC_FLAGS);

    // Staging plan: granule e = tid + i * NT is (LDS column pxa = e / WB, granule gc = e % WB).
    unsigned stOff[NL];
    int stLds[NL];
    bool stOn[NL];
#pragma unroll
    for(int i = 0; i < NL; ++i)
    {
        const int e = tid + i * NT, pxa = e / WB, gc = e % WB, f = pxa / LWS;
        const int ix = S * x0 - a.pw + pxa % LWS;
        stOn[i] = pxa < LW;
        stOff[i] = stOn[i] && ix >= 0 && ix < a.wi
                       ? (unsigned)((f * imgE + (long)ix * a.c + cB + gc * 8) * (long)sizeof(T))
                       : CONV_DWM_OOB;
        stLds[i] = (pxa * WB + (gc ^ ((pxa >> S) & KM))) * 8;
    }
    const unsigned rowB = (unsigned)((long)a.wi * a.c * (long)sizeof(T));
    auto gload = [&](int iy, v8* d) {
        const bool ok = iy >= 0 && iy < a.hi;
#pragma unroll
        for(int i = 0; i < NL; ++i)
            d[i] = __builtin_bit_cast(
                v8,
                __builtin_amdgcn_raw_buffer_load_b128(
                    xr,
                    ok && stOff[i] != CONV_DWM_OOB ? stOff[i] + (unsigned)iy * rowB : CONV_DWM_OOB,
                    0,
                    0));
    };
    auto lwrite = [&](int buf, const v8* d) {
#pragma unroll
        for(int i = 0; i < NL; ++i)
            if(stOn[i])
                *reinterpret_cast<v8*>(&inL[buf][stLds[i]]) = d[i];
    };
    int bSlot[KWP];
    {
        const int f = (2 * j) / SW;
#pragma unroll
        for(int ph = 0; ph < KWP; ++ph)
        {
            const int pxa = f * LWS + S * (2 * j - f * SW) + pos + P * ph;
            bSlot[ph] = (pxa * WB + (wave ^ ((pxa >> S) & KM))) * 8 + chOff;
        }
    }
    // The accumulator lane holds channels 4 (g % 2) .. + 3 of output column 2j + g / 2.
    const int opx = 2 * j + (g >> 1);
    const int oSlot = (opx * WB + (wave ^ ((opx >> 1) & KM))) * 8 + 4 * (g & 1);
    unsigned oOff[NO];
    int oRd[NO];
    bool oOn[NO];
#pragma unroll
    for(int i = 0; i < NO; ++i)
    {
        const int e = tid + i * NT, px = e / WB, gc = e % WB, f = px / SW;
        const int ox = x0 + px % SW;
        oOn[i] = px < 32;
        oOff[i] = oOn[i] && ox < a.wo
                      ? (unsigned)((f * oimgE + (long)ox * a.c + cB + gc * 8) * (long)sizeof(T))
                      : CONV_DWM_OOB;
        oRd[i] = (px * WB + (gc ^ ((px >> 1) & KM))) * 8;
    }
    const unsigned oRowB = (unsigned)((long)a.wo * a.c * (long)sizeof(T));
    auto ostore = [&](int buf, int oy) {
        const bool ok = oy >= oy0 && oy < oy1;
#pragma unroll
        for(int i = 0; i < NO; ++i)
            if(oOn[i])
                __builtin_amdgcn_raw_buffer_store_b128(
                    __builtin_bit_cast(u4, *reinterpret_cast<const v8*>(&outL[buf][oRd[i]])),
                    yr,
                    ok && oOff[i] != CONV_DWM_OOB ? oOff[i] + (unsigned)oy * oRowB : CONV_DWM_OOB,
                    0,
                    0);
    };

    f4 acc[RING];
#pragma unroll
    for(int r = 0; r < RING; ++r)
        acc[r] = f4{0.0f, 0.0f, 0.0f, 0.0f};
    // Step s reads input row S oy0 - ph + s. Filter row r of it feeds output row oy0 + (s - r) / S
    // when S divides s - r; output row oy0 + o completes in step S o + kh - 1.
    const int steps = S * (oy1 - oy0 - 1) + KH;
    const int iyBase = S * oy0 - a.ph;
    v8 stg[D][NL];
    {
        v8 first[NL];
        gload(iyBase, first);
        lwrite(0, first);
    }
#pragma unroll
    for(int i = 1; i <= D; ++i)
        gload(iyBase + i, stg[i % D]);
    ldsBarrier();
    for(int sb = 0;; sb += U)
    {
#pragma unroll
        for(int t = 0; t < U; ++t)
        {
            const int s = sb + t;
            frag bf[KWP];
#pragma unroll
            for(int ph = 0; ph < KWP; ++ph)
                bf[ph] = *reinterpret_cast<const frag*>(&inL[t & 1][bSlot[ph]]);
#pragma unroll
            for(int r = 0; r < KH; ++r)
            {
                if((t - r + BIAS) % S != 0)
                    continue;
                const int slot = (t - r + BIAS) / S % RING;
#pragma unroll
                for(int ph = 0; ph < KWP; ++ph)
                {
                    if constexpr(SPARSE)
                        acc[slot] = smfmac32<T>(As[r][ph], bf[ph], acc[slot], Ai);
                    else
                        acc[slot] = Am::mma(A[r][ph], bf[ph], acc[slot]);
                }
            }
            // A completed output row goes to the LDS row of its parity.
            if((t - (KH - 1) + BIAS) % S == 0)
            {
                const int done = (t - (KH - 1) + BIAS) / S % RING;
                const int parity = (t - (KH - 1) + BIAS) / S & 1;
                // Element-wise narrowing: __builtin_convertvector kept every ring entry live.
                v4 o;
#pragma unroll
                for(int v = 0; v < 4; ++v)
                    o[v] = (T)acc[done][v];
                *reinterpret_cast<v4*>(&outL[parity][oSlot]) = o;
                acc[done] = f4{0.0f, 0.0f, 0.0f, 0.0f};
            }
            lwrite((t + 1) & 1, stg[(t + 1) % D]);
            gload(iyBase + s + 1 + D, stg[(t + 1) % D]);
            // The row the previous step completed (t > 0 is compile-time: no branch past the
            // first step).
            if((t - KH + BIAS) % S == 0 && (t > 0 || sb > 0))
                ostore((t - KH + BIAS) / S & 1, oy0 + (s - KH) / S);
            ldsBarrier();
            if(s + 1 >= steps)
                goto drained;
        }
    }
drained:
    ostore((oy1 - oy0 - 1) & 1, oy1 - 1);
}

// Channels-first: the MFMA runs channel pairs along K instead of the channel diagonal.
//
//   A (16 x KA) = input: row j, k = P c + p (channel c of the pair, position p, P = KA / 2)
//                 holds channel c at input column S 8j + p + P pass -- a lane's elements are
//                 consecutive columns of one channel row, so it loads them straight from global
//                 memory (one 16- or 8-byte load, unaligned: the CDNA buffer path allows it)
//   B (KA x 16) = taps: column n = 8 c' + q, B[P c + p][8c' + q] = (c == c') w[c'][P pass + p - S q]
//
// so D[j][8c' + q] is channel c' at output column 8j + q: one MFMA covers 2 channels x 128 columns
// and every tap of a filter row up to kw = P + 1 - 7 S in one pass. A wave owns one channel pair
// and needs no LDS: its rows are contiguous in memory, so the 16 lanes of a column block read
// one run. FOLD images share the 16 column blocks (16 / FOLD each) for narrow images. Channels
// past c, columns past the row and rows past the image read zero through masks and
// out-of-range offsets; outputs past them store to an out-of-range offset.
template <class T, int KA, int KH, int KW, int S, int D, int F, bool FLIP>
__device__ void depthwiseMfmaNchw(const ConvDepthwiseMfmaArgs& a)
{
    using Am = Atom<T, KA>;
    typedef typename Am::frag frag;
    constexpr int KPL = Am::KPL;
    constexpr int P = KA / 2; // positions per channel per pass
    constexpr int KWP = (KW + 7 * S + P - 1) / P; // passes: ceil((kw + 7 S) / P)
    constexpr int JB = 16 / F; // column blocks per image segment
    constexpr int SW = 8 * JB; // output columns per segment
    constexpr int RING = (KH + S - 1) / S;
    constexpr int U = lcm(S * RING, D);
    constexpr int BIAS = 2 * S * RING * KH;
    constexpr int NW = KPL / 2; // 32-bit words per fragment
    typedef unsigned uw __attribute__((ext_vector_type(NW)));

    const T* x = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.filter);
    T* y = static_cast<T*>(a.out);
    const int lane = threadIdx.x & 63, waves = blockDim.x >> 6;
    const int pairBlocks = ((a.c + 1) / 2 + waves - 1) / waves;
    const int wTiles = (a.wo + SW - 1) / SW;
    int b = blockIdx.x;
    // Wave-uniform (readfirstlane): the edge-row branch below must stay a scalar branch.
    const int pair = __builtin_amdgcn_readfirstlane(b % pairBlocks * waves + (threadIdx.x >> 6));
    b /= pairBlocks;
    const int tx = b % wTiles;
    b /= wTiles;
    const int chk = b % a.chunks;
    const int gi = b / a.chunks;
    const int c0 = 2 * pair, x0 = tx * SW;
    const int oy0 = chk * a.rows, oy1 = min(a.ho, oy0 + a.rows);
    const int j = lane & 15, kb = lane >> 4;
    const int f = j / JB, jl = j % JB;

    // Taps: lane (n = 8 c' + q, kb) holds B[KPL kb .. + KPL - 1][n].
    frag Bt[KH][KWP];
    {
        const int cq = j >> 3, q = j & 7;
        const int cw = c0 + cq;
#pragma unroll
        for(int r = 0; r < KH; ++r)
#pragma unroll
            for(int ph = 0; ph < KWP; ++ph)
#pragma unroll
                for(int e = 0; e < KPL; ++e)
                {
                    const int k = KPL * kb + e, c = k / P, p = k % P;
                    const int s = P * ph + p - S * q;
                    T v = (T)0.0f;
                    if(c == cq && s >= 0 && s < KW && cw < a.c)
                        v = w[((long)cw * KH + (FLIP ? KH - 1 - r : r)) * KW
                              + (FLIP ? KW - 1 - s : s)];
                    Bt[r][ph][e] = v;
                }
    }

    // Input: lane (j, kb) loads channel c0 + c, columns S (x0 + 8 jl) - pw + p0 + P pass.
    const long planeI = (long)a.hi * a.wi, planeO = (long)a.ho * a.wo;
    const long imgI = planeI * a.c, imgO = planeO * a.c;
    const int nimg = min(F, a.n - gi * F);
    const __amdgpu_buffer_rsrc_t xr
        = __builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(x + gi * F * imgI),
                                            (short)0,
                                            (int)(nimg * imgI * (long)sizeof(T)),
                                            CONV_DWM_RSRC_FLAGS);
    const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
        y + gi * F * imgO, (short)0, (int)(nimg * imgO * (long)sizeof(T)), CONV_DWM_RSRC_FLAGS);
    const int cIn = (KPL * kb) / P, p0 = (KPL * kb) % P;
    const bool chanIn = c0 + cIn < a.c && f < nimg;
    unsigned inOff[KWP];
    uw inMask[KWP];
#pragma unroll
    for(int ph = 0; ph < KWP; ++ph)
    {
        const int col = S * (x0 + 8 * jl) - a.pw + p0 + P * ph;
        inOff[ph] = (unsigned)((f * imgI + (long)(c0 + cIn) * planeI + col) * (long)sizeof(T));
        // Elements outside the row read a neighbour's data (or nothing): masked to zero.
#pragma unroll
        for(int e = 0; e < NW; ++e)
        {
            const bool lo = chanIn && col + 2 * e >= 0 && col + 2 * e < a.wi;
            const bool hi = chanIn && col + 2 * e + 1 >= 0 && col + 2 * e + 1 < a.wi;
            inMask[ph][e] = (lo ? 0x0000ffffu : 0u) | (hi ? 0xffff0000u : 0u);
        }
    }
    const unsigned rowB = (unsigned)((long)a.wi * (long)sizeof(T));
    // A lane's columns may run past its row on either side (padding, the tile's edge): the
    // offsets reach into the neighbouring rows, whose elements the mask clears. The buffer
    // (the block's images) answers a load that straddles either of its ends with zeros, valid
    // elements included, so a row whose loads may reach before the buffer's first element or
    // past its last loads element by element. That is the first planes' top rows and the last
    // planes' bottom rows; for a plane narrower than the padding plus a tile (a 1 x 1 dy of a
    // 7 x 7 backward data, padded by 6), it is several channel pairs, not only the first and
    // the last. The bounds are the wave's lowest and highest element over its lanes (image,
    // channel of the pair, column and pass), so the branch stays wave-uniform.
    const long colLow = (long)S * x0 - a.pw;
    const long colHigh = (long)S * (x0 + 8 * (JB - 1)) - a.pw + P * (KWP - 1) + P;
    const long pairLow = (long)c0 * planeI + colLow;
    const long pairHigh = (long)(nimg - 1) * imgI + (long)(c0 + 1) * planeI + colHigh;
    const long bufferElements = (long)nimg * imgI;
    // Raw words: the mask is applied where the MFMA reads them, so a load's wait stays there.
    auto gload = [&](int iy, uw* d) {
        const bool ok = iy >= 0 && iy < a.hi;
        if(ok && (pairLow + (long)iy * a.wi < 0 || pairHigh + (long)iy * a.wi > bufferElements))
        {
#pragma unroll
            for(int ph = 0; ph < KWP; ++ph)
            {
                const int col = S * (x0 + 8 * jl) - a.pw + p0 + P * ph;
                const long start = f * imgI + (long)(c0 + cIn) * planeI + (long)iy * a.wi + col;
                frag v;
#pragma unroll
                for(int e = 0; e < KPL; ++e)
                {
                    const bool in = ok && chanIn && col + e >= 0 && col + e < a.wi;
                    const unsigned short h = __builtin_amdgcn_raw_buffer_load_b16(
                        xr, in ? (unsigned)((start + e) * (long)sizeof(T)) : CONV_DWM_OOB, 0, 0);
                    v[e] = __builtin_bit_cast(T, h);
                }
                d[ph] = __builtin_bit_cast(uw, v);
            }
            return;
        }
#pragma unroll
        for(int ph = 0; ph < KWP; ++ph)
        {
            const unsigned o = ok ? inOff[ph] + (unsigned)iy * rowB : CONV_DWM_OOB;
            if constexpr(NW == 4)
                d[ph] = __builtin_amdgcn_raw_buffer_load_b128(xr, o, 0, 0);
            else
                d[ph] = __builtin_amdgcn_raw_buffer_load_b64(xr, o, 0, 0);
        }
    };

    // Output: lane (n = 8 c' + q, rb) holds rows 4 rb .. + 3 (column blocks j) of column n.
    const int cOut = j >> 3, q = j & 7, rb = kb;
    unsigned outOff[4];
#pragma unroll
    for(int v = 0; v < 4; ++v)
    {
        const int jj = 4 * rb + v, fo = jj / JB, ox = x0 + 8 * (jj % JB) + q;
        outOff[v]
            = c0 + cOut < a.c && fo < nimg && ox < a.wo
                  ? (unsigned)((fo * imgO + (long)(c0 + cOut) * planeO + ox) * (long)sizeof(T))
                  : CONV_DWM_OOB;
    }
    const unsigned oRowB = (unsigned)((long)a.wo * (long)sizeof(T));
    auto ostore = [&](const f4& acc, int oy) {
        const bool ok = oy >= oy0 && oy < oy1;
#pragma unroll
        for(int v = 0; v < 4; ++v)
        {
            const T o = (T)acc[v];
            __builtin_amdgcn_raw_buffer_store_b16(
                __builtin_bit_cast(unsigned short, o),
                yr,
                ok && outOff[v] != CONV_DWM_OOB ? outOff[v] + (unsigned)oy * oRowB : CONV_DWM_OOB,
                0,
                0);
        }
    };

    f4 acc[RING];
#pragma unroll
    for(int r = 0; r < RING; ++r)
        acc[r] = f4{0.0f, 0.0f, 0.0f, 0.0f};
    const int steps = S * (oy1 - oy0 - 1) + KH;
    const int iyBase = S * oy0 - a.ph;
    uw ring[D][KWP];
#pragma unroll
    for(int i = 0; i < D; ++i)
        gload(iyBase + i, ring[i]);
    for(int sb = 0;; sb += U)
    {
#pragma unroll
        for(int t = 0; t < U; ++t)
        {
            const int s = sb + t;
#pragma unroll
            for(int r = 0; r < KH; ++r)
            {
                if((t - r + BIAS) % S != 0)
                    continue;
                const int slot = (t - r + BIAS) / S % RING;
#pragma unroll
                for(int ph = 0; ph < KWP; ++ph)
                    acc[slot] = Am::mma(__builtin_bit_cast(frag, ring[t % D][ph] & inMask[ph]),
                                        Bt[r][ph],
                                        acc[slot]);
            }
            gload(iyBase + s + D, ring[t % D]);
            if((t - (KH - 1) + BIAS) % S == 0)
            {
                const int done = (t - (KH - 1) + BIAS) / S % RING;
                ostore(acc[done], oy0 + (s - (KH - 1)) / S);
                acc[done] = f4{0.0f, 0.0f, 0.0f, 0.0f};
            }
            if(s + 1 >= steps)
                return;
        }
    }
}

} // namespace conv_depthwise_mfma

using ElementType = CONV_DWM_TYPE;

extern "C" __global__ void __launch_bounds__(64 * CONV_DWM_WAVES)
    ConvDepthwiseMfma(ConvDepthwiseMfmaArgs args)
{
#if CONV_DWM_LAYOUT == 0
    conv_depthwise_mfma::depthwiseMfma<ElementType,
                                       CONV_DWM_ATOM,
                                       CONV_DWM_KH,
                                       CONV_DWM_KW,
                                       CONV_DWM_STRIDE,
                                       CONV_DWM_WAVES,
                                       CONV_DWM_DEPTH,
                                       CONV_DWM_FOLD,
                                       CONV_DWM_FLIP != 0>(args);
#else
    conv_depthwise_mfma::depthwiseMfmaNchw<ElementType,
                                           CONV_DWM_ATOM,
                                           CONV_DWM_KH,
                                           CONV_DWM_KW,
                                           CONV_DWM_STRIDE,
                                           CONV_DWM_DEPTH,
                                           CONV_DWM_FOLD,
                                           CONV_DWM_FLIP != 0>(args);
#endif
}

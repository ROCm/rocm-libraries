// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit-stride 2-D depthwise convolution on CDNA matrix cores, channels-last, 16-bit.
//
// A wave owns 8 channels and 32 output columns and streams down the output rows, holding one
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
//   CONV_DWM_ATOM    32: v_mfma_f32_16x16x32 f16/bf16 (gfx950); 16: v_mfma_f32_16x16x16 f16 /
//                    bf16_1k (gfx908 f16, gfx90a, gfx94x, gfx950)
//   CONV_DWM_KH/KW   filter extent
//   CONV_DWM_WAVES   waves per block (8 channels each)
//   CONV_DWM_DEPTH   input rows in flight
//   CONV_DWM_FOLD    images sharing the 32 output columns (1, 2 or 4)
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

/// Waits for this block's LDS traffic only (no vmcnt drain), then barriers.
__device__ inline void ldsBarrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <class T, int KA, int KH, int KW, int WB, int D, int F, bool FLIP>
__device__ void depthwiseMfma(const ConvDepthwiseMfmaArgs& a)
{
    using Am = Atom<T, KA>;
    typedef typename Am::frag frag;
    typedef T v4 __attribute__((ext_vector_type(4)));
    typedef T v8 __attribute__((ext_vector_type(8)));
    constexpr int P = Am::P, KPL = Am::KPL;
    constexpr int KWP = (KW + P) / P; // passes per filter row: ceil((kw + 1) / P)
    constexpr int SW = 32 / F; // output columns per image segment
    constexpr int LWS = SW + P * KWP; // staged input columns per segment
    constexpr int LW = F * LWS;
    constexpr int NT = 64 * WB;
    constexpr int NL = (LW * WB + NT - 1) / NT; // staged granules per thread per row
    constexpr int NO = (32 * WB + NT - 1) / NT; // stored granules per thread per row
    constexpr int KM = WB < 8 ? WB - 1 : 7; // LDS swizzle key mask (granules per bank row)
    constexpr int U = lcm(lcm(KH, D), 2); // unroll: accumulator ring, register ring, LDS
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
    frag A[KH][KWP];
#pragma unroll
    for(int r = 0; r < KH; ++r)
#pragma unroll
        for(int ph = 0; ph < KWP; ++ph)
        {
            const int s = P * ph + pos - q;
            T v = (T)0.0f;
            if(s >= 0 && s < KW && kc >= chOff && kc < chOff + KPL)
                v = w[((long)(c0 + kc) * KH + (FLIP ? KH - 1 - r : r)) * KW
                      + (FLIP ? KW - 1 - s : s)];
#pragma unroll
            for(int e = 0; e < KPL; ++e)
                A[r][ph][e] = chOff + e == kc ? v : (T)0.0f;
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
        const int ix = x0 - a.pw + pxa % LWS;
        stOn[i] = pxa < LW;
        stOff[i] = stOn[i] && ix >= 0 && ix < a.wi
                       ? (unsigned)((f * imgE + (long)ix * a.c + cB + gc * 8) * (long)sizeof(T))
                       : CONV_DWM_OOB;
        stLds[i] = (pxa * WB + (gc ^ ((pxa >> 1) & KM))) * 8;
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
            const int pxa = f * LWS + 2 * j - f * SW + pos + P * ph;
            bSlot[ph] = (pxa * WB + (wave ^ ((pxa >> 1) & KM))) * 8 + chOff;
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

    f4 acc[KH];
#pragma unroll
    for(int r = 0; r < KH; ++r)
        acc[r] = f4{0.0f, 0.0f, 0.0f, 0.0f};
    // Step s reads input row oy0 - ph + s; output row oy0 + s - (kh - 1) completes in it.
    const int steps = (oy1 - oy0) + KH - 1;
    const int iyBase = oy0 - a.ph;
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
                const int slot = ((t - r) % KH + KH) % KH;
#pragma unroll
                for(int ph = 0; ph < KWP; ++ph)
                    acc[slot] = Am::mma(A[r][ph], bf[ph], acc[slot]);
            }
            const int done = (t + 1) % KH;
            // Element-wise narrowing: __builtin_convertvector kept every ring entry live (+40 VGPRs).
            v4 o;
#pragma unroll
            for(int v = 0; v < 4; ++v)
                o[v] = (T)acc[done][v];
            *reinterpret_cast<v4*>(&outL[t & 1][oSlot]) = o;
            acc[done] = f4{0.0f, 0.0f, 0.0f, 0.0f};
            lwrite((t + 1) & 1, stg[(t + 1) % D]);
            gload(iyBase + s + 1 + D, stg[(t + 1) % D]);
            // The row the previous step finished (t > 0 is compile-time: no branch past the first).
            if(t > 0 || sb > 0)
                ostore((t + 1) & 1, oy0 + s - KH);
            ldsBarrier();
            if(s + 1 >= steps)
                goto drained;
        }
    }
drained:
    ostore((steps - 1) & 1, oy1 - 1);
}

} // namespace conv_depthwise_mfma

using ElementType = CONV_DWM_TYPE;

extern "C" __global__ void __launch_bounds__(64 * CONV_DWM_WAVES)
    ConvDepthwiseMfma(ConvDepthwiseMfmaArgs args)
{
    conv_depthwise_mfma::depthwiseMfma<ElementType,
                                       CONV_DWM_ATOM,
                                       CONV_DWM_KH,
                                       CONV_DWM_KW,
                                       CONV_DWM_WAVES,
                                       CONV_DWM_DEPTH,
                                       CONV_DWM_FOLD,
                                       CONV_DWM_FLIP != 0>(args);
}

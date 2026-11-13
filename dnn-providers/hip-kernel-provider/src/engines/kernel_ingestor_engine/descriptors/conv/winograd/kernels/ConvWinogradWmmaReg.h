// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3) on RDNA WMMA (gfx11 / gfx12, wave32), fp16 and bf16, fp32 accumulate,
// NCHW, forward form. The register-operand variant: the input transform lands straight in the
// WMMA B fragments (no LDS for the transformed input, no barrier in the main loop) and the
// transformed filter fragments come straight from the pre-pass buffer (global / L0).
//
// Work item = (group, BK out channels, 16*TG tiles); 4*TG waves. Wave w owns transform row
// i = w % 4 (bins 4i .. 4i+3) for 16 tiles (tile group w / 4); per bin a BK x 16 block of fp32
// accumulators (BK/16 WMMA tiles). Lane: tile column lane % 16, channel half h = lane / 16 of each
// 16-channel chunk. The row-i transform needs input rows (ra, rb) only:
//   t[col] = d[ra][col] +- d[rb][col]          (column pass, B^T row i)
//   V[i][0..3] = t0 - t2, t1 + t2, t2 - t1, t1 - t3   (row pass)
// so the 4 waves of a tile group split the transform without redundancy. F16MATH: fp16 packed
// math on channel pairs (v_pk_add_f16; two roundings, like MIOpen's asm); otherwise fp32 math,
// rounded once (bf16 always).
//
// Fragment k order: a WMMA half-wave (lanes 16h .. 16h+15) reads A and B from its own lanes only
// (gfx11 computes D rows 2e + h in half h), so each half may order k its own way as long as A and
// B agree. gfx11: half h holds [own 8 channels, the other half's 8] (v_permlanex16 exchange);
// gfx12: half h holds its own 8 channels (the native split).
//
// Epilogue: the wave applies the right A (row i's 4 bins -> 2 values) in registers, the two
// values go to LDS (8 x BK x 16*TG fp32), then A^T over the 4 rows -> y with the parity mask.
// Filter pre-pass: k5WmmaFilterTransform (U[G][16][cgo][cgiPad] T).
#pragma once

#include "ConvWinogradWmma.h"

#ifndef K5_WREG_EDGE
#define K5_WREG_EDGE 1
#endif
// Keeps the scheduler from moving loads across pipeline stages.
#ifndef K5_SCHED_BARRIER
#define K5_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#endif

typedef _Float16 k5h2 __attribute__((ext_vector_type(2)));
typedef unsigned k5u2 __attribute__((ext_vector_type(2)));
typedef unsigned k5u4 __attribute__((ext_vector_type(4)));
typedef unsigned k5u8 __attribute__((ext_vector_type(8)));

// One 16-bit element of a dword (hi: the upper half) as fp32, and two fp32 rounded to a packed
// dword (round to nearest even; bf16 NaN payloads stay NaN).
template <class T>
__device__ inline float k5WregHalf(unsigned d, bool hi)
{
    if constexpr(__is_same(T, k5_bf16))
        return __uint_as_float(hi ? d & 0xffff0000u : d << 16);
    else
        return static_cast<float>(__builtin_bit_cast(k5h2, d)[hi ? 1 : 0]);
}
__device__ inline unsigned k5WregBfRound(float f)
{
    const unsigned u = __float_as_uint(f);
    const unsigned r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return (u & 0x7fffffffu) > 0x7f800000u ? (u >> 16) | 0x40u : r;
}
template <class T>
__device__ inline unsigned k5WregPack(float lo, float hi)
{
    if constexpr(__is_same(T, k5_bf16))
        return k5WregBfRound(lo) | (k5WregBfRound(hi) << 16);
    else
        return __builtin_bit_cast(unsigned,
                                  k5h2{static_cast<_Float16>(lo), static_cast<_Float16>(hi)});
}

// The other half-wave's value of the same lane position (gfx11 v_permlanex16). __GFX11__, not a
// chip list: every gfx11 target, gfx11-generic included, has the instruction.
__device__ inline unsigned k5SwapHalves(unsigned x)
{
#if defined(__GFX11__) && K5_WAVE32
    return static_cast<unsigned>(__builtin_amdgcn_permlanex16(
        static_cast<int>(x), static_cast<int>(x), 0x76543210, 0xfedcba98, true, false));
#else
    __builtin_trap();
    return x;
#endif
}

template <int V>
struct K5Int
{
    static constexpr int value = V;
};

// BINS bins of a transform row per wave (4: the whole row; 2: half, two waves per row).
template <class T, bool GFX12, int BK, int TG, int BINS = 4>
struct K5WregCfg
{
    static constexpr int NJ = 4 / BINS; // waves per row
    static constexpr int MT = BK / 16, WPG = 4 * NJ, WAVES = WPG * TG, THREADS = 32 * WAVES,
                         BT = 16 * TG;
    static constexpr int LDS = 8 * NJ * BK * BT; // floats: Z[row i][s][part j][k][t]
    static constexpr int FD = GFX12 ? 4 : 8; // dwords per fragment
};

// Bin row I of B^T: t = d[RA] + SIGN * d[RB].
template <int I>
struct K5Row
{
    static constexpr int RA = I == 0 ? 0 : I == 2 ? 2 : 1;
    static constexpr int RB = I == 0 ? 2 : I == 1 ? 2 : I == 2 ? 1 : 3;
    static constexpr bool PLUS = I == 1;
};

__device__ inline k5h2 k5AsH2(unsigned v)
{
    return __builtin_bit_cast(k5h2, v);
}
__device__ inline unsigned k5AsU(k5h2 v)
{
    return __builtin_bit_cast(unsigned, v);
}

template <class T, bool GFX12, int BK, int TG, bool F16MATH, int BINS, int I, int J>
__device__ __attribute__((always_inline)) void k5WregBody(const K5Args& a, float* Zs)
{
    typedef K5Wmma<T, GFX12> W;
    typedef K5WregCfg<T, GFX12, BK, TG, BINS> C;
    constexpr int NJ = C::NJ, B0 = J * BINS; // this wave's first bin within the row
    typedef K5Row<I> R;
    constexpr int MT = C::MT, BT = C::BT, THREADS = C::THREADS, FD = C::FD;
    typedef typename W::frag frag;

    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int nCh = cgiPad / 16;
    const int chanStride = a.hi * a.wi;
    // The input is read as aligned dwords (rows start at any half): a 12-byte load covers a row's
    // 4 columns. A load partly past the buffer's end returns all zeros, and one before its start
    // would leave the allocation, so the chunks next to either end of the tensor read per element
    // (the edge path); the records round up to a dword (never past the allocation's last dword).
    const long long inElems = static_cast<long long>(a.n) * a.ci * a.hi * a.wi;
    const __amdgpu_buffer_rsrc_t rIn
        = k5Rsrc(a.in, (inElems * static_cast<long long>(sizeof(T)) + 3) / 4 * 4);
    // Odd channel strides alternate the dword phase between channels.
    const bool csOdd = (chanStride & 1) != 0;
    const bool evenW = (a.wi & 1) == 0;
    const __amdgpu_buffer_rsrc_t rU = k5Rsrc(a.u,
                                             static_cast<long long>(a.groups) * 16 * cgo * cgiPad
                                                 * static_cast<long long>(sizeof(T)));
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, tg = wave / C::WPG;
    const int h = lane >> 4, col = lane & 15;
    T* out = static_cast<T*>(a.out);
    const unsigned chanBytes = static_cast<unsigned>(chanStride) * sizeof(T);

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int kb = static_cast<int>(item % nKB);
        const long long r0 = item / nKB;
        const int tb = static_cast<int>(r0 % nTB);
        const int g = static_cast<int>(r0 / nTB);
        const int k0 = kb * BK, t0 = tb * BT;

        // This lane's tile and the byte offsets of its two input rows (channel 8h of the group).
        const int t = t0 + tg * 16 + col;
        // Per row: the element offset of column 0 (channel 8h of the group; -1 when the row is
        // padding), and for the fast path the aligned dword offsets and half shift for an even /
        // odd channel phase.
        int eA = -1, eB = -1;
        bool rowA = false, rowB = false;
        unsigned voA[2] = {K5_OOB, K5_OOB}, voB[2] = {K5_OOB, K5_OOB}, shA[2] = {0, 0},
                 shB[2] = {0, 0};
        unsigned m01 = 0, m23 = 0; // column masks (half lanes)
        bool c0 = false, c1 = false, c2 = false, c3 = false;
        if(t < a.tiles)
        {
            const int tw = t % a.tw, rr = t / a.tw;
            const int th = rr % a.th, img = rr / a.th;
            const int ih0 = th * 2 - a.ph, iw0 = tw * 2 - a.pw;
            const int base = ((img * a.ci + g * cgi + 8 * h) * a.hi) * a.wi + iw0;
            const auto rowOffsets
                = [&](int row, int& e, bool& ok, unsigned(&vo)[2], unsigned(&sh)[2]) {
                      if(row < 0 || row >= a.hi)
                          return;
                      ok = true;
                      e = base + row * a.wi;
                      const int b = e * static_cast<int>(sizeof(T));
#pragma unroll
                      for(int r = 0; r < 2; ++r)
                      {
                          // Biased by 16 bytes (taken back from the channel offset) so that left
                          // padding never makes the per-lane offset negative.
                          const int br = b + 2 * r;
                          if(br >= -16)
                          {
                              vo[r] = static_cast<unsigned>((br & ~3) + 16);
                              sh[r] = static_cast<unsigned>((br & 2) * 8);
                          }
                      }
                  };
            rowOffsets(ih0 + R::RA, eA, rowA, voA, shA);
            rowOffsets(ih0 + R::RB, eB, rowB, voB, shB);
            c0 = iw0 >= 0 && iw0 < a.wi;
            c1 = iw0 + 1 >= 0 && iw0 + 1 < a.wi;
            c2 = iw0 + 2 >= 0 && iw0 + 2 < a.wi;
            c3 = iw0 + 3 >= 0 && iw0 + 3 < a.wi;
            m01 = (c0 ? 0xffffu : 0u) | (c1 ? 0xffff0000u : 0u);
            m23 = (c2 ? 0xffffu : 0u) | (c3 ? 0xffff0000u : 0u);
        }
        // Even-width fast path (fp16 math): every row of every channel starts at the same half
        // phase pw & 1 (uniform), so the column pass runs on the loaded dwords as they are and one
        // v_perm per column and channel pair picks the column's halves of both channels, or zeros
        // for a padding column. Selectors for phase 0 and 1.
        unsigned selP[2][4];
        {
            const bool cok[4] = {c0, c1, c2, c3};
#pragma unroll
            for(int ph = 0; ph < 2; ++ph)
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    selP[ph][j] = !cok[j]          ? 0x0c0c0c0cu
                                  : ((j + ph) & 1) ? 0x07060302u
                                                   : 0x05040100u;
        }
        // The item's first and last images (uniform), for the edge-path test.
        const int perImage = a.th * a.tw;
        const int imgFirst = t0 / perImage;
        const int imgLast = (t0 + BT < a.tiles ? t0 + BT - 1 : a.tiles - 1) / perImage;

        // Raw input for one chunk: rows a and b, 8 channels: 3 aligned dwords each (fast path,
        // shifted into place in the transform) or the 2 dwords already in place (edge path).
        typedef unsigned k5u3 __attribute__((ext_vector_type(3)));
        k5u3 rA[8], rB[8];
        bool edge = false;
        auto load = [&](int ch, bool mayEdge) __attribute__((always_inline))
        {
            const int cBase = ch * 16 + 8 * h; // first channel of this half
            // Edge path: the chunk's first element is within reach of the buffer's start (left
            // padding of row 0), or its last row reaches the end.
            const long long first
                = (static_cast<long long>(imgFirst) * a.ci + g * cgi + ch * 16) * chanStride;
            const int cEnd = ch * 16 + 16 < cgi ? ch * 16 + 16 : cgi;
            const long long last
                = (static_cast<long long>(imgLast) * a.ci + g * cgi + cEnd) * chanStride;
            edge = mayEdge && (first < 16 + a.pw || last + 8 > inElems);
            if(edge)
            {
#pragma unroll
                for(int c = 0; c < 8; ++c)
                {
                    const bool ok = cBase + c < cgi;
#pragma unroll
                    for(int rw = 0; rw < 2; ++rw)
                    {
                        const int e0 = (rw == 0 ? eA : eB) + c * chanStride;
                        const bool rok = ok && (rw == 0 ? rowA : rowB);
                        unsigned hv[4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            const int e = e0 + j;
                            hv[j] = __builtin_amdgcn_raw_buffer_load_b16(
                                rIn,
                                rok && e >= 0 ? static_cast<unsigned>(e) * sizeof(T)
                                                    + static_cast<unsigned>(ch * 16) * chanBytes
                                              : static_cast<unsigned>(K5_OOB),
                                0,
                                0);
                        }
                        const k5u3 w3{hv[0] | (hv[1] << 16), hv[2] | (hv[3] << 16), 0u};
                        (rw == 0 ? rA : rB)[c] = w3;
                    }
                    K5_SCHED_BARRIER(); // rare path: one channel in flight at a time
                }
                return;
            }
#pragma unroll
            for(int c = 0; c < 8; ++c)
            {
                const bool ok = cBase + c < cgi;
                const unsigned so = static_cast<unsigned>(ch * 16 + c) * chanBytes;
                const int r = csOdd && (c & 1) ? 1 : 0; // the channel offset's half phase (uniform)
                // Chunks past the first carry >= 32 bytes of channel offset to take the bias from;
                // in the first chunk (fast path) every offset is >= 0 unbiased.
                const unsigned bias = ch > 0 ? 16u : 0u;
                const unsigned soA = so - 2 * r - bias;
                rA[c] = __builtin_bit_cast(
                    k5u3,
                    __builtin_amdgcn_raw_buffer_load_b96(
                        rIn, ok ? voA[r] - (16 - bias) : static_cast<unsigned>(K5_OOB), soA, 0));
                rB[c] = __builtin_bit_cast(
                    k5u3,
                    __builtin_amdgcn_raw_buffer_load_b96(
                        rIn, ok ? voB[r] - (16 - bias) : static_cast<unsigned>(K5_OOB), soA, 0));
            }
        };
        // Transformed chunk: 4 bins x 4 channel-pair dwords (this half's channels).
        unsigned v[BINS][4];
        // The row pass for this wave's bins (the others are dead code).
        auto rowPass = [&](int q, k5h2 p0, k5h2 p1, k5h2 p2, k5h2 p3) __attribute__((always_inline))
        {
            const unsigned all[4]
                = {k5AsU(p0 - p2), k5AsU(p1 + p2), k5AsU(p2 - p1), k5AsU(p1 - p3)};
#pragma unroll
            for(int b = 0; b < BINS; ++b)
                v[b][q] = all[B0 + b];
        };
        auto transformEven = [&](auto phaseTag) __attribute__((always_inline))
        {
            constexpr int S = decltype(phaseTag)::value;
            constexpr int ND = S ? 3 : 2; // dwords holding the 4 columns
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                unsigned t[2][3];
#pragma unroll
                for(int e = 0; e < 2; ++e)
                {
                    const k5u3 ra = rA[2 * q + e], rb = rB[2 * q + e];
                    const unsigned av[3] = {ra.x, ra.y, ra.z}, bv[3] = {rb.x, rb.y, rb.z};
#pragma unroll
                    for(int d = 0; d < ND; ++d)
                        t[e][d] = k5AsU(R::PLUS ? k5AsH2(av[d]) + k5AsH2(bv[d])
                                                : k5AsH2(av[d]) - k5AsH2(bv[d]));
                }
                k5h2 p[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    const int d = (j + S) >> 1;
                    p[j] = k5AsH2(__builtin_amdgcn_perm(t[1][d], t[0][d], selP[S][j]));
                }
                rowPass(q, p[0], p[1], p[2], p[3]);
            }
        };
        auto transform = [&]() {
            if constexpr(F16MATH && __is_same(T, k5_half))
            {
                if(evenW)
                {
                    if(!edge && (a.pw & 1))
                        transformEven(K5Int<1>{});
                    else
                        transformEven(K5Int<0>{});
                    return;
                }
            }
            k5u2 dA[8], dB[8];
#pragma unroll
            for(int c = 0; c < 8; ++c)
            {
                const int r = csOdd && (c & 1) ? 1 : 0;
                const unsigned sa = edge ? 0u : shA[r], sb = edge ? 0u : shB[r];
                dA[c] = k5u2{__builtin_amdgcn_alignbit(rA[c].y, rA[c].x, sa),
                             __builtin_amdgcn_alignbit(rA[c].z, rA[c].y, sa)};
                dB[c] = k5u2{__builtin_amdgcn_alignbit(rB[c].y, rB[c].x, sb),
                             __builtin_amdgcn_alignbit(rB[c].z, rB[c].y, sb)};
            }
#pragma unroll
            for(int q = 0; q < 4; ++q)
            {
                if constexpr(F16MATH)
                {
                    k5h2 tc[2][2]; // [channel of the pair][column pair]
#pragma unroll
                    for(int e = 0; e < 2; ++e)
                    {
                        const int c = 2 * q + e;
                        const k5h2 a01 = k5AsH2(dA[c].x), a23 = k5AsH2(dA[c].y);
                        const k5h2 b01 = k5AsH2(dB[c].x), b23 = k5AsH2(dB[c].y);
                        tc[e][0] = R::PLUS ? a01 + b01 : a01 - b01;
                        tc[e][1] = R::PLUS ? a23 + b23 : a23 - b23;
                        tc[e][0] = k5AsH2(k5AsU(tc[e][0]) & m01);
                        tc[e][1] = k5AsH2(k5AsU(tc[e][1]) & m23);
                    }
                    // Per column, the channel pair.
                    const k5h2 p0 = __builtin_shufflevector(tc[0][0], tc[1][0], 0, 2);
                    const k5h2 p1 = __builtin_shufflevector(tc[0][0], tc[1][0], 1, 3);
                    const k5h2 p2 = __builtin_shufflevector(tc[0][1], tc[1][1], 0, 2);
                    const k5h2 p3 = __builtin_shufflevector(tc[0][1], tc[1][1], 1, 3);
                    rowPass(q, p0, p1, p2, p3);
                }
                else
                {
                    float o[2][4];
#pragma unroll
                    for(int e = 0; e < 2; ++e)
                    {
                        const int c = 2 * q + e;
                        const unsigned ad[2] = {dA[c].x, dA[c].y}, bd[2] = {dB[c].x, dB[c].y};
                        float tcol[4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            const float af = k5WregHalf<T>(ad[j >> 1], j & 1);
                            const float bf = k5WregHalf<T>(bd[j >> 1], j & 1);
                            tcol[j] = R::PLUS ? af + bf : af - bf;
                        }
                        tcol[0] = c0 ? tcol[0] : 0.0f;
                        tcol[1] = c1 ? tcol[1] : 0.0f;
                        tcol[2] = c2 ? tcol[2] : 0.0f;
                        tcol[3] = c3 ? tcol[3] : 0.0f;
                        o[e][0] = tcol[0] - tcol[2];
                        o[e][1] = tcol[1] + tcol[2];
                        o[e][2] = tcol[2] - tcol[1];
                        o[e][3] = tcol[1] - tcol[3];
                    }
#pragma unroll
                    for(int b = 0; b < BINS; ++b)
                        v[b][q] = k5WregPack<T>(o[0][B0 + b], o[1][B0 + b]);
                }
            }
        };

        k5f8 acc[BINS][MT];
#pragma unroll
        for(int b = 0; b < BINS; ++b)
#pragma unroll
            for(int m = 0; m < MT; ++m)
                acc[b][m] = k5f8{0, 0, 0, 0, 0, 0, 0, 0};

        // U fragments: out channel k0 + m*16 + col; this half's channels first (gfx11). Rows past
        // cgo read other bins' data (or zeros past the buffer): they only feed accumulator rows
        // the epilogue drops. All but the lane part of the offset is uniform.
        const int uRow = k0 + col;
        const unsigned uLane
            = static_cast<unsigned>(((g * 16 + 4 * I + B0) * cgo + uRow) * cgiPad + 8 * h)
              * sizeof(T);
        const unsigned uLane2
            = uLane + static_cast<unsigned>((8 - 16 * h) * static_cast<int>(sizeof(T)));
        const unsigned binBytes = static_cast<unsigned>(cgo) * cgiPad * sizeof(T);
        frag ufr[BINS][MT];
        auto loadU = [&](int ch, int b) {
#pragma unroll
            for(int m = 0; m < MT; ++m)
            {
                const unsigned so = static_cast<unsigned>(ch * 16) * sizeof(T) + b * binBytes
                                    + static_cast<unsigned>(m * 16 * cgiPad) * sizeof(T);
                if constexpr(GFX12)
                {
                    ufr[b][m] = __builtin_bit_cast(
                        frag, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane, so, 0));
                }
                else
                {
                    const k5u4 lo = __builtin_bit_cast(
                        k5u4, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane, so, 0));
                    const k5u4 hi = __builtin_bit_cast(
                        k5u4, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane2, so, 0));
                    const k5u8 f8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                    ufr[b][m] = __builtin_bit_cast(frag, f8);
                }
            }
        };

        // Software pipeline: chunk ch's WMMAs run while chunk ch+1's input (issued first) and,
        // bin by bin into the registers just consumed, its U fragments load. Items that may touch
        // either end of the input run a copy with the edge-path test (its branch blurs the wait
        // counts); the rest run without it.
        auto chunks = [&](bool mayEdge) __attribute__((always_inline))
        {
            load(0, mayEdge);
#pragma unroll
            for(int b = 0; b < BINS; ++b)
                loadU(0, b);
            transform();
            for(int ch = 0; ch < nCh; ++ch)
            {
                // The last chunk reloads itself: unconditional loads keep the wait counts exact.
                const int nx = ch + 1 < nCh ? ch + 1 : ch;
                load(nx, mayEdge);
                K5_SCHED_BARRIER();
#pragma unroll
                for(int b = 0; b < BINS; ++b)
                {
                    // This bin's B fragment, built just before its WMMAs.
                    frag bfr;
                    if constexpr(!GFX12)
                    {
                        k5u8 f8{};
#pragma unroll
                        for(int q = 0; q < 4; ++q)
                        {
                            f8[q] = v[b][q];
                            f8[4 + q] = k5SwapHalves(v[b][q]);
                        }
                        bfr = __builtin_bit_cast(frag, f8);
                    }
                    else
                    {
                        const k5u4 f4{v[b][0], v[b][1], v[b][2], v[b][3]};
                        bfr = __builtin_bit_cast(frag, f4);
                    }
#pragma unroll
                    for(int m = 0; m < MT; ++m)
                        acc[b][m] = W::mma(ufr[b][m], bfr, acc[b][m]);
                    loadU(nx, b);
                    K5_SCHED_BARRIER();
                }
                transform();
            }
        };
        const bool itemEdge
            = K5_WREG_EDGE
              && ((static_cast<long long>(imgFirst) * a.ci + g * cgi) * chanStride < 16 + a.pw
                  || (static_cast<long long>(imgLast) * a.ci + g * cgi + cgi) * chanStride + 8
                         > inElems);
        if(itemEdge)
            chunks(true);
        else
            chunks(false);

            // Epilogue: right A per row i in registers, Z[i][s] to LDS, then A^T over the rows.
#pragma unroll
        for(int m = 0; m < MT; ++m)
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                // This wave's part of Z0 = M0 + M1 + M2 and Z1 = M1 - M2 - M3.
                float z0 = 0.0f, z1 = 0.0f;
#pragma unroll
                for(int b = 0; b < BINS; ++b)
                {
                    const int j = B0 + b;
                    const float mv = acc[b][m][e];
                    z0 += j < 3 ? mv : 0.0f;
                    z1 += j == 1 ? mv : j >= 2 ? -mv : 0.0f;
                }
                const int kk = m * 16 + W::dRow(lane, e), tt = tg * 16 + col;
                Zs[(((I * 2 + 0) * NJ + J) * BK + kk) * BT + tt] = z0;
                Zs[(((I * 2 + 1) * NJ + J) * BK + kk) * BT + tt] = z1;
            }
        __syncthreads();
        for(int pi = tid; pi < BK * BT; pi += THREADS)
        {
            const int kk = pi / BT, tt = pi % BT, k = k0 + kk, t2 = t0 + tt;
            if(k >= cgo || t2 >= a.tiles)
                continue;
            float z[4][2];
#pragma unroll
            for(int i = 0; i < 4; ++i)
#pragma unroll
                for(int s = 0; s < 2; ++s)
                {
                    z[i][s] = 0.0f;
#pragma unroll
                    for(int j = 0; j < NJ; ++j)
                        z[i][s] += Zs[(((i * 2 + s) * NJ + j) * BK + kk) * BT + tt];
                }
            float y[2][2];
#pragma unroll
            for(int s = 0; s < 2; ++s)
            {
                y[0][s] = z[0][s] + z[1][s] + z[2][s];
                y[1][s] = z[1][s] - z[2][s] - z[3][s];
            }
            const int tw = t2 % a.tw, r1 = t2 / a.tw;
            const int th = r1 % a.th, img = r1 / a.th;
            T* o = out + ((static_cast<long long>(img) * a.co + g * cgo + k) * a.ho + th * 2) * a.wo
                   + tw * 2;
#pragma unroll
            for(int r = 0; r < 2; ++r)
            {
                const int oh = th * 2 + r;
                if(oh >= a.ho)
                    continue;
#pragma unroll
                for(int s = 0; s < 2; ++s)
                {
                    const int ow = tw * 2 + s;
                    if(ow < a.wo)
                        o[r * a.wo + s] = K5Elem<T>::store(k5Live(a, oh, ow) ? y[r][s] : 0.0f);
                }
            }
        }
        __syncthreads();
    }
}

template <class T, bool GFX12, int BK = 64, int TG = 1, bool F16MATH = true, int BINS = 4>
__device__ __attribute__((always_inline)) void k5WregConv(const K5Args& a)
{
    typedef K5WregCfg<T, GFX12, BK, TG, BINS> C;
    __shared__ float Zs[C::LDS];
    // Wave-uniform (row, part): one specialization each.
    const int w = (threadIdx.x / 32) % C::WPG;
    if constexpr(BINS == 4)
    {
        switch(w)
        {
        case 0:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 4, 0, 0>(a, Zs);
            break;
        case 1:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 4, 1, 0>(a, Zs);
            break;
        case 2:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 4, 2, 0>(a, Zs);
            break;
        default:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 4, 3, 0>(a, Zs);
            break;
        }
    }
    else
    {
        switch(w)
        {
        case 0:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 0, 0>(a, Zs);
            break;
        case 1:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 0, 1>(a, Zs);
            break;
        case 2:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 1, 0>(a, Zs);
            break;
        case 3:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 1, 1>(a, Zs);
            break;
        case 4:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 2, 0>(a, Zs);
            break;
        case 5:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 2, 1>(a, Zs);
            break;
        case 6:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 3, 0>(a, Zs);
            break;
        default:
            k5WregBody<T, GFX12, BK, TG, F16MATH, 2, 3, 1>(a, Zs);
            break;
        }
    }
}

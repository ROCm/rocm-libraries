// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3), VALU, register operands (gfx9 wave64; runs on any wave64 target): the
// transformed input never leaves the lane that made it, and the transformed filter is a wave-uniform
// operand read through the scalar cache. NCHW, forward form, fp32 (FMA) and fp16 (v_dot2_f32_f16
// on channel pairs).
//
// Work item = (group, BK out channels, 64 tiles); 4 waves, one per transform row i (bins 4i ..
// 4i+3); lane = tile. Per channel (fp16: channel pair) the lane loads its two input rows (the
// row-i combination of B^T), transforms them into its 4 bins (column pass, then row pass), and
// accumulates acc[bin][k] += U[bin][k][c] * V[bin] over the BK out channels, U being uniform
// across the wave (scalar loads). No LDS and no barrier in the channel loop. Epilogue as in
// ConvWinogradWmmaReg.h: the right A in registers, then A^T over the rows through LDS.
//
// U: the fused pre-pass layout ([G][16][ceil(cgi/VW)][cgo] dwords, out channel fastest), padded
// by the host with >= BK dwords so the uniform loads of a partial K block stay inside it.
#pragma once

#include "ConvWinogradWmmaReg.h" // K5Row and the 16-bit helpers

#ifndef K5_SREG_UNROLL
#define K5_SREG_UNROLL 2
#endif

template <class T, int BK, bool F16MATH, int I>
__device__ __attribute__((always_inline)) void k5SregBody(const K5Args& a, float* Zs)
{
    constexpr bool HALF = __is_same(T, k5_half);
    constexpr int VW = HALF ? 2 : 1, BT = 64, THREADS = 256;
    typedef K5Row<I> R;
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiV = (cgi + VW - 1) / VW;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int chanStride = a.hi * a.wi;
    const long long inElems = static_cast<long long>(a.n) * a.ci * a.hi * a.wi;
    const __amdgpu_buffer_rsrc_t rIn
        = k5Rsrc(a.in, (inElems * static_cast<long long>(sizeof(T)) + 3) / 4 * 4);
    const unsigned* U = static_cast<const unsigned*>(a.u);
    const int tid = threadIdx.x, lane = tid % 64;
    T* out = static_cast<T*>(a.out);
    const unsigned chanBytes = static_cast<unsigned>(chanStride) * sizeof(T);
    const bool csOdd = HALF && (chanStride & 1) != 0;

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int kb = static_cast<int>(item % nKB);
        const long long r0 = item / nKB;
        const int tb = static_cast<int>(r0 % nTB);
        const int g = static_cast<int>(r0 / nTB);
        const int k0 = kb * BK, t0 = tb * BT;
        const int t = t0 + lane;

        // Row offsets (elements, channel 0 of the group), validity, column masks.
        int eA = 0, eB = 0;
        bool rowA = false, rowB = false, cv[4] = {false, false, false, false};
        if(t < a.tiles)
        {
            const int tw = t % a.tw, rr = t / a.tw;
            const int th = rr % a.th, img = rr / a.th;
            const int ih0 = th * 2 - a.ph, iw0 = tw * 2 - a.pw;
            const int base = (img * a.ci + g * cgi) * chanStride + iw0;
            rowA = ih0 + R::RA >= 0 && ih0 + R::RA < a.hi;
            rowB = ih0 + R::RB >= 0 && ih0 + R::RB < a.hi;
            eA = base + (ih0 + R::RA) * a.wi;
            eB = base + (ih0 + R::RB) * a.wi;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                cv[j] = iw0 + j >= 0 && iw0 + j < a.wi;
        }
        const int perImage = a.th * a.tw;
        const int imgFirst = t0 / perImage;
        const int imgLast = (t0 + BT < a.tiles ? t0 + BT - 1 : a.tiles - 1) / perImage;
        // Items within reach of either end of the input read per element (see the WMMA kernel).
        const bool itemEdge
            = (static_cast<long long>(imgFirst) * a.ci + g * cgi) * chanStride < 16 + a.pw
              || (static_cast<long long>(imgLast) * a.ci + g * cgi + cgi) * chanStride + 8
                     > inElems;

        float acc[4][BK];
#pragma unroll
        for(int b = 0; b < 4; ++b)
#pragma unroll
            for(int k = 0; k < BK; ++k)
                acc[b][k] = 0.0f;

        // One input row of one channel: 4 columns, as 4 fp32 (fp32) or 2 half2 dwords (fp16).
        // Raw row: 4 dwords as loaded (fp16 fast path: 3 aligned dwords and the half shift that
        // brings the row's first column to bit 0; finished in the compute stage, so the loads of
        // the next channel stay in flight).
        struct Raw
        {
            unsigned w[4];
            unsigned sh;
        };
        auto loadRow
            = [&](int e, bool ok, int c, bool edge, Raw& raw) __attribute__((always_inline))
        {
            unsigned(&w)[4] = raw.w;
            raw.sh = 0;
            const bool cok = ok && c < cgi;
            if(edge)
            {
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    const long long el
                        = static_cast<long long>(e) + static_cast<long long>(c) * chanStride + j;
                    const unsigned off = cok && el >= 0 ? static_cast<unsigned>(el) * sizeof(T)
                                                        : static_cast<unsigned>(K5_OOB);
                    w[j] = HALF ? __builtin_amdgcn_raw_buffer_load_b16(rIn, off, 0, 0)
                                : __builtin_amdgcn_raw_buffer_load_b32(rIn, off, 0, 0);
                }
                if constexpr(HALF)
                {
                    w[0] = w[0] | (w[1] << 16);
                    w[1] = w[2] | (w[3] << 16);
                    w[2] = 0;
                }
                return;
            }
            // Fast path: never before the buffer's start nor past its end (itemEdge).
            const unsigned so = static_cast<unsigned>(c) * chanBytes;
            if constexpr(HALF)
            {
                typedef unsigned u3 __attribute__((ext_vector_type(3)));
                const unsigned b = static_cast<unsigned>(e) * 2u + (so & 2u);
                const u3 x = __builtin_bit_cast(
                    u3,
                    __builtin_amdgcn_raw_buffer_load_b96(
                        rIn, cok ? (b & ~3u) : static_cast<unsigned>(K5_OOB), so & ~3u, 0));
                raw.sh = (b & 2u) * 8u;
                w[0] = x.x, w[1] = x.y, w[2] = x.z;
            }
            else
            {
                typedef unsigned u4 __attribute__((ext_vector_type(4)));
                const u4 x = __builtin_bit_cast(
                    u4,
                    __builtin_amdgcn_raw_buffer_load_b128(rIn,
                                                          cok ? static_cast<unsigned>(e) * 4u
                                                              : static_cast<unsigned>(K5_OOB),
                                                          so,
                                                          0));
                w[0] = x.x, w[1] = x.y, w[2] = x.z, w[3] = x.w;
            }
        };

        // The 4 bins of row i for channel dword cvi: fp32 values, or half2 (channel pair) dwords.
        struct Fetch
        {
            Raw a[2], b[2]; // [channel of the dword]
        };
        auto fetch = [&](int cvi, bool edge, Fetch& f) __attribute__((always_inline))
        {
            if constexpr(HALF)
            {
                loadRow(eA, rowA, 2 * cvi, edge, f.a[0]);
                loadRow(eB, rowB, 2 * cvi, edge, f.b[0]);
                loadRow(eA, rowA, 2 * cvi + 1, edge, f.a[1]);
                loadRow(eB, rowB, 2 * cvi + 1, edge, f.b[1]);
            }
            else
            {
                loadRow(eA, rowA, cvi, edge, f.a[0]);
                loadRow(eB, rowB, cvi, edge, f.b[0]);
            }
        };
        auto compute = [&](int cvi, const Fetch& f) __attribute__((always_inline))
        {
            unsigned vb[4];
            if constexpr(HALF)
            {
                unsigned ra[2][2], rb[2][2];
#pragma unroll
                for(int e = 0; e < 2; ++e)
                {
                    ra[e][0] = __builtin_amdgcn_alignbit(f.a[e].w[1], f.a[e].w[0], f.a[e].sh);
                    ra[e][1] = __builtin_amdgcn_alignbit(f.a[e].w[2], f.a[e].w[1], f.a[e].sh);
                    rb[e][0] = __builtin_amdgcn_alignbit(f.b[e].w[1], f.b[e].w[0], f.b[e].sh);
                    rb[e][1] = __builtin_amdgcn_alignbit(f.b[e].w[2], f.b[e].w[1], f.b[e].sh);
                }
                const unsigned m01 = (cv[0] ? 0xffffu : 0u) | (cv[1] ? 0xffff0000u : 0u);
                const unsigned m23 = (cv[2] ? 0xffffu : 0u) | (cv[3] ? 0xffff0000u : 0u);
                if constexpr(F16MATH)
                {
                    k5h2 tc[2][2];
#pragma unroll
                    for(int e = 0; e < 2; ++e)
                    {
                        const k5h2 a01 = __builtin_bit_cast(k5h2, ra[e][0]),
                                   a23 = __builtin_bit_cast(k5h2, ra[e][1]);
                        const k5h2 b01 = __builtin_bit_cast(k5h2, rb[e][0]),
                                   b23 = __builtin_bit_cast(k5h2, rb[e][1]);
                        tc[e][0] = __builtin_bit_cast(
                            k5h2,
                            __builtin_bit_cast(unsigned, R::PLUS ? a01 + b01 : a01 - b01) & m01);
                        tc[e][1] = __builtin_bit_cast(
                            k5h2,
                            __builtin_bit_cast(unsigned, R::PLUS ? a23 + b23 : a23 - b23) & m23);
                    }
                    const k5h2 p0 = __builtin_shufflevector(tc[0][0], tc[1][0], 0, 2);
                    const k5h2 p1 = __builtin_shufflevector(tc[0][0], tc[1][0], 1, 3);
                    const k5h2 p2 = __builtin_shufflevector(tc[0][1], tc[1][1], 0, 2);
                    const k5h2 p3 = __builtin_shufflevector(tc[0][1], tc[1][1], 1, 3);
                    vb[0] = __builtin_bit_cast(unsigned, p0 - p2);
                    vb[1] = __builtin_bit_cast(unsigned, p1 + p2);
                    vb[2] = __builtin_bit_cast(unsigned, p2 - p1);
                    vb[3] = __builtin_bit_cast(unsigned, p1 - p3);
                }
                else
                {
                    float o[2][4];
#pragma unroll
                    for(int e = 0; e < 2; ++e)
                    {
                        float tcol[4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            const float af = k5WregHalf<T>(ra[e][j >> 1], j & 1);
                            const float bf = k5WregHalf<T>(rb[e][j >> 1], j & 1);
                            tcol[j] = cv[j] ? (R::PLUS ? af + bf : af - bf) : 0.0f;
                        }
                        o[e][0] = tcol[0] - tcol[2];
                        o[e][1] = tcol[1] + tcol[2];
                        o[e][2] = tcol[2] - tcol[1];
                        o[e][3] = tcol[1] - tcol[3];
                    }
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        vb[j] = k5WregPack<T>(o[0][j], o[1][j]);
                }
            }
            else
            {
                const unsigned(&ra)[4] = f.a[0].w;
                const unsigned(&rb)[4] = f.b[0].w;
                float tcol[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    const float af = __uint_as_float(ra[j]), bf = __uint_as_float(rb[j]);
                    tcol[j] = cv[j] ? (R::PLUS ? af + bf : af - bf) : 0.0f;
                }
                vb[0] = __float_as_uint(tcol[0] - tcol[2]);
                vb[1] = __float_as_uint(tcol[1] + tcol[2]);
                vb[2] = __float_as_uint(tcol[2] - tcol[1]);
                vb[3] = __float_as_uint(tcol[1] - tcol[3]);
            }
            // acc[b][k] += U[g][4i + b][cvi][k0 + k] * V[b]; U is wave-uniform (scalar loads).
            const int uBase
                = __builtin_amdgcn_readfirstlane(((g * 16 + 4 * I) * cgiV + cvi) * cgo + k0);
            const int binStride = cgiV * cgo;
#pragma unroll
            for(int b = 0; b < 4; ++b)
            {
                const unsigned* Ub = U + uBase + b * binStride;
#pragma unroll
                for(int k = 0; k < BK; ++k)
                {
                    const unsigned u = Ub[k];
                    if constexpr(HALF)
                        acc[b][k] = __builtin_amdgcn_fdot2(__builtin_bit_cast(k5h2, u),
                                                           __builtin_bit_cast(k5h2, vb[b]),
                                                           acc[b][k],
                                                           false);
                    else
                        acc[b][k]
                            = __builtin_fmaf(__uint_as_float(u), __uint_as_float(vb[b]), acc[b][k]);
                }
            }
        };

        // One channel of lookahead: the next channel's rows load while this one computes.
        auto channels = [&](bool edge) __attribute__((always_inline))
        {
            Fetch cur;
            fetch(0, edge, cur);
#pragma unroll K5_SREG_UNROLL
            for(int cvi = 0; cvi < cgiV; ++cvi)
            {
                Fetch nxt;
                fetch(cvi + 1 < cgiV ? cvi + 1 : cvi, edge, nxt);
                compute(cvi, cur);
                cur = nxt;
            }
        };
        if(itemEdge)
            channels(true);
        else
            channels(false);

            // Epilogue: right A per row in registers, Z[i][s] to LDS, then A^T over the rows.
#pragma unroll
        for(int k = 0; k < BK; ++k)
        {
            Zs[((I * 2 + 0) * BK + k) * BT + lane] = acc[0][k] + acc[1][k] + acc[2][k];
            Zs[((I * 2 + 1) * BK + k) * BT + lane] = acc[1][k] - acc[2][k] - acc[3][k];
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
                    z[i][s] = Zs[((i * 2 + s) * BK + kk) * BT + tt];
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
                    const float y
                        = r == 0 ? z[0][s] + z[1][s] + z[2][s] : z[1][s] - z[2][s] - z[3][s];
                    if(ow < a.wo)
                        o[r * a.wo + s] = K5Elem<T>::store(k5Live(a, oh, ow) ? y : 0.0f);
                }
            }
        }
        __syncthreads();
    }
}

template <class T, int BK = 16, bool F16MATH = true>
__device__ void k5SregConv(const K5Args& a)
{
    __shared__ float Zs[8 * BK * 64];
    switch(threadIdx.x / 64)
    {
    case 0:
        k5SregBody<T, BK, F16MATH, 0>(a, Zs);
        break;
    case 1:
        k5SregBody<T, BK, F16MATH, 1>(a, Zs);
        break;
    case 2:
        k5SregBody<T, BK, F16MATH, 2>(a, Zs);
        break;
    default:
        k5SregBody<T, BK, F16MATH, 3>(a, Zs);
        break;
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3) on RDNA WMMA (gfx11 / gfx12, wave32), fp16 and bf16, fp32
// accumulate, NCHW, forward and backward data (forward form, see K5Winograd.h).
// HIPRTC-compatible: no standard headers.
//
// Persistent kernel. Work item = (group, BK out channels, 32 tiles). WAVES waves per workgroup, each
// owning 16/WAVES of the 16 Winograd bins; per bin a BK x 32 block of accumulators = (BK/16) x 2
// WMMA 16x16 tiles (fp32, 8 VGPRs each). Default BK = 64, 8 waves: 128 accumulator VGPRs per lane.
// Per C chunk of 16 channels (one WMMA K step):
//   U [bin][k][c 16] (T)   - copied from the pre-transformed filter (16-byte loads, prefetched);
//   V [bin][t 32][c 16] (T) - input transform, 4 lanes per 4x4 patch (column pass per lane, row
//                             pass via DPP quad_perm), channel pairs packed, rounded once to T.
//   M[bin] += U[bin] (BK x 16) * V[bin] (16 x 32).
// Epilogue: accumulators -> LDS M fp32 (32 out channels per pass), A^T M A -> y with the parity mask.
//
// Filter pre-pass (k5WmmaFilterTransform): U[G][16][cgo][cgiPad] (T), cgiPad = ceil(cgi/16)*16,
// zero-padded channels. Workspace = G * 16 * cgo * cgiPad * sizeof(T).
#pragma once

#include "ConvWinogradFused.h"

#ifndef K5_WMMA_UPF
#define K5_WMMA_UPF 4
#endif

// WMMA builtins exist only in wave32 builds; a wave64 test build of gfx11 (-mwavefrontsize64)
// defines K5_WAVE64_BUILD.
#if defined(K5_WAVE64_BUILD)
#define K5_WAVE32 0
#else
#define K5_WAVE32 1
#endif

template <class T>
struct K5WmmaElem; // fragment element type of the builtin
template <>
struct K5WmmaElem<k5_half>
{
    typedef _Float16 e;
};
template <>
struct K5WmmaElem<k5_bf16>
{
    typedef short e;
};

template <class T, int N>
using k5wv = typename K5WmmaElem<T>::e __attribute__((ext_vector_type(N)));
typedef float k5f8 __attribute__((ext_vector_type(8)));

// D = A(16x16) * B(16x16) + C. gfx11: a lane holds all 16 k of row/col lane%16; D element i is
// row 2i + lane/16. gfx12: a lane holds k (lane/16)*8 .. +7; D element i is row (lane/16)*8 + i.
template <class T, bool GFX12>
struct K5Wmma
{
    static constexpr int KPL = GFX12 ? 8 : 16;
    typedef k5wv<T, KPL> frag;
    __device__ static int dRow(int lane, int i)
    {
        return GFX12 ? (lane >> 4) * 8 + i : 2 * i + (lane >> 4);
    }
    __device__ static int kBase(int lane)
    {
        return GFX12 ? (lane >> 4) * 8 : 0;
    }
    __device__ static k5f8 mma(frag a, frag b, k5f8 c)
    {
        // The WMMA families, not __GFX12__ / __GFX11__: those also cover gfx1250 and gfx1170,
        // which lack these builtins.
#if HKP_ARCH_GFX12 && K5_WAVE32
        if constexpr(GFX12)
        {
            if constexpr(__is_same(T, k5_half))
                return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
            else
                return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
        }
#elif HKP_ARCH_GFX11 && K5_WAVE32
        if constexpr(!GFX12)
        {
            if constexpr(__is_same(T, k5_half))
                return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
            else
                return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
        }
#endif
        __builtin_trap();
        return c;
    }
};

// Filter pre-pass. One thread per (g, o, i < cgiPad), i fastest.
template <class T, int LAYOUT, int DIR>
__device__ void k5WmmaFilterTransform(const K5Args& a)
{
    constexpr int P = 16;
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= static_cast<long long>(a.groups) * cgo * cgiPad)
        return;
    const int i = static_cast<int>(idx % cgiPad);
    const int o = static_cast<int>((idx / cgiPad) % cgo);
    const int g = static_cast<int>(idx / (static_cast<long long>(cgiPad) * cgo));
    const int cgf = DIR == K5_FORWARD ? cgi : cgo;
    const bool rot = (DIR == K5_BACKWARD_DATA) != (a.flip != 0);
    const T* w = static_cast<const T*>(a.w);
    float f[9], u[P];
#pragma unroll
    for(int q = 0; q < 9; ++q)
        f[q] = 0.0f;
    if(i < cgi)
    {
        const int kf = DIR == K5_FORWARD ? g * cgo + o : g * cgi + i;
        const int cf = DIR == K5_FORWARD ? i : o;
#pragma unroll
        for(int q = 0; q < 9; ++q)
        {
            const int rs = rot ? 8 - q : q;
            const long long off = LAYOUT == K5_NCHW
                                      ? (static_cast<long long>(kf) * cgf + cf) * 9 + rs
                                      : (static_cast<long long>(kf) * 9 + rs) * cgf + cf;
            f[q] = K5Elem<T>::load(w[off]);
        }
    }
    k5FilterTile<2>(f, u);
    T* U = static_cast<T*>(a.u);
    const long long ps = static_cast<long long>(cgo) * cgiPad;
    const long long base
        = static_cast<long long>(g) * P * ps + static_cast<long long>(o) * cgiPad + i;
#pragma unroll
    for(int p = 0; p < P; ++p)
        U[base + p * ps] = K5Elem<T>::store(u[p]);
}

// Main kernel. BK = out channels per item (32 or 64), WAVES = waves per workgroup (4 or 8; each
// wave owns 16/WAVES bins). BT = 32 tiles, C chunk = 16 channels (one WMMA K step).
// LDS (halves): U [bin][k BK][c 16] single-buffered (refilled from registers between two barriers),
// V [bin][t 32][c 16] double-buffered (the next chunk's transform overlaps the WMMAs).
template <class T, bool GFX12, int BK = 64, int WAVES = 8>
struct K5WmmaCfg
{
    static constexpr int P = 16, BT = 32, CC = 16, THREADS = 32 * WAVES, BPW = P / WAVES;
    static constexpr int MT = BK / 16, NT = BT / 16;
    static constexpr int UB = P * BK * CC, VB = P * BT * CC; // halves
    static constexpr int LDS
        = ((UB + 2 * VB) / 2 > P * 32 * BT ? (UB + 2 * VB) / 2 : P * 32 * BT); // dwords
    static constexpr int NUQ = UB / 8 / THREADS; // 16-byte U slots per lane
    static constexpr int EPASS = BK / 32; // epilogue passes (32 out channels each)
    static_assert(P % WAVES == 0 && (THREADS / 2) % BK == 0 && UB % (8 * THREADS) == 0, "config");
};

template <class T, bool GFX12, int BK = 64, int WAVES = 8>
__device__ void k5WmmaConv(const K5Args& a)
{
    typedef K5Wmma<T, GFX12> W;
    typedef K5WmmaCfg<T, GFX12, BK, WAVES> C;
    constexpr int P = C::P, BT = C::BT, CC = C::CC, THREADS = C::THREADS, BPW = C::BPW, MT = C::MT,
                  NT = C::NT;
    constexpr int UB = C::UB, VB = C::VB, NUQ = C::NUQ;
    __shared__ __attribute__((aligned(16))) unsigned lds[C::LDS];
    unsigned short* Us = reinterpret_cast<unsigned short*>(lds);
    unsigned short* Vbase = Us + UB;

    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int nCh = cgiPad / CC;
    const int chanStride = a.hi * a.wi;
    const __amdgpu_buffer_rsrc_t rIn = k5Rsrc(
        a.in, static_cast<long long>(a.n) * a.ci * a.hi * a.wi * static_cast<long long>(sizeof(T)));
    const __amdgpu_buffer_rsrc_t rU = k5Rsrc(a.u,
                                             static_cast<long long>(a.groups) * P * cgo * cgiPad
                                                 * static_cast<long long>(sizeof(T)));
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    T* out = static_cast<T*>(a.out);

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int kb = static_cast<int>(item % nKB);
        const long long r0 = item / nKB;
        const int tb = static_cast<int>(r0 % nTB);
        const int g = static_cast<int>(r0 / nTB);
        const int k0 = kb * BK, t0 = tb * BT;

        // U slot q: 16-byte piece e = tid + q*THREADS -> row e/2 (bin p = row / BK, k = row % BK), half
        // e % 2. With (THREADS/2) % BK == 0, k and the half are lane constants and p steps uniformly.
        const int uK = (tid / 2) % BK, uHf = tid % 2, uP0 = (tid / 2) / BK;
        constexpr int UPQ = (THREADS / 2) / BK; // bins per slot step
        const bool uOk = k0 + uK < cgo;
        uint4 ureg[NUQ];
        auto loadU = [&](int ch) {
            const unsigned off
                = uOk ? static_cast<unsigned>(
                            ((((g * P + uP0) * cgo + k0 + uK) * cgiPad + ch * CC) * 2) + uHf * 16)
                      : static_cast<unsigned>(K5_OOB);
#pragma unroll
            for(int q = 0; q < NUQ; ++q)
                ureg[q] = __builtin_bit_cast(
                    uint4,
                    __builtin_amdgcn_raw_buffer_load_b128(
                        rU, off, static_cast<unsigned>(q * UPQ * cgo * cgiPad * 2), 0));
        };
        auto storeU = [&]() {
#pragma unroll
            for(int q = 0; q < NUQ; ++q)
            {
                const int e = tid + q * THREADS;
                *reinterpret_cast<uint4*>(Us + (e / 2) * CC + (e % 2) * 8) = ureg[q];
            }
        };

        // Input transform: a quad of lanes per 4x4 patch and channel pair; the lane's tile is fixed,
        // rounds walk the channel pairs. Each lane writes one half2 word per output row.
        constexpr int QPR = THREADS / 4 / BT; // channel pairs per round
        const int j = lane & 3, tl = (tid / 4) % BT, cp0 = (tid / 4) / BT, t = t0 + tl;
        int vBase = 0;
        unsigned vRowOk = 0;
        bool vColOk = false;
        if(t < a.tiles)
        {
            const int tw = t % a.tw, rr = t / a.tw;
            const int th = rr % a.th, img = rr / a.th;
            const int ih0 = th * 2 - a.ph, iw = tw * 2 - a.pw + j;
            vColOk = iw >= 0 && iw < a.wi;
            vBase = ((img * a.ci + g * cgi) * a.hi + ih0) * a.wi + iw;
#pragma unroll
            for(int i = 0; i < 4; ++i)
                vRowOk |= (ih0 + i >= 0 && ih0 + i < a.hi) ? (1u << i) : 0u;
        }
        auto stageV = [&](int ch, int buf) {
            unsigned* Vw = reinterpret_cast<unsigned*>(Vbase + buf * VB);
            const float sgn = j == 1 ? 1.0f : -1.0f;
#pragma unroll 1
            for(int r = 0; r < CC / 2 / QPR; ++r)
            {
                const int cp = r * QPR + cp0;
                float o[2][4];
#pragma unroll
                for(int v = 0; v < 2; ++v)
                {
                    const int c = ch * CC + cp * 2 + v;
                    const bool ok = vColOk && c < cgi;
                    float d[4], tcol[4];
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                    {
                        const bool rok = ok && ((vRowOk >> i) & 1u);
                        d[i] = k5BufLoad<T>(
                            rIn,
                            rok ? static_cast<unsigned>((vBase + c * chanStride + i * a.wi)
                                                        * static_cast<int>(sizeof(T)))
                                : static_cast<unsigned>(K5_OOB),
                            0);
                    }
                    K5Wino<2>::bt(d, 1, tcol, 1);
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                    {
                        const float pv = __int_as_float(__builtin_amdgcn_update_dpp(
                            0, __float_as_int(tcol[i]), 0x64, 0xF, 0xF, true));
                        const float qv = __int_as_float(__builtin_amdgcn_update_dpp(
                            0, __float_as_int(tcol[i]), 0xDA, 0xF, 0xF, true));
                        o[v][i] = __builtin_fmaf(sgn, qv, pv);
                    }
                }
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    const unsigned lo
                        = __builtin_bit_cast(unsigned short, K5Elem<T>::store(o[0][i]));
                    const unsigned hi
                        = __builtin_bit_cast(unsigned short, K5Elem<T>::store(o[1][i]));
                    Vw[(((i * 4 + j) * BT + tl) * CC) / 2 + cp] = lo | (hi << 16);
                }
            }
        };

        k5f8 acc[BPW][MT][NT];
#pragma unroll
        for(int b = 0; b < BPW; ++b)
#pragma unroll
            for(int m = 0; m < MT; ++m)
#pragma unroll
                for(int n = 0; n < NT; ++n)
                    acc[b][m][n] = k5f8{0, 0, 0, 0, 0, 0, 0, 0};

        loadU(0);
        storeU();
        stageV(0, 0);
        __syncthreads();
        for(int ch = 0; ch < nCh; ++ch)
        {
            const bool more = ch + 1 < nCh;
            if(more)
                loadU(ch + 1);
            const unsigned short* Vs = Vbase + (ch & 1) * VB;
#pragma unroll
            for(int b = 0; b < BPW; ++b)
            {
                const int bin = wave * BPW + b;
                typename W::frag af[MT], bf[NT];
#pragma unroll
                for(int m = 0; m < MT; ++m)
                    af[m] = *reinterpret_cast<const typename W::frag*>(
                        Us + (bin * BK + m * 16 + lane % 16) * CC + W::kBase(lane));
#pragma unroll
                for(int n = 0; n < NT; ++n)
                    bf[n] = *reinterpret_cast<const typename W::frag*>(
                        Vs + (bin * BT + n * 16 + lane % 16) * CC + W::kBase(lane));
#pragma unroll
                for(int m = 0; m < MT; ++m)
#pragma unroll
                    for(int n = 0; n < NT; ++n)
                        acc[b][m][n] = W::mma(af[m], bf[n], acc[b][m][n]);
                // One bin's fragments live at a time.
                asm volatile("" ::: "memory");
            }
            if(more)
                stageV(ch + 1, (ch + 1) & 1);
            __syncthreads(); // all WMMAs on U done; V(ch+1) complete
            if(more)
            {
                storeU();
                __syncthreads();
            }
        }

        // Epilogue: per pass, 32 out channels: M[bin][k 32][t 32] fp32 in LDS (64 KiB), then A^T M A.
        float* Ms = reinterpret_cast<float*>(lds);
#pragma unroll
        for(int e = 0; e < C::EPASS; ++e)
        {
            if(e > 0)
                __syncthreads();
#pragma unroll
            for(int b = 0; b < BPW; ++b)
#pragma unroll
                for(int m = 0; m < 2; ++m)
#pragma unroll
                    for(int n = 0; n < NT; ++n)
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                            Ms[((wave * BPW + b) * 32 + m * 16 + W::dRow(lane, i)) * BT + n * 16
                               + lane % 16]
                                = acc[b][e * 2 + m][n][i];
            __syncthreads();
            for(int pi = tid; pi < 32 * BT; pi += THREADS)
            {
                const int kk = pi / BT, tt = pi % BT, k = k0 + e * 32 + kk, t2 = t0 + tt;
                if(k >= cgo || t2 >= a.tiles)
                    continue;
                float mm[P], tmp[8], y[4];
#pragma unroll
                for(int p = 0; p < P; ++p)
                    mm[p] = Ms[(p * 32 + kk) * BT + tt];
#pragma unroll
                for(int jj = 0; jj < 4; ++jj)
                    K5Wino<2>::at(mm + jj, 4, tmp + jj, 4);
#pragma unroll
                for(int i = 0; i < 2; ++i)
                    K5Wino<2>::at(tmp + i * 4, 1, y + i * 2, 1);
                const int tw = t2 % a.tw, r1 = t2 / a.tw;
                const int th = r1 % a.th, img = r1 / a.th;
                T* o = out
                       + ((static_cast<long long>(img) * a.co + g * cgo + k) * a.ho + th * 2) * a.wo
                       + tw * 2;
#pragma unroll
                for(int i = 0; i < 2; ++i)
                {
                    const int oh = th * 2 + i;
                    if(oh >= a.ho)
                        continue;
#pragma unroll
                    for(int jj = 0; jj < 2; ++jj)
                    {
                        const int ow = tw * 2 + jj;
                        if(ow < a.wo)
                            o[i * a.wo + jj]
                                = K5Elem<T>::store(k5Live(a, oh, ow) ? y[i * 2 + jj] : 0.0f);
                    }
                }
            }
        }
        __syncthreads();
    }
}

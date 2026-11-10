// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3) on RDNA WMMA (gfx11 / gfx12, wave32), fp16 and bf16, fp32 accumulate, NCHW,
// forward form (backward data runs it on the rotated filter). The cooperative variant: the whole
// block transforms each 16-channel chunk of input once into LDS, and every wave reads its B
// fragments from there, so one transformed value feeds all of the block's out channels.
//
// On RDNA, VALU and WMMA share the SIMD: every VALU cycle is a cycle without WMMA. The input
// transform costs about 64 VALU per (tile, channel pair) and chunk; spread over BK out channels it
// is a few VALU per WMMA.
//
// Block: 4 * KW waves; wave w owns transform row i = w % 4 (bins 4i .. 4i+3) and out channels
// k0 + (w / 4) * 16 * MT .. + 16 * MT; BT = 16 * NT tiles. Item = (group, BK = 16 * MT * KW out
// channels, BT tiles).
//   - A fragments (transformed filter U) come straight from the pre-pass buffer, prefetched one
//     chunk ahead bin by bin into the registers just consumed.
//   - Input: a lane owns one tile and one channel pair: 4 rows x 2 channels as 12-byte loads,
//     realigned, column pass, channel pairing, row pass (fp16 packed math, or fp32 for bf16), 16
//     dword LDS stores V[bin][tile][channel]. Double-buffered: chunk ch + 1 is transformed after
//     chunk ch's WMMAs; one barrier per chunk.
//   - Epilogue: per 16-channel slab, the wave folds its row's 4 bins to 2 values in registers
//     (Z = M A), Z goes to LDS, then A^T over the 4 rows gives the 2x2 outputs.
// Filter pre-pass: k5WmmaFilterTransform (U[G][16][cgo][cgiPad] T).
#pragma once

#include "ConvWinogradWmmaReg.h"

// Out-of-range voffset: the tensor is below 2 GiB, so voffset + soffset never wraps into range.
#define K5C_OOB 0x7fffffff

// Workgroup barrier that orders LDS only: __syncthreads() also waits for every outstanding global
// load (vmcnt(0)), which would expose the prefetched U and input loads at every chunk.
__device__ inline void k5cLdsBarrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <class T, bool GFX12, int MT, int NT, int KW>
struct K5CoopCfg
{
    static constexpr int BK = 16 * MT * KW, BT = 16 * NT, WAVES = 4 * KW, THREADS = 32 * WAVES;
    static constexpr int VB = 16 * BT * 16; // halves per V buffer: [bin][tile][16 channels]
    static constexpr int JW = 4 * NT; // waves of (tile, channel pair) jobs per chunk
    static constexpr int JT = JW > WAVES ? JW / WAVES : 1; // jobs per thread
    static constexpr int KWP = KW < 2 ? KW : 2; // K wave groups per epilogue pass
    static constexpr int ZB = 4 * 2 * 16 * KWP * BT; // floats per epilogue pass
    static constexpr int LDSB = 2 * VB * 2 > ZB * 4 ? 2 * VB * 2 : ZB * 4; // bytes
    static constexpr int FD = GFX12 ? 4 : 8; // dwords per fragment
    // Waves per SIMD the kernel is built for (VGPR budget 1536 / WPE): the loop is latency-bound,
    // so the single-tile configs hold 8 (192 VGPRs); 2 x 2 tiles need the registers.
    static constexpr int WPE = MT * NT <= 2 ? 8 : 4;
    static_assert(JW % WAVES == 0 || WAVES % JW == 0, "job split");
};

template <class T, bool GFX12, int MT, int NT, int KW, bool F16MATH>
__device__ __attribute__((always_inline)) void k5CoopConv(const K5Args& a)
{
    typedef K5Wmma<T, GFX12> W;
    typedef K5CoopCfg<T, GFX12, MT, NT, KW> C;
    typedef typename W::frag frag;
    constexpr int BK = C::BK, BT = C::BT, THREADS = C::THREADS, VB = C::VB, JT = C::JT, JW = C::JW;
    constexpr bool PACKED = F16MATH && __is_same(T, k5_half);
    __shared__ __attribute__((aligned(16))) unsigned char ldsRaw[C::LDSB];
    unsigned* Vd = reinterpret_cast<unsigned*>(ldsRaw); // two buffers of VB halves
    float* Zs = reinterpret_cast<float*>(ldsRaw);

    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int nCh = cgiPad / 16;
    const int chanStride = a.hi * a.wi;
    const long long inElems = static_cast<long long>(a.n) * a.ci * a.hi * a.wi;
    const __amdgpu_buffer_rsrc_t rIn
        = k5Rsrc(a.in, (inElems * static_cast<long long>(sizeof(T)) + 3) / 4 * 4);
    const __amdgpu_buffer_rsrc_t rU = k5Rsrc(a.u,
                                             static_cast<long long>(a.groups) * 16 * cgo * cgiPad
                                                 * static_cast<long long>(sizeof(T)));
    const int tid = threadIdx.x, lane = tid % 32;
    // Wave-uniform role.
    const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
    const int row = wave % 4, kw = wave / 4;
    const int col = lane & 15;
    T* out = static_cast<T*>(a.out);
    const unsigned chunkBytes = static_cast<unsigned>(16 * chanStride) * sizeof(T);

    // A fragment lane offset (bytes) within a U row block: k row = col, channels kBase.
    const unsigned uLane = static_cast<unsigned>(col * cgiPad + W::kBase(lane)) * sizeof(T);
    const unsigned binBytes = static_cast<unsigned>(cgo) * cgiPad * sizeof(T);

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        const int kb = static_cast<int>(item % nKB);
        const long long r0 = item / nKB;
        const int tb = static_cast<int>(r0 % nTB);
        const int g = static_cast<int>(r0 / nTB);
        const int k0 = kb * BK, t0 = tb * BT;

        // ---- Transform jobs: lane -> (tile, channel pair). Job wave jw covers tiles
        // 16 * (jw % NT) + col and channel pair 2 * (jw / NT) + lane / 16.
        int voff[JT][4][2]; // aligned byte offsets of the 12-byte row loads (chunk 0; may be < 0)
        unsigned shf[JT]; // per (row, channel) half shift bits: bit 2 * r + v
        unsigned selc[JT][4]; // channel-pairing perm selectors per column (0x0c: padding)
        int tl[JT], cp[JT];
        bool jobOn[JT];
        bool anyNeg = false;
#pragma unroll
        for(int q = 0; q < JT; ++q)
        {
            const int jw = JT > 1 ? wave + q * C::WAVES : wave;
            jobOn[q] = jw < JW;
            tl[q] = 16 * (jw % NT) + col;
            cp[q] = 2 * (jw / NT) + (lane >> 4);
            shf[q] = 0;
#pragma unroll
            for(int r = 0; r < 4; ++r)
                voff[q][r][0] = voff[q][r][1] = K5C_OOB;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                selc[q][j] = 0x0c0c0c0cu;
            const int t = t0 + tl[q];
            if(jobOn[q] && t < a.tiles)
            {
                const int tw = t % a.tw, rr = t / a.tw;
                const int th = rr % a.th, img = rr / a.th;
                const int ih0 = th * 2 - a.ph, iw0 = tw * 2 - a.pw;
#pragma unroll
                for(int v = 0; v < 2; ++v)
                {
                    const int c = 2 * cp[q] + v;
                    const long long cbase
                        = (static_cast<long long>(img) * a.ci + g * cgi + c) * chanStride;
#pragma unroll
                    for(int r = 0; r < 4; ++r)
                    {
                        const int ih = ih0 + r;
                        if(ih < 0 || ih >= a.hi || c >= cgi)
                            continue;
                        const long long e = cbase + static_cast<long long>(ih) * a.wi + iw0;
                        // Only the first row of the tensor with left padding starts before it.
                        anyNeg = anyNeg || e < 0;
                        voff[q][r][v] = static_cast<int>((e & ~1LL) * 2);
                        shf[q] |= static_cast<unsigned>(e & 1) << (2 * r + v);
                    }
                }
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    const int iw = iw0 + j;
                    if(iw >= 0 && iw < a.wi)
                        selc[q][j] = (j & 1) ? 0x07060302u : 0x05040100u;
                }
            }
        }
        // The slow path for the rare lanes whose first row element is before the tensor start:
        // per-element loads (chunk 0 only; later chunks are 16 planes further on).
        const bool negWave = __builtin_amdgcn_ballot_w32(anyNeg) != 0;

        typedef unsigned k5u3 __attribute__((ext_vector_type(3)));
        k5u3 raw[JT][4][2]; // the next chunk's input
        // negWave: the offsets of some lane are negative in chunk 0 (that chunk goes through the
        // slow path); later chunks add the chunk offset per lane.
        // Branch-free: the per-lane offset carries the chunk offset (a lane's chunk-0 offset may be
        // negative at the start of the tensor; that chunk goes through the slow path). Branches
        // here made the compiler wait for all loads before each one.
        auto loadIn = [&](int ch) __attribute__((always_inline))
        {
            const unsigned so = static_cast<unsigned>(ch) * chunkBytes;
#pragma unroll
            for(int q = 0; q < JT; ++q)
#pragma unroll
                for(int v = 0; v < 2; ++v)
                {
                    const bool cok = ch * 16 + 2 * cp[q] + v < cgi;
#pragma unroll
                    for(int r = 0; r < 4; ++r)
                    {
                        const bool live = cok && voff[q][r][v] != K5C_OOB;
                        unsigned vo = live ? static_cast<unsigned>(voff[q][r][v]) + so
                                           : static_cast<unsigned>(K5C_OOB);
                        // Opaque: keeps one load after a select (else the compiler branches per
                        // load and waits for every outstanding load in each branch).
                        asm volatile("" : "+v"(vo));
                        raw[q][r][v] = __builtin_bit_cast(
                            k5u3, __builtin_amdgcn_raw_buffer_load_b96(rIn, vo, 0, 0));
                    }
                }
        };
        // Per-element reload of chunk 0 for waves holding a lane before the tensor start.
        auto loadInSlow = [&]() __attribute__((always_inline))
        {
#pragma unroll
            for(int q = 0; q < JT; ++q)
            {
                const int t = t0 + tl[q];
                if(!(jobOn[q] && t < a.tiles))
                    continue;
                const int tw = t % a.tw, rr = t / a.tw;
                const int th = rr % a.th, img = rr / a.th;
                const int ih0 = th * 2 - a.ph, iw0 = tw * 2 - a.pw;
#pragma unroll
                for(int v = 0; v < 2; ++v)
                {
                    const int c = 2 * cp[q] + v;
#pragma unroll
                    for(int r = 0; r < 4; ++r)
                    {
                        const int ih = ih0 + r;
                        const bool rok = ih >= 0 && ih < a.hi && c < cgi;
                        unsigned hv[4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            const int iw = iw0 + j;
                            const long long e
                                = ((static_cast<long long>(img) * a.ci + g * cgi + c) * a.hi + ih)
                                      * a.wi
                                  + iw;
                            hv[j] = __builtin_amdgcn_raw_buffer_load_b16(
                                rIn,
                                rok && iw >= 0 && iw < a.wi
                                    ? static_cast<unsigned>(e * static_cast<long long>(sizeof(T)))
                                    : static_cast<unsigned>(K5C_OOB),
                                0,
                                0);
                        }
                        // Already aligned: shift 0.
                        raw[q][r][v] = k5u3{hv[0] | (hv[1] << 16), hv[2] | (hv[3] << 16), 0u};
                    }
                }
            }
        };

        // Transform the raw chunk into V buffer buf.
        auto transform = [&](int buf, bool aligned) __attribute__((always_inline))
        {
            unsigned* Vw = Vd + buf * (VB / 2);
#pragma unroll
            for(int q = 0; q < JT; ++q)
            {
                if(JT == 1 && JW < C::WAVES && !jobOn[q])
                    continue; // uniform: idle job wave
                // Realign: cols (0,1) and (2,3) of each (row, channel).
                unsigned x[4][2][2];
#pragma unroll
                for(int r = 0; r < 4; ++r)
#pragma unroll
                    for(int v = 0; v < 2; ++v)
                    {
                        const unsigned s = aligned ? 0u : ((shf[q] >> (2 * r + v)) & 1u) * 16u;
                        const k5u3 d = raw[q][r][v];
                        x[r][v][0] = __builtin_amdgcn_alignbit(d.y, d.x, s);
                        x[r][v][1] = __builtin_amdgcn_alignbit(d.z, d.y, s);
                    }
                unsigned vout[16];
                if constexpr(PACKED)
                {
                    // Column pass (B^T rows) on the packed column pairs, per channel.
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                    {
                        const int ra = i == 0 ? 0 : i == 2 ? 2 : 1;
                        const int rb = i == 0 ? 2 : i == 1 ? 2 : i == 2 ? 1 : 3;
                        const bool plus = i == 1;
                        unsigned tcol[2][2];
#pragma unroll
                        for(int v = 0; v < 2; ++v)
#pragma unroll
                            for(int h2 = 0; h2 < 2; ++h2)
                            {
                                const k5h2 av = k5AsH2(x[ra][v][h2]), bv = k5AsH2(x[rb][v][h2]);
                                tcol[v][h2] = k5AsU(plus ? av + bv : av - bv);
                            }
                        // Pair the channels per column (padding columns -> 0).
                        k5h2 p[4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                            p[j] = k5AsH2(__builtin_amdgcn_perm(
                                tcol[1][j >> 1], tcol[0][j >> 1], selc[q][j]));
                        // Row pass (B on the right).
                        vout[4 * i + 0] = k5AsU(p[0] - p[2]);
                        vout[4 * i + 1] = k5AsU(p[1] + p[2]);
                        vout[4 * i + 2] = k5AsU(p[2] - p[1]);
                        vout[4 * i + 3] = k5AsU(p[1] - p[3]);
                    }
                }
                else
                {
                    float f[2][4][4]; // [channel][row][col]
#pragma unroll
                    for(int v = 0; v < 2; ++v)
#pragma unroll
                        for(int r = 0; r < 4; ++r)
#pragma unroll
                            for(int j = 0; j < 4; ++j)
                            {
                                const bool cok = selc[q][j] != 0x0c0c0c0cu;
                                const float e = k5WregHalf<T>(x[r][v][j >> 1], (j & 1) != 0);
                                f[v][r][j] = cok ? e : 0.0f;
                            }
                    float o[2][16];
#pragma unroll
                    for(int v = 0; v < 2; ++v)
                    {
                        float tc[4][4];
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            tc[0][j] = f[v][0][j] - f[v][2][j];
                            tc[1][j] = f[v][1][j] + f[v][2][j];
                            tc[2][j] = f[v][2][j] - f[v][1][j];
                            tc[3][j] = f[v][1][j] - f[v][3][j];
                        }
#pragma unroll
                        for(int i = 0; i < 4; ++i)
                        {
                            o[v][4 * i + 0] = tc[i][0] - tc[i][2];
                            o[v][4 * i + 1] = tc[i][1] + tc[i][2];
                            o[v][4 * i + 2] = tc[i][2] - tc[i][1];
                            o[v][4 * i + 3] = tc[i][1] - tc[i][3];
                        }
                    }
#pragma unroll
                    for(int b = 0; b < 16; ++b)
                        vout[b] = k5WregPack<T>(o[0][b], o[1][b]);
                }
                // V[bin][tile][channel]: dword (bin * BT + tile) * 8 + channel pair.
#pragma unroll
                for(int b = 0; b < 16; ++b)
                    Vw[(b * BT + tl[q]) * 8 + cp[q]] = vout[b];
            }
        };

        k5f8 acc[4][MT][NT];
#pragma unroll
        for(int b = 0; b < 4; ++b)
#pragma unroll
            for(int m = 0; m < MT; ++m)
#pragma unroll
                for(int n = 0; n < NT; ++n)
                    acc[b][m][n] = k5f8{0, 0, 0, 0, 0, 0, 0, 0};

        // A fragments for this wave: bins 4 * row + b, out channels kRow0 + m * 16 + col.
        const int kRow0 = k0 + kw * 16 * MT;
        const unsigned uWave
            = static_cast<unsigned>((g * 16 + 4 * row) * cgo + kRow0) * cgiPad * sizeof(T);
        frag ufr[4][MT];
        auto loadU = [&](int ch, int b) __attribute__((always_inline))
        {
#pragma unroll
            for(int m = 0; m < MT; ++m)
            {
                const unsigned so = uWave + b * binBytes
                                    + static_cast<unsigned>(m * 16 * cgiPad + ch * 16) * sizeof(T);
                if constexpr(GFX12)
                    ufr[b][m] = __builtin_bit_cast(
                        frag, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane, so, 0));
                else
                {
                    const k5u4 lo = __builtin_bit_cast(
                        k5u4, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane, so, 0));
                    const k5u4 hi = __builtin_bit_cast(
                        k5u4, __builtin_amdgcn_raw_buffer_load_b128(rU, uLane + 16, so, 0));
                    ufr[b][m] = __builtin_bit_cast(
                        frag, k5u8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
                }
            }
        };
        // B fragment: tile n * 16 + col of bin, channels kBase.
        auto loadB = [&](const unsigned* Vs, int bin, int n) __attribute__((always_inline))
        {
            const unsigned* p = Vs + ((bin * BT + n * 16 + col) * 16 + W::kBase(lane)) / 2;
            if constexpr(GFX12)
                return __builtin_bit_cast(frag, *reinterpret_cast<const k5u4*>(p));
            else
            {
                const k5u4 lo = *reinterpret_cast<const k5u4*>(p);
                const k5u4 hi = *reinterpret_cast<const k5u4*>(p + 4);
                return __builtin_bit_cast(frag,
                                          k5u8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
            }
        };

        // Prologue: chunk 0 in, transformed into buffer 0; U of chunk 0.
        loadIn(0);
        if(negWave)
            loadInSlow();
#pragma unroll
        for(int b = 0; b < 4; ++b)
            loadU(0, b);
        transform(0, negWave);
        __syncthreads();
        // Chunk ch (parity P): V buffer P holds it; load chunk ch + 1's input, run the WMMAs, then
        // transform chunk ch + 1 into V buffer 1 - P.
        auto step = [&](int ch, auto parTag) __attribute__((always_inline))
        {
            constexpr int P = decltype(parTag)::value;
            const int last = nCh - 1;
            const int nx = ch + 1 < last ? ch + 1 : last; // the last chunk reloads itself
            loadIn(nx);
            K5_SCHED_BARRIER();
            const unsigned* Vs = Vd + P * (VB / 2);
#pragma unroll
            for(int b = 0; b < 4; ++b)
            {
                frag bfr[NT];
#pragma unroll
                for(int n = 0; n < NT; ++n)
                    bfr[n] = loadB(Vs, 4 * row + b, n);
#pragma unroll
                for(int m = 0; m < MT; ++m)
#pragma unroll
                    for(int n = 0; n < NT; ++n)
                        acc[b][m][n] = W::mma(ufr[b][m], bfr[n], acc[b][m][n]);
                loadU(nx, b);
                K5_SCHED_BARRIER();
            }
            if(ch + 1 < nCh)
                transform(1 - P, false);
            k5cLdsBarrier();
        };
        for(int ch = 0; ch < nCh; ch += 2)
        {
            step(ch, K5Int<0>{});
            if(ch + 1 < nCh)
                step(ch + 1, K5Int<1>{});
        }

        // Epilogue, one 16-channel slab (m) at a time: Z[i][s][kw * 16 + kk][t] in LDS.
        // This thread's epilogue tile (fixed per item): output base, per-position live bits (bit
        // 2r + s), row bits (4 + r), second column in range (6), and the paired-store case.
        const int eTt = tid % BT;
        const int eT2 = t0 + eTt;
        const bool eTile = eT2 < a.tiles;
        long long eBase = 0;
        const long long eChan = static_cast<long long>(a.ho) * a.wo;
        unsigned eMask = 0;
        bool ePair = false;
        if(eTile)
        {
            const int tw = eT2 % a.tw, r1 = eT2 / a.tw;
            const int th = r1 % a.th, img = r1 / a.th;
            eBase = (static_cast<long long>(img) * a.co + g * cgo) * eChan
                    + static_cast<long long>(th * 2) * a.wo + tw * 2;
#pragma unroll
            for(int r = 0; r < 2; ++r)
            {
                const int oh = th * 2 + r;
                if(oh < a.ho)
                    eMask |= 1u << (4 + r);
#pragma unroll
                for(int s = 0; s < 2; ++s)
                {
                    const int ow = tw * 2 + s;
                    if(oh < a.ho && ow < a.wo && k5Live(a, oh, ow))
                        eMask |= 1u << (2 * r + s);
                }
            }
            if(tw * 2 + 1 < a.wo)
                eMask |= 1u << 6;
            ePair = (a.wo & 1) == 0 && sizeof(T) == 2;
        }
        // Passes: 16-channel slab m of the K wave groups pg * KWP .. + KWP.
        constexpr int KWP = C::KWP;
#pragma unroll
        for(int m = 0; m < MT; ++m)
#pragma unroll
            for(int pg = 0; pg < KW / KWP; ++pg)
            {
                if(kw / KWP == pg)
                {
#pragma unroll
                    for(int n = 0; n < NT; ++n)
#pragma unroll
                        for(int e = 0; e < 8; ++e)
                        {
                            const float z0 = acc[0][m][n][e] + acc[1][m][n][e] + acc[2][m][n][e];
                            const float z1 = acc[1][m][n][e] - acc[2][m][n][e] - acc[3][m][n][e];
                            const int kk = (kw % KWP) * 16 + W::dRow(lane, e), tt = n * 16 + col;
                            Zs[((row * 2 + 0) * 16 * KWP + kk) * BT + tt] = z0;
                            Zs[((row * 2 + 1) * 16 * KWP + kk) * BT + tt] = z1;
                        }
                }
                k5cLdsBarrier();
                // THREADS is a multiple of BT: the thread's tile is fixed, only the channel steps.
                static_assert(THREADS % BT == 0, "epilogue tile per thread");
#pragma unroll
                for(int pi = tid; pi < 16 * KWP * BT; pi += THREADS)
                {
                    const int kk = pi / BT;
                    const int k = k0 + (pg * KWP + kk / 16) * 16 * MT + m * 16 + kk % 16;
                    if(!eTile || k >= cgo)
                        continue;
                    float z[4][2];
#pragma unroll
                    for(int i = 0; i < 4; ++i)
#pragma unroll
                        for(int s = 0; s < 2; ++s)
                            z[i][s] = Zs[((i * 2 + s) * 16 * KWP + kk) * BT + eTt];
                    float y[2][2];
#pragma unroll
                    for(int s = 0; s < 2; ++s)
                    {
                        y[0][s] = z[0][s] + z[1][s] + z[2][s];
                        y[1][s] = z[1][s] - z[2][s] - z[3][s];
                    }
                    T* o = out + eBase + static_cast<long long>(k) * eChan;
#pragma unroll
                    for(int r = 0; r < 2; ++r)
                    {
                        if(!((eMask >> (4 + r)) & 1u))
                            continue; // row past the output
                        const T v0 = K5Elem<T>::store(((eMask >> (2 * r)) & 1u) ? y[r][0] : 0.0f);
                        const T v1
                            = K5Elem<T>::store(((eMask >> (2 * r + 1)) & 1u) ? y[r][1] : 0.0f);
                        if(ePair)
                        {
                            // Both columns, a 4-byte aligned pair (even output width).
                            const unsigned lo = __builtin_bit_cast(unsigned short, v0);
                            const unsigned hi = __builtin_bit_cast(unsigned short, v1);
                            *reinterpret_cast<unsigned*>(o + r * a.wo) = lo | (hi << 16);
                        }
                        else
                        {
                            o[r * a.wo] = v0;
                            if((eMask >> 6) & 1u)
                                o[r * a.wo + 1] = v1;
                        }
                    }
                }
                k5cLdsBarrier();
            }
    }
}

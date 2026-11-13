// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused Winograd F(2,3) on matrix cores, fp16 and bf16, fp32 accumulate, NCHW, forward form
// (backward data runs it on the rotated filter): RDNA WMMA (gfx11 / gfx12, wave32) and CDNA MFMA
// (gfx908+, wave64), one body over the atom traits below. F(3,2) (M = 3) runs stride 2 by phases
// on the same 16 bins: B^T is F(2,3)'s, only the filter and output transforms and the phase
// gather (forward) / scatter (backward data) differ. The cooperative variant: the whole
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

// The cooperative kernel's filter pre-pass: U fragment-major, so a wave's A fragment is one
// contiguous run (512 bytes) instead of 16 rows a channel stride apart. Unit (g, bin, kb = k / 16,
// ch = c / 16) holds 256 elements in the atom's A lane order: (row r = k % 16, channel j = c % 16)
// at ((j / KPL) * 16 + r) * KPL + j % KPL. Units run [g][bin][ceil(cgo / 16)][cgiPad / 16]; the
// rows of a partial last k block are never written (their outputs are discarded).
// M 2: F(2,3) on the 3x3 filter; M 3: F(3,2) on the stride-2 phase sub-filters (virtual channels).
template <class T, int LAYOUT, int DIR, int KPL, int M = 2>
__device__ void k5CoopFilterTransform(const K5Args& a)
{
    constexpr int P = 16;
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= static_cast<long long>(a.groups) * cgo * cgiPad)
        return;
    const int i = static_cast<int>(idx % cgiPad);
    const int o = static_cast<int>((idx / cgiPad) % cgo);
    const int g = static_cast<int>(idx / (static_cast<long long>(cgiPad) * cgo));
    float u[P];
    if constexpr(M == 3)
    {
        float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if(i < cgi)
            k5GatherFilter<T, LAYOUT, DIR, 2>(a, g, o, i, f);
        k5FilterTile<3>(f, u);
    }
    else
    {
        const int cgf = DIR == K5_FORWARD ? cgi : cgo;
        const bool rot = (DIR == K5_BACKWARD_DATA) != (a.flip != 0);
        const T* w = static_cast<const T*>(a.w);
        float f[9];
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
    }
    T* U = static_cast<T*>(a.u);
    const int kbs = (cgo + 15) / 16, chs = cgiPad / 16;
    const int j = i % 16;
    const long long inUnit = ((j / KPL) * 16 + o % 16) * KPL + j % KPL;
#pragma unroll
    for(int p = 0; p < P; ++p)
    {
        const long long unit = ((static_cast<long long>(g) * P + p) * kbs + o / 16) * chs + i / 16;
        U[unit * 256 + inUnit] = K5Elem<T>::store(u[p]);
    }
}

// Workgroup barrier that orders LDS only: __syncthreads() also waits for every outstanding global
// load (vmcnt(0)), which would expose the prefetched U and input loads at every chunk.
__device__ inline void k5cLdsBarrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The 16 x 16 x 16 matrix atom the cooperative kernel runs on, with how its A (transformed
// filter, from global) and B (transformed input, from LDS) fragments load.
//   RDNA WMMA (wave32): a lane holds 16 (gfx11) or 8 (gfx12) k of its row / column; 8 accumulators.
//   CDNA MFMA (wave64, v_mfma_f32_16x16x16 f16 / bf16_1k): a lane holds k 4 (lane / 16) .. + 3 of
//   row / column lane % 16; 4 accumulators, rows 4 (lane / 16) + i.
template <class T, bool GFX12>
struct K5CoopWmma
{
    typedef K5Wmma<T, GFX12> W;
    typedef typename W::frag frag;
    typedef k5f8 acc;
    static constexpr int WAVE = 32, ACCN = 8, KPL = W::KPL;
    __device__ static int kBase(int lane)
    {
        return W::kBase(lane);
    }
    __device__ static int dRow(int lane, int i)
    {
        return W::dRow(lane, i);
    }
    __device__ static acc mma(frag a, frag b, acc c)
    {
        return W::mma(a, b, c);
    }
    __device__ static frag loadA(__amdgpu_buffer_rsrc_t r, unsigned laneOff, unsigned so)
    {
        if constexpr(GFX12)
            return __builtin_bit_cast(frag,
                                      __builtin_amdgcn_raw_buffer_load_b128(r, laneOff, so, 0));
        else
        {
            const k5u4 lo = __builtin_bit_cast(
                k5u4, __builtin_amdgcn_raw_buffer_load_b128(r, laneOff, so, 0));
            const k5u4 hi = __builtin_bit_cast(
                k5u4, __builtin_amdgcn_raw_buffer_load_b128(r, laneOff + 16, so, 0));
            return __builtin_bit_cast(frag, k5u8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
        }
    }
    __device__ static frag loadB(const unsigned* p)
    {
        if constexpr(GFX12)
            return __builtin_bit_cast(frag, *reinterpret_cast<const k5u4*>(p));
        else
        {
            const k5u4 lo = *reinterpret_cast<const k5u4*>(p);
            const k5u4 hi = *reinterpret_cast<const k5u4*>(p + 4);
            return __builtin_bit_cast(frag, k5u8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
        }
    }
};

template <class T>
struct K5CoopMfma
{
    typedef typename K5WmmaElem<T>::e e;
    typedef e frag __attribute__((ext_vector_type(4)));
    typedef float acc __attribute__((ext_vector_type(4)));
    static constexpr int WAVE = 64, ACCN = 4, KPL = 4;
    __device__ static int kBase(int lane)
    {
        return (lane >> 4) * 4;
    }
    __device__ static int dRow(int lane, int i)
    {
        return (lane >> 4) * 4 + i;
    }
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) \
    || defined(__gfx942__) || defined(__gfx950__)
        if constexpr(__is_same(T, k5_half))
            return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#if !defined(__gfx908__)
        else
            return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
#endif
#endif
        __builtin_trap();
        return c;
    }
    __device__ static frag loadA(__amdgpu_buffer_rsrc_t r, unsigned laneOff, unsigned so)
    {
        typedef unsigned u2 __attribute__((ext_vector_type(2)));
        return __builtin_bit_cast(
            frag, __builtin_bit_cast(u2, __builtin_amdgcn_raw_buffer_load_b64(r, laneOff, so, 0)));
    }
    __device__ static frag loadB(const unsigned* p)
    {
        typedef unsigned u2 __attribute__((ext_vector_type(2)));
        return __builtin_bit_cast(frag, *reinterpret_cast<const u2*>(p));
    }
};

template <class T, class A, int MT, int NT, int KW, int M = 2>
struct K5CoopCfg
{
    static constexpr int BK = 16 * MT * KW, BT = 16 * NT, WAVES = 4 * KW, THREADS = A::WAVE * WAVES;
    static constexpr int VB = 16 * BT * 16; // halves per V buffer: [bin][tile][16 channels]
    // (tile, channel pair) jobs per chunk: BT x 8; a lane takes one, 16 lanes per channel pair.
    static constexpr int JW = BT * 8 / A::WAVE; // waves of jobs per chunk
    static constexpr int JT = JW > WAVES ? JW / WAVES : 1; // jobs per thread
    // K wave groups per epilogue pass: 2 where Z fits the V buffers' LDS (F(3,2)'s 3 values per
    // bin row do not: 1).
    static constexpr int KWP = KW < 2 ? KW : M == 2 ? 2 : 1;
    // Z row stride (floats): on CDNA the accumulator lanes 16 apart write rows 4 apart, which a
    // BT stride puts on one bank; BT + 4 spreads them over the 64 banks -- where that does not
    // grow the block's LDS past the V buffers (64 KiB parts). Z holds M values per bin row.
    static constexpr int ZS
        = A::WAVE == 64 && 4 * M * 16 * KWP * (BT + 4) * 4 <= 2 * VB * 2 ? BT + 4 : BT;
    static constexpr int ZB = 4 * M * 16 * KWP * ZS; // floats per epilogue pass
    static constexpr int LDSB = 2 * VB * 2 > ZB * 4 ? 2 * VB * 2 : ZB * 4; // bytes
    // RDNA: waves per SIMD the kernel is built for (VGPR budget 1536 / WPE): the loop is
    // latency-bound, so the single-tile configs hold 8 (192 VGPRs); 2 x 2 tiles need the
    // registers. CDNA (512 registers per lane, accumulators may sit in AGPRs): 2.
    static constexpr int WPE = A::WAVE == 64 ? 2 : MT * NT <= 2 ? 8 : 4;
    static_assert(JW % WAVES == 0 || WAVES % JW == 0, "job split");
};

// M 2: F(2,3), stride 1 (mode 0). M 3: F(3,2) on the stride-2 phase sub-filters (K5Args::mode):
// forward (mode 1) a lane's channel pair is one real channel's two column phases of one row
// parity, so a row is one 8-element run of x (two 12-byte loads, de-interleaved); backward data
// (mode 2) reads dy like stride 1 and scatters each virtual out channel's 3x3 to its dx phase.
template <class T,
          class W,
          int MT,
          int NT,
          int KW,
          bool F16MATH,
          int M = 2,
          int DIR = K5_FORWARD,
          int PF = 0>
__device__ __attribute__((always_inline)) void k5CoopConvA(const K5Args& a)
{
    typedef K5CoopCfg<T, W, MT, NT, KW, M> C;
    typedef typename W::frag frag;
    constexpr int WAVE = W::WAVE, ACCN = W::ACCN;
    constexpr int BK = C::BK, BT = C::BT, THREADS = C::THREADS, VB = C::VB, JT = C::JT, JW = C::JW;
    constexpr bool PACKED = F16MATH && __is_same(T, k5_half);
    constexpr int MODE = M == 3 ? (DIR == K5_FORWARD ? 1 : 2) : 0;
    static_assert(M == 2 || M == 3, "F(2,3) or F(3,2)");
    __shared__ __attribute__((aligned(16))) unsigned char ldsRaw[C::LDSB];
    unsigned* Vd = reinterpret_cast<unsigned*>(ldsRaw); // two buffers of VB halves
    float* Zs = reinterpret_cast<float*>(ldsRaw);

    const int cgi = a.ci / a.groups, cgo = a.co / a.groups, cgiPad = (cgi + 15) / 16 * 16;
    const int nKB = (cgo + BK - 1) / BK, nTB = (a.tiles + BT - 1) / BT;
    const long long items = static_cast<long long>(a.groups) * nKB * nTB;
    const int nCh = cgiPad / 16;
    // Mode 1 reads x itself: 4 virtual channels (phases) per plane of x.
    const int chanStride = MODE == 1 ? a.xh * a.xw : a.hi * a.wi;
    const long long inElems
        = static_cast<long long>(a.n) * (MODE == 1 ? a.ci / 4 : a.ci) * chanStride;
    const __amdgpu_buffer_rsrc_t rIn
        = k5Rsrc(a.in, (inElems * static_cast<long long>(sizeof(T)) + 3) / 4 * 4);
    const int kbs = (cgo + 15) / 16;
    const __amdgpu_buffer_rsrc_t rU = k5Rsrc(a.u,
                                             static_cast<long long>(a.groups) * 16 * kbs * 16
                                                 * cgiPad * static_cast<long long>(sizeof(T)));
    const int tid = threadIdx.x, lane = tid % WAVE;
    // Wave-uniform role.
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int row = wave % 4, kw = wave / 4;
    const int col = lane & 15;
    T* out = static_cast<T*>(a.out);
    const unsigned chunkBytes
        = static_cast<unsigned>((MODE == 1 ? 4 : 16) * chanStride) * sizeof(T);
    // V swizzle: tiles 8 apart share 64-bank rows, so tiles with bit 3 set take the other half
    // of their 8 channel-pair dwords: the transform's dword stores and the B reads then spread
    // over the banks (CDNA: 16 tiles x 4 pairs a wave, conflict cycles 3.2e8 -> 1.2e8 on gfx950
    // unet; gfx12: unet -4%). gfx11 reads all 8 dwords of a tile in one fragment (KPL 16), which
    // the half swap would split: not swizzled.
    auto vSwz = [](int tile) __attribute__((always_inline))
    {
        return W::KPL <= 8 ? ((tile >> 3) & 1) << 2 : 0;
    };

    // A fragment lane offset (bytes) within a U unit (k5CoopFilterTransform): the lane's KPL
    // elements (gfx11's wave32 lanes 16-31 repeat rows 0-15).
    const unsigned uLane
        = static_cast<unsigned>((W::KPL == 16 ? lane % 16 : lane) * W::KPL) * sizeof(T);
    constexpr unsigned UNIT = 256 * sizeof(T);

    // The transform jobs' tables (set per item, setupJobs below), the input chunk in flight, and
    // this wave's U offset. PF (prefetch) 1 sets them up one item ahead: the last chunk step of an
    // item loads the next item's chunk 0 and its U, so their latency hides behind the epilogue.
    int voff[JT][4][2]; // aligned byte offsets of the 12-byte row loads (chunk 0; may be < 0)
    unsigned shf[JT]; // per (row, channel) half shift bits: bit 2 * r + v
    unsigned selc[JT][4]; // channel-pairing perm selectors per column (0x0c: padding)
    int tl[JT], cp[JT];
    bool jobOn[JT];
    bool negWave = false;
    unsigned uWave = 0;
    typedef unsigned k5u3 __attribute__((ext_vector_type(3)));
    k5u3 raw[JT][4][2]; // the next chunk's input
    // The U fragments: with PF 1 the last chunk step loads the next item's chunk 0 here, so they
    // must outlive the item (a per-item array would be a fresh, uninitialised one each item).
    frag ufr[4][MT];

    for(long long item = blockIdx.x; item < items; item += gridDim.x)
    {
        // Item order (K5Args::order): 0 K blocks fastest (the tile block's input stays in cache
        // across them), 1 tile blocks fastest (the K block's U stays in cache across them: large
        // C x K, where U re-reads would dominate).
        int kb, tb, g;
        if(a.order == 1)
        {
            tb = static_cast<int>(item % nTB);
            const long long r0 = item / nTB;
            kb = static_cast<int>(r0 % nKB);
            g = static_cast<int>(r0 / nKB);
        }
        else
        {
            kb = static_cast<int>(item % nKB);
            const long long r0 = item / nKB;
            tb = static_cast<int>(r0 % nTB);
            g = static_cast<int>(r0 / nTB);
        }
        const int k0 = kb * BK, t0 = tb * BT;

        // ---- Transform jobs: lane -> (tile, channel pair). Job wave jw covers tiles
        // 16 * (jw % NT) + col and channel pair 2 * (jw / NT) + lane / 16.
        auto setupJobs = [&](int t0, int g) __attribute__((always_inline))
        {
            bool anyNeg = false;
#pragma unroll
            for(int q = 0; q < JT; ++q)
            {
                const int jw = JT > 1 ? wave + q * C::WAVES : wave;
                jobOn[q] = jw < JW;
                tl[q] = 16 * (jw % NT) + col;
                cp[q] = (WAVE / 16) * (jw / NT) + (lane >> 4);
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
                    if constexpr(MODE == 1)
                    {
                        // Virtual channels 2 cp, 2 cp + 1 = real channel cp / 2, row parity cp % 2,
                        // column phases 0 and 1: row r of the tile is x row 2 (th M + r) + parity,
                        // columns 2 (tw M) - xpw + 0 .. 7 (phase = column parity).
                        const int c = cp[q] >> 1, rp = cp[q] & 1;
                        const long long cbase
                            = (static_cast<long long>(img) * (a.ci >> 2) + g * (cgi >> 2) + c)
                              * chanStride;
                        const int iw0 = 2 * tw * M - a.xpw;
#pragma unroll
                        for(int r = 0; r < 4; ++r)
                        {
                            const int ih = 2 * (th * M + r) + rp - a.xph;
                            if(ih < 0 || ih >= a.xh || 2 * cp[q] >= cgi)
                                continue;
                            const long long e = cbase + static_cast<long long>(ih) * a.xw + iw0;
                            anyNeg = anyNeg || e < 0;
                            voff[q][r][0] = static_cast<int>((e & ~1LL) * 2);
                            voff[q][r][1] = voff[q][r][0] + 12;
                            shf[q] |= static_cast<unsigned>(e & 1) << (2 * r);
                        }
#pragma unroll
                        for(int j = 0; j < 4; ++j)
                        {
                            // Per channel (half): the column's half of each channel, or 0 (0x0c).
                            const int iw = iw0 + 2 * j;
                            const unsigned odd = (j & 1) ? 0x0202u : 0u;
                            const unsigned lo = iw >= 0 && iw < a.xw ? 0x0100u + odd : 0x0c0cu;
                            const unsigned hi
                                = iw + 1 >= 0 && iw + 1 < a.xw ? 0x0504u + odd : 0x0c0cu;
                            selc[q][j] = lo | (hi << 16);
                        }
                    }
                    else
                    {
                        const int ih0 = th * M - a.ph, iw0 = tw * M - a.pw;
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
            }
            // The slow path for the rare lanes whose first row element is before the tensor start:
            // per-element loads (chunk 0 only; later chunks are 16 planes further on).
            return __any(anyNeg) != 0; // wave-uniform
        };
        if(PF == 0 || item == blockIdx.x)
            negWave = setupJobs(t0, g);

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
                if constexpr(MODE == 1)
                {
                    // The row's 8-element run, element by element, in the aligned load format.
                    const int c = cp[q] >> 1, rp = cp[q] & 1;
                    const int iw0 = 2 * tw * M - a.xpw;
#pragma unroll
                    for(int r = 0; r < 4; ++r)
                    {
                        const int ih = 2 * (th * M + r) + rp - a.xph;
                        const bool rok = ih >= 0 && ih < a.xh && 2 * cp[q] < cgi;
                        unsigned hv[8];
#pragma unroll
                        for(int j = 0; j < 8; ++j)
                        {
                            const int iw = iw0 + j;
                            const long long e
                                = ((static_cast<long long>(img) * (a.ci >> 2) + g * (cgi >> 2) + c)
                                       * a.xh
                                   + ih)
                                      * a.xw
                                  + iw;
                            hv[j] = __builtin_amdgcn_raw_buffer_load_b16(
                                rIn,
                                rok && iw >= 0 && iw < a.xw
                                    ? static_cast<unsigned>(e * static_cast<long long>(sizeof(T)))
                                    : static_cast<unsigned>(K5C_OOB),
                                0,
                                0);
                        }
                        raw[q][r][0] = k5u3{
                            hv[0] | (hv[1] << 16), hv[2] | (hv[3] << 16), hv[4] | (hv[5] << 16)};
                        raw[q][r][1] = k5u3{hv[6] | (hv[7] << 16), 0u, 0u};
                    }
                    continue;
                }
                const int ih0 = th * M - a.ph, iw0 = tw * M - a.pw;
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
                {
                    if constexpr(MODE == 1)
                    {
                        // Realign the row's run, then de-interleave the column phases.
                        const unsigned s = aligned ? 0u : ((shf[q] >> (2 * r)) & 1u) * 16u;
                        const k5u3 d = raw[q][r][0], e = raw[q][r][1];
                        const unsigned a0 = __builtin_amdgcn_alignbit(d.y, d.x, s);
                        const unsigned a1 = __builtin_amdgcn_alignbit(d.z, d.y, s);
                        const unsigned a2 = __builtin_amdgcn_alignbit(e.x, d.z, s);
                        const unsigned a3 = __builtin_amdgcn_alignbit(e.y, e.x, s);
                        x[r][0][0] = __builtin_amdgcn_perm(a1, a0, 0x05040100u);
                        x[r][0][1] = __builtin_amdgcn_perm(a3, a2, 0x05040100u);
                        x[r][1][0] = __builtin_amdgcn_perm(a1, a0, 0x07060302u);
                        x[r][1][1] = __builtin_amdgcn_perm(a3, a2, 0x07060302u);
                    }
                    else
                    {
#pragma unroll
                        for(int v = 0; v < 2; ++v)
                        {
                            const unsigned s = aligned ? 0u : ((shf[q] >> (2 * r + v)) & 1u) * 16u;
                            const k5u3 d = raw[q][r][v];
                            x[r][v][0] = __builtin_amdgcn_alignbit(d.y, d.x, s);
                            x[r][v][1] = __builtin_amdgcn_alignbit(d.z, d.y, s);
                        }
                    }
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
                                // Per channel: mode 1's column phases end at different columns.
                                const bool cok = ((selc[q][j] >> (16 * v)) & 0xffffu) != 0x0c0cu;
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
                // V[bin][tile][channel]: dword (bin * BT + tile) * 8 + channel pair (swizzled).
#pragma unroll
                for(int b = 0; b < 16; ++b)
                    Vw[(b * BT + tl[q]) * 8 + (cp[q] ^ vSwz(tl[q]))] = vout[b];
            }
        };

        typename W::acc acc[4][MT][NT];
#pragma unroll
        for(int b = 0; b < 4; ++b)
#pragma unroll
            for(int m = 0; m < MT; ++m)
#pragma unroll
                for(int n = 0; n < NT; ++n)
                    acc[b][m][n] = typename W::acc{};

        // A fragments for this wave: bins 4 * row + b, out channels kRow0 + m * 16 + col.
        const int kRow0 = k0 + kw * 16 * MT;
        // Unit (g, bin 4 row + b, k block kRow0 / 16 + m, chunk ch).
        const int nChU = cgiPad / 16;
        const auto uOf = [&](int g, int kRow0) __attribute__((always_inline))
        {
            return static_cast<unsigned>(((g * 16 + 4 * row) * kbs + kRow0 / 16) * nChU) * UNIT;
        };
        if(PF == 0 || item == blockIdx.x)
            uWave = uOf(g, kRow0);
        const unsigned binBytes = static_cast<unsigned>(kbs * nChU) * UNIT;
        auto loadU = [&](int ch, int b) __attribute__((always_inline))
        {
            // uWave is wave-uniform (row is); readfirstlane says so. With PF 1 it is set under a
            // branch the compiler cannot prove uniform, and a per-lane scalar offset turns every
            // U load into a waterfall loop.
            const unsigned uw = __builtin_amdgcn_readfirstlane(uWave);
#pragma unroll
            for(int m = 0; m < MT; ++m)
            {
                const unsigned so = uw + b * binBytes + static_cast<unsigned>(m * nChU + ch) * UNIT;
                ufr[b][m] = W::loadA(rU, uLane, so);
            }
        };
        // B fragment: tile n * 16 + col of bin, channels kBase.
        auto loadB = [&](const unsigned* Vs, int bin, int n) __attribute__((always_inline))
        {
            const unsigned* p
                = Vs + (bin * BT + n * 16 + col) * 8 + ((W::kBase(lane) / 2) ^ vSwz(col));
            return W::loadB(p);
        };

        // Prologue: chunk 0 in, transformed into buffer 0; U of chunk 0 (PF 1: but the first
        // item's were loaded by the previous item).
        if(PF == 0 || item == blockIdx.x)
            loadIn(0);
        if(negWave)
            loadInSlow();
        if(PF == 0 || item == blockIdx.x)
        {
#pragma unroll
            for(int b = 0; b < 4; ++b)
                loadU(0, b);
        }
        transform(0, negWave);
        __syncthreads();
        // Chunk ch (parity P): V buffer P holds it; load chunk ch + 1's input, run the WMMAs, then
        // transform chunk ch + 1 into V buffer 1 - P.
        auto step = [&](int ch, auto parTag) __attribute__((always_inline))
        {
            constexpr int P = decltype(parTag)::value;
            const int last = nCh - 1;
            int nx = ch + 1 < last ? ch + 1 : last; // the last chunk reloads itself
            if constexpr(PF == 1)
            {
                // ... or loads the next item's chunk 0 (and the U loads below its U).
                const long long next = item + gridDim.x;
                if(ch == last && next < items)
                {
                    int kb2, tb2, g2;
                    if(a.order == 1)
                    {
                        tb2 = static_cast<int>(next % nTB);
                        kb2 = static_cast<int>(next / nTB % nKB);
                        g2 = static_cast<int>(next / nTB / nKB);
                    }
                    else
                    {
                        kb2 = static_cast<int>(next % nKB);
                        tb2 = static_cast<int>(next / nKB % nTB);
                        g2 = static_cast<int>(next / nKB / nTB);
                    }
                    negWave = setupJobs(tb2 * BT, g2);
                    uWave = uOf(g2, kb2 * BK + kw * 16 * MT);
                    nx = 0;
                }
            }
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

        if constexpr(M == 3)
        {
            // F(3,2) epilogue, per 16-channel slab: each bin row folds to 3 values (Z = M A) in
            // LDS, then A^T over the 4 rows gives the tile's 3x3. Mode 1 writes y in place; mode
            // 2 writes virtual channel 4c + p to dx row 2 (3 th + r) + p / 2 - xph, column
            // 2 (3 tw + s) + p % 2 - xpw.
            const int eTt = tid % BT;
            const int eT2 = t0 + eTt;
            const bool eTile = eT2 < a.tiles;
            int eTh = 0, eTw = 0, eImg = 0;
            // Mode 2: the tile's 6 dx rows (bit 2r + pr) and columns (bit j) with a dy element
            // under the filter (k5LiveBwd2, separable), and the columns inside dx.
            unsigned eRowLive = 0, eColLive = 0, eColIn = 0;
            if(eTile)
            {
                eTw = eT2 % a.tw;
                const int r1 = eT2 / a.tw;
                eTh = r1 % a.th;
                eImg = r1 / a.th;
                if constexpr(MODE == 2)
                {
                    auto cover = [](int x, int pad, int n) __attribute__((always_inline))
                    {
                        const int x0 = (x + pad - 1) >> 1, x1 = (x + pad) >> 1;
                        return (x0 > 0 ? x0 : 0) <= (x1 < n - 1 ? x1 : n - 1);
                    };
#pragma unroll
                    for(int j = 0; j < 6; ++j)
                    {
                        const int h = 6 * eTh + j - a.xph, w = 6 * eTw + j - a.xpw;
                        eRowLive |= static_cast<unsigned>(cover(h, a.xph, a.hi)) << j;
                        eColLive |= static_cast<unsigned>(cover(w, a.xpw, a.wi)) << j;
                        eColIn |= static_cast<unsigned>(w >= 0 && w < a.xw) << j;
                    }
                }
            }
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
                            for(int e = 0; e < ACCN; ++e)
                            {
                                const float m0 = acc[0][m][n][e], m1 = acc[1][m][n][e],
                                            m2 = acc[2][m][n][e], m3 = acc[3][m][n][e];
                                const int kk = (kw % KWP) * 16 + W::dRow(lane, e),
                                          tt = n * 16 + col;
                                Zs[((row * 3 + 0) * 16 * KWP + kk) * C::ZS + tt] = m0 + m1 + m2;
                                Zs[((row * 3 + 1) * 16 * KWP + kk) * C::ZS + tt] = m1 - m2;
                                Zs[((row * 3 + 2) * 16 * KWP + kk) * C::ZS + tt] = m1 + m2 + m3;
                            }
                    }
                    k5cLdsBarrier();
                    static_assert(THREADS % BT == 0, "epilogue tile per thread");
                    // The tile's 3x3 of virtual channel kk (Z rows i, columns s).
                    auto tileOut = [&](int kk, float(&y)[3][3]) __attribute__((always_inline))
                    {
                        float z[4][3];
#pragma unroll
                        for(int i = 0; i < 4; ++i)
#pragma unroll
                            for(int s = 0; s < 3; ++s)
                                z[i][s] = Zs[((i * 3 + s) * 16 * KWP + kk) * C::ZS + eTt];
#pragma unroll
                        for(int s = 0; s < 3; ++s)
                        {
                            y[0][s] = z[0][s] + z[1][s] + z[2][s];
                            y[1][s] = z[1][s] - z[2][s];
                            y[2][s] = z[1][s] + z[2][s] + z[3][s];
                        }
                    };
                    if constexpr(MODE == 1)
                    {
#pragma unroll
                        for(int pi = tid; pi < 16 * KWP * BT; pi += THREADS)
                        {
                            const int kk = pi / BT;
                            const int k = k0 + (pg * KWP + kk / 16) * 16 * MT + m * 16 + kk % 16;
                            if(!eTile || k >= cgo)
                                continue;
                            float y[3][3];
                            tileOut(kk, y);
                            T* o = out
                                   + ((static_cast<long long>(eImg) * a.co + g * cgo + k) * a.ho
                                      + eTh * 3)
                                         * a.wo
                                   + eTw * 3;
#pragma unroll
                            for(int r = 0; r < 3; ++r)
#pragma unroll
                                for(int s = 0; s < 3; ++s)
                                {
                                    const int oh = eTh * 3 + r, ow = eTw * 3 + s;
                                    if(oh < a.ho && ow < a.wo)
                                        o[r * a.wo + s]
                                            = K5Elem<T>::store(k5Live(a, oh, ow) ? y[r][s] : 0.0f);
                                }
                        }
                    }
                    else
                    {
                        // A thread takes one real channel's two column phases of one row parity
                        // pr: dx rows 6 th - xph + 2 r + pr, the tile's 6 contiguous columns
                        // 6 tw - xpw + 2 s + pc, stored in aligned pairs where both are in range.
#pragma unroll
                        for(int pi = tid; pi < 8 * KWP * BT; pi += THREADS)
                        {
                            const int pr = (pi / BT) & 1, kq = pi / (2 * BT);
                            const int kk0 = 4 * kq;
                            const int k = k0 + (pg * KWP + kk0 / 16) * 16 * MT + m * 16 + kk0 % 16;
                            if(!eTile || k >= cgo)
                                continue;
                            float y[2][3][3];
                            tileOut(kk0 + 2 * pr, y[0]);
                            tileOut(kk0 + 2 * pr + 1, y[1]);
                            const long long plane = (static_cast<long long>(eImg) * (a.co >> 2)
                                                     + g * (cgo >> 2) + (k >> 2))
                                                    * a.xh * a.xw;
                            const int w0 = 6 * eTw - a.xpw;
#pragma unroll
                            for(int r = 0; r < 3; ++r)
                            {
                                const int h = 6 * eTh + 2 * r + pr - a.xph;
                                if(h < 0 || h >= a.xh)
                                    continue;
                                const bool hLive = (eRowLive >> (2 * r + pr)) & 1u;
                                const long long rowOff
                                    = plane + static_cast<long long>(h) * a.xw + w0;
                                T* o = out + rowOff;
                                unsigned short v[6];
#pragma unroll
                                for(int j = 0; j < 6; ++j)
                                {
                                    const bool live = hLive && ((eColLive >> j) & 1u);
                                    v[j] = __builtin_bit_cast(
                                        unsigned short,
                                        K5Elem<T>::store(live ? y[j & 1][r][j >> 1] : 0.0f));
                                }
                                auto one = [&](int j) __attribute__((always_inline))
                                {
                                    if((eColIn >> j) & 1u)
                                        o[j] = __builtin_bit_cast(T, v[j]);
                                };
                                auto two = [&](int j) __attribute__((always_inline))
                                {
                                    if(((eColIn >> j) & 3u) == 3u)
                                        *reinterpret_cast<unsigned*>(o + j)
                                            = v[j] | (static_cast<unsigned>(v[j + 1]) << 16);
                                    else
                                    {
                                        one(j);
                                        one(j + 1);
                                    }
                                };
                                if((rowOff & 1) == 0)
                                {
                                    two(0);
                                    two(2);
                                    two(4);
                                }
                                else
                                {
                                    one(0);
                                    two(1);
                                    two(3);
                                    one(5);
                                }
                            }
                        }
                    }
                    k5cLdsBarrier();
                }
            continue;
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
                        for(int e = 0; e < ACCN; ++e)
                        {
                            const float z0 = acc[0][m][n][e] + acc[1][m][n][e] + acc[2][m][n][e];
                            const float z1 = acc[1][m][n][e] - acc[2][m][n][e] - acc[3][m][n][e];
                            const int kk = (kw % KWP) * 16 + W::dRow(lane, e), tt = n * 16 + col;
                            Zs[((row * 2 + 0) * 16 * KWP + kk) * C::ZS + tt] = z0;
                            Zs[((row * 2 + 1) * 16 * KWP + kk) * C::ZS + tt] = z1;
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
                            z[i][s] = Zs[((i * 2 + s) * 16 * KWP + kk) * C::ZS + eTt];
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

template <class T, bool GFX12, int MT, int NT, int KW, bool F16MATH>
__device__ __attribute__((always_inline)) void k5CoopConv(const K5Args& a)
{
    k5CoopConvA<T, K5CoopWmma<T, GFX12>, MT, NT, KW, F16MATH>(a);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K1 flex tile (CONV_IGEMM_FLEX 1): the implicit GEMM over flattened output positions as one
// 256-position x BN-channel block (BN 256, 192 or 128) of eight waves on gfx950's 16x16x32 matrix cores, for 16-bit
// channels-last forward and unit-stride backward-data graphs. Included by ConvIgemm.cpp after
// ConvIgemmCore.h (atoms, K1Args, k1Setup, k1Rsrc, k1XcdOrder). HIPRTC-compatible.
//
// GEMM: rows m are the output positions of all images in order (the block's 256 rows may span
// planes and images, so no spatial tile is padded), columns n the output channels of one group,
// the reduction runs over 64-channel slices of the reduction channels (outer) and filter taps
// (inner, t r s). A step is one slice and one tap: consecutive steps read the same channels of
// the same positions shifted by one tap, so most of a step's input rows are L1 hits.
//
// Operands reach LDS by 16-byte LDS DMA, both of them, with no register staging. A row (one
// position or one output channel) is 64 channels, one 128-byte fibre: whole-fibre rows move
// 1.4x the bytes per cycle of half-fibre ones on gfx950.
//   - X (positions x 64 channels): each lane's rows are fixed for the block; per row it keeps the
//     row's offset at tap 0 and one in-image bit per tap of each axis, so a step's address is the
//     row offset plus a uniform tap delta and its bound check is one mask test (out-of-image taps
//     and channels past the reduction read zero through the buffer range check);
//   - W (output channels x 64 channels): forward reads the K T R S C/g filter in place, backward
//     data the [g][tap][c][k] filter the pack stages; channels past the group read zero.
// Granule g of row r sits at slot g ^ (r % 8), so the 16-lane fragment reads of 16 consecutive
// rows hit 8 distinct 16-byte bank slots per 128 bytes.
//
// Stages: two X tiles (32 KiB each) and three W tiles (BN rows), separate LDS objects so the
// compiler tracks each DMA against its own stage. A step's first half issues the X tile of the
// next step (mostly L1 hits: the previous tap read the same rows), its second half the W tile of
// the step after next, so each memory phase carries part of the DMA and W, read from L2, has a
// whole step to land.
//
// Waves: two groups of four, the BN / 2-channel halves of the block (ping-pong). A group's four
// waves split the 256 positions, so a wave owns 64 positions x BN / 2 channels: 4 x BN / 32 MMAs
// per 32 channels from 4 X and BN / 32 W fragments (the squarest wave tile reads the fewest LDS
// bytes for its MMAs, and its accumulators leave BN 256 room without spills). A step is two halves of 32 channels, each "memory phase | barrier | MMAs
// at raised priority | barrier"; the second group runs one barrier behind the first, so on every
// SIMD one wave multiplies while the other reads LDS and issues DMA.
//
// Epilogue: each wave stages its tile in LDS and stores whole 16-byte runs of a position's
// channels (direct 8-byte stores where the channels are not whole runs).
//
// Split-K (a.splits > 1): split z runs a contiguous range of the steps and writes fp32 partials
// indexed like the output, which ConvIgemmReduce adds in split order. A launch of the trailing row
// tiles only (the tail split) indexes its partials from row tile a.partialTile.
#pragma once

/// s_barrier without the fences of __syncthreads: the in-flight LDS DMA of the next stage must stay
/// in flight. Memory clobbers keep the compiler from moving LDS accesses across it.
__device__ inline void k1FlexBarrier()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

/// Output positions per image of a gather.
__device__ inline K1Pos spatialOf(const K1Gather& gather)
{
    return static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
}

template <class T, class Atom, int DIR, int BN>
__device__ __attribute__((always_inline)) void k1ConvFlex(const K1Args& a CONV_EPI_FN_PARAM)
{
    constexpr int WAVE = 64, DIM = 16;
    constexpr int BM = 256;
    constexpr int KS = 64; // reduction channels a step
    constexpr int GR = KS * 2 / 16; // granules a row (8)
    constexpr int NG = BM * GR, NGW = BN * GR; // granules of the X and W tiles
    constexpr int RPI = WAVE / GR; // rows a DMA instruction (8)
    constexpr int IPW = BM / RPI / 8, IPWW = BN / RPI / 8; // DMA instructions a wave: X, W
    constexpr int WM = BM / 4, WN = BN / 2; // a wave's output positions and channels
    constexpr int FM = WM / DIM, FN = WN / DIM; // a wave's position x channel fragments
    static_assert(sizeof(T) == 2 && Atom::KA == 32 && Atom::KPL == 8 && Atom::WAVE == WAVE,
                  "16-bit, 16x16x32 atoms");
    static_assert((BN == 256 || BN == 192 || BN == 128) && IPW == 4, "tile");
    typedef typename K1Raw<16>::type Raw16;
    typedef typename Atom::frag Frag;

    __shared__ Raw16 sx0[NG], sx1[NG];
    __shared__ Raw16 sw0[NGW], sw1[NGW], sw2[NGW];

    K1Gather gather;
    K1TapMap map;
    int g, phase;
    unsigned bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    k1XcdOrder(bx, by, bz, K1_XCDS); // gfx950: 8 XCDs
    const int splits = a.splits > 1 ? a.splits : 1;
    const int split = static_cast<int>(bz) % splits;
    k1Setup<DIR>(a, static_cast<int>(bz) / splits, gather, map, g, phase);
    const long long m0 = (static_cast<long long>(bx) + a.tileBase) * BM;
    if(m0 >= gather.rows)
    {
        return;
    }
    const int cg = a.c / a.groups, kg = a.k / a.groups;
    const int ncols = DIR == K1_FORWARD ? kg : cg;
    const int red = gather.cgroup; // reduction channels
    const int n0 = static_cast<int>(by) * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int grp = wave / 4, wx = wave % 4;

    // Steps: slices (outer) x taps (t, r, s, inner), over the depth and height taps that reach
    // the input for some row of the tile (tapRange): where the input is padded, as in backward
    // data of an unpadded forward layer over few planes, the others multiply zeros only.
    const int taps2 = gather.taps[2];
    int q0 = 0, q1 = gather.taps[0], r0 = 0, r1 = gather.taps[1];
    {
        const long long last = (m0 + BM < gather.rows ? m0 + BM : gather.rows) - 1;
        const K1Pos plane = static_cast<K1Pos>(gather.out[1]) * gather.out[2];
        const long long imgF = m0 / spatialOf(gather), imgL = last / spatialOf(gather);
        const K1Pos restF = static_cast<K1Pos>(m0 - imgF * spatialOf(gather));
        const K1Pos restL = static_cast<K1Pos>(last - imgL * spatialOf(gather));
        const int odF = static_cast<int>(restF / plane), odL = static_cast<int>(restL / plane);
        // The tap range of axis a over output positions [oF, oL]: taps whose input lies in the
        // image for some position (the positions' inputs are monotonic in the tap).
        auto tapRange = [&](int ax, int oF, int oL, int& lo, int& hi) __attribute__((always_inline))
        {
            int qlo = gather.taps[ax], qhi = 0;
            for(int q = 0; q < gather.taps[ax]; ++q)
            {
                const int iF = oF * gather.stride[ax] - gather.pad[ax] + q * gather.step[ax];
                const int iL = oL * gather.stride[ax] - gather.pad[ax] + q * gather.step[ax];
                const int iMin = iF < iL ? iF : iL, iMax = iF < iL ? iL : iF;
                if(iMax >= 0 && iMin < gather.in[ax])
                {
                    qlo = q < qlo ? q : qlo;
                    qhi = q + 1;
                }
            }
            // No tap reaches: one tap still runs, so the rows' zeros are written.
            lo = qhi > qlo ? qlo : 0;
            hi = qhi > qlo ? qhi : 1;
        };
        if(imgF == imgL)
        {
            tapRange(0, odF, odL, q0, q1);
            if(odF == odL)
            {
                tapRange(1,
                         static_cast<int>((restF - odF * plane) / gather.out[2]),
                         static_cast<int>((restL - odL * plane) / gather.out[2]),
                         r0,
                         r1);
            }
        }
        q0 = __builtin_amdgcn_readfirstlane(q0);
        q1 = __builtin_amdgcn_readfirstlane(q1);
        r0 = __builtin_amdgcn_readfirstlane(r0);
        r1 = __builtin_amdgcn_readfirstlane(r1);
    }
    const int taps0 = gather.taps[0], taps1 = gather.taps[1];
    const int nTaps = (q1 - q0) * (r1 - r0) * taps2;
    const int total = nTaps * ((red + KS - 1) / KS);
    // Split z: steps [bound(z), bound(z + 1)), bounds on whole loop bodies but the last.
    auto bound = [&](int z) __attribute__((always_inline))
    {
        return z >= splits ? total
                           : static_cast<int>(static_cast<long long>(total) * z / splits) / 6 * 6;
    };
    const int vFirst = bound(split);
    const int vEnd = bound(split + 1);

    // X rows of this lane: r = 32 wave + 8 i + lane / 8, granule slot lane % 8, which holds
    // granule (lane % 8) ^ (lane / 8) of the row.
    const T* src = static_cast<const T*>(DIR == K1_FORWARD ? a.x : a.y);
    const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
    const long long imageElems
        = static_cast<long long>(gather.in[0]) * gather.in[1] * gather.in[2] * gather.chans;
    const long long firstImage = m0 / spatial;
    const long long blockBase = firstImage * imageElems;
    const __amdgpu_buffer_rsrc_t xr = k1Rsrc(
        src + blockBase, (a.n - firstImage) * imageElems * static_cast<long long>(sizeof(T)));
    const int gran = ((lane % GR) ^ (lane / GR)) * 8; // the lane's granule, in elements
    int xOff[IPW]; // the row's element offset at tap 0 (relative to the buffer base)
    unsigned xMask[IPW]; // in-image taps: bits 0..9 depth, 10..19 height, 20..29 width
#pragma unroll
    for(int i = 0; i < IPW; ++i)
    {
        const long long m = m0 + 32 * wave + RPI * i + lane / GR;
        xOff[i] = 0;
        xMask[i] = 0;
        if(m < gather.rows)
        {
            const long long img = m / spatial;
            K1Pos rest = static_cast<K1Pos>(m - img * spatial);
            const int ow = static_cast<int>(rest % gather.out[2]);
            rest /= gather.out[2];
            const int oh = static_cast<int>(rest % gather.out[1]);
            const int od = static_cast<int>(rest / gather.out[1]);
            const int d = od * gather.stride[0] - gather.pad[0];
            const int h = oh * gather.stride[1] - gather.pad[1];
            const int w = ow * gather.stride[2] - gather.pad[2];
            xOff[i] = static_cast<int>(
                ((((img - firstImage) * gather.in[0] + d) * gather.in[1] + h) * gather.in[2] + w)
                    * gather.chans
                + gather.chanOffset);
            unsigned mk = 0;
            for(int q = 0; q < taps0; ++q)
                mk |= (static_cast<unsigned>(d + q * gather.step[0])
                       < static_cast<unsigned>(gather.in[0]))
                      << q;
            for(int q = 0; q < taps1; ++q)
                mk |= (static_cast<unsigned>(h + q * gather.step[1])
                       < static_cast<unsigned>(gather.in[1]))
                      << (10 + q);
            for(int q = 0; q < taps2; ++q)
                mk |= (static_cast<unsigned>(w + q * gather.step[2])
                       < static_cast<unsigned>(gather.in[2]))
                      << (20 + q);
            xMask[i] = mk;
        }
    }
    // W rows of this lane (the GEMM's columns n0 + r).
    const long long filterTaps = static_cast<long long>(a.ft) * a.fr * a.fs;
    const long long wTapStride = DIR == K1_FORWARD  ? cg
                                 : a.wTapStride > 0 ? a.wTapStride
                                                    : static_cast<long long>(cg) * kg;
    const long long wRowStride
        = DIR == K1_FORWARD ? filterTaps * cg : (a.wRowStride > 0 ? a.wRowStride : kg);
    const long long wGroupBase = static_cast<long long>(g) * kg * filterTaps * cg;
    const __amdgpu_buffer_rsrc_t wr = k1Rsrc(
        a.w, static_cast<long long>(a.k) * filterTaps * cg * static_cast<long long>(sizeof(T)));
    int wOff[IPWW];
    bool wOk[IPWW];
#pragma unroll
    for(int i = 0; i < IPWW; ++i)
    {
        const int n = n0 + IPWW * RPI * wave + RPI * i + lane / GR;
        wOk[i] = n < ncols;
        wOff[i] = static_cast<int>(wGroupBase + (wOk[i] ? n * wRowStride : 0));
    }

    // A DMA cursor: the step the next load of one operand fetches, decoded (tap qt, qr, qs;
    // slice c0) and advanced by compare-and-wrap. Steps past the range (the padding to whole loop
    // bodies) load zeros.
    struct Cursor
    {
        int v, qt, qr, qs, c0;
    };
    auto cursorAt = [&](int v) __attribute__((always_inline))
    {
        Cursor c;
        c.v = v;
        const int sl = v / nTaps;
        int tap = v - sl * nTaps;
        c.c0 = sl * KS;
        c.qs = tap % taps2;
        tap /= taps2;
        c.qr = r0 + tap % (r1 - r0);
        c.qt = q0 + tap / (r1 - r0);
        return c;
    };
    auto advance = [&](Cursor & c) __attribute__((always_inline))
    {
        ++c.v;
        if(++c.qs == taps2)
        {
            c.qs = 0;
            if(++c.qr == r1)
            {
                c.qr = r0;
                if(++c.qt == q1)
                {
                    c.qt = q0;
                    c.c0 += KS;
                }
            }
        }
    };
    Cursor cx = cursorAt(vFirst), cw = cx;

    // This wave's share of the X tile of cursor cx's step into stage xs.
    auto dmaX = [&](Raw16 * xs) __attribute__((always_inline))
    {
        const int xDelta
            = ((cx.qt * gather.step[0] * gather.in[1] + cx.qr * gather.step[1]) * gather.in[2]
               + cx.qs * gather.step[2])
                  * gather.chans
              + cx.c0;
        const unsigned bits = (1u << cx.qt) | (1u << (10 + cx.qr)) | (1u << (20 + cx.qs));
        const bool cok = cx.v < vEnd && cx.c0 + gran < red;
#pragma unroll
        for(int i = 0; i < IPW; ++i)
        {
            const bool xok = cok && (xMask[i] & bits) == bits;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                xr,
                (__attribute__((address_space(3))) void*)(xs + (IPW * wave + i) * WAVE),
                16,
                xok ? static_cast<unsigned>((xOff[i] + xDelta + gran) * 2) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
        advance(cx);
    };
    // This wave's share of the W tile of cursor cw's step into stage ws.
    auto dmaW = [&](Raw16 * ws) __attribute__((always_inline))
    {
        const int ft = map.first[0] + cw.qt * map.step[0];
        const int fr = map.first[1] + cw.qr * map.step[1];
        const int fs = map.first[2] + cw.qs * map.step[2];
        const int wDelta = static_cast<int>(((ft * a.fr + fr) * a.fs + fs) * wTapStride) + cw.c0;
        const bool cok = cw.v < vEnd && cw.c0 + gran < red;
#pragma unroll
        for(int i = 0; i < IPWW; ++i)
        {
            const bool wok = cok && wOk[i];
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                wr,
                (__attribute__((address_space(3))) void*)(ws + (IPWW * wave + i) * WAVE),
                16,
                wok ? static_cast<unsigned>((wOff[i] + wDelta + gran) * 2) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
        advance(cw);
    };

    typename Atom::acc acc[FM][FN];
#pragma unroll
    for(int i = 0; i < FM; ++i)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[i][j] = typename Atom::acc{};

    // Fragment reads: lane l takes row base + l % 16 (base a multiple of 16), granule
    // 4 half + l / 16, at slot granule ^ (l % 8).
    const int fr16 = lane % DIM, fg = lane / DIM;
    const int xRow0 = wx * WM + fr16, wRow0 = grp * WN + fr16;
    auto at = [&](int r, int hf) __attribute__((always_inline))
    {
        return r * GR + ((4 * hf + fg) ^ (fr16 % 8));
    };

    // One half of a step: fragment reads of 32 channels from stage (xs, ws), then (mid) DMA issue
    // and waits, then the MMAs at raised priority. Reads go first, ahead of any DMA in the
    // wave's memory queue.
    auto half
        = [&](const Raw16* xs, const Raw16* ws, int hf, auto mid) __attribute__((always_inline))
    {
        Frag xf[FM], wf[FN];
#pragma unroll
        for(int i = 0; i < FM; ++i)
            xf[i] = __builtin_bit_cast(Frag, xs[at(xRow0 + i * DIM, hf)]);
#pragma unroll
        for(int j = 0; j < FN; ++j)
            wf[j] = __builtin_bit_cast(Frag, ws[at(wRow0 + j * DIM, hf)]);
        __builtin_amdgcn_sched_barrier(0);
        mid();
        __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
        k1FlexBarrier();
        __builtin_amdgcn_s_setprio(1);
#pragma unroll
        for(int j = 0; j < FN; ++j)
#pragma unroll
            for(int i = 0; i < FM; ++i)
                acc[i][j] = Atom::mma(wf[j], xf[i], acc[i][j]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(0);
        k1FlexBarrier();
    };
    // Step from X stage xs and W stage ws. The first half's memory phase issues the X tile of
    // the next step into dxs (two phases to land), the second half's the W tile of the step after
    // next into dws (four phases), then waits until only that W tile is in flight.
    auto step = [&](const Raw16* xs, const Raw16* ws, Raw16* dxs, Raw16* dws)
        __attribute__((always_inline))
    {
        half(xs, ws, 0, [&]() __attribute__((always_inline)) { dmaX(dxs); });
        half(
            xs, ws, 1, [&]() __attribute__((always_inline)) {
                dmaW(dws);
                __builtin_amdgcn_s_waitcnt(0x0F70 | IPWW); // vmcnt(IPWW)
            });
    };

    // Prologue: X and W of step vFirst, W of step vFirst + 1. The second group starts one barrier
    // behind.
    dmaX(sx0);
    dmaW(sw0);
    dmaW(sw1);
    __builtin_amdgcn_s_waitcnt(0x0F70 | IPWW);
    k1FlexBarrier();
    if(grp == 1)
        k1FlexBarrier();
    // Step v reads X stage (v - vFirst) % 2 and W stage (v - vFirst) % 3; the range is padded to
    // whole loop bodies of 6 steps (zero stages), so the body has no exits.
    for(int v = vFirst; v < vEnd; v += 6)
    {
        step(sx0, sw0, sx1, sw2);
        step(sx1, sw1, sx0, sw0);
        step(sx0, sw2, sx1, sw1);
        step(sx1, sw0, sx0, sw2);
        step(sx0, sw1, sx1, sw0);
        step(sx1, sw2, sx0, sw1);
    }
    if(grp == 0)
        k1FlexBarrier();
    __builtin_amdgcn_s_waitcnt(0x0F70);

    // Epilogue: lane l holds channels 4 (l / 16) .. + 3 of fragment column j for position l % 16
    // of fragment row i.
    const long long outChans = DIR == K1_FORWARD ? a.k : a.c;
    const long long chanBase = static_cast<long long>(g) * ncols;
    const int nl = n0 + grp * WN + 4 * fg;
    const bool vec = (ncols & 3) == 0;
    if(splits > 1)
    {
        float* partial = a.partial + static_cast<long long>(split) * a.outElems;
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
            const long long m = m0 + wx * WM + i * DIM + fr16;
            if(m >= gather.rows)
                continue;
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int n = nl + j * DIM;
                float* o = partial + (m - static_cast<long long>(a.partialTile) * BM) * outChans
                           + chanBase + n;
                if(vec && n + 3 < ncols)
                    *reinterpret_cast<typename Atom::acc*>(o) = acc[i][j];
                else
#pragma unroll
                    for(int e = 0; e < 4; ++e)
                        if(n + e < ncols)
                            o[e] = acc[i][j][e];
            }
        }
        return;
    }
    T* out = static_cast<T*>(DIR == K1_FORWARD ? a.y : a.x);
#if defined(CONV_EPI) && CONV_EPI
    ConvEpiGen::Chan chans[FN][4];
#pragma unroll
    for(int j = 0; j < FN; ++j)
#pragma unroll
        for(int e = 0; e < 4; ++e)
        {
            const int n = nl + j * DIM + e;
            chans[j][e] = epi.chan(static_cast<int>(chanBase) + n, n < ncols);
        }
#endif
    typedef T v4 __attribute__((ext_vector_type(4)));
    auto rounded = [&](int i, int j) __attribute__((always_inline))
    {
        v4 o;
#pragma unroll
        for(int e = 0; e < 4; ++e)
#if defined(CONV_EPI) && CONV_EPI
            o[e] = K1Elem<T>::store(epi(acc[i][j][e], chans[j][e], 0, true));
#else
            o[e] = K1Elem<T>::store(acc[i][j][e]);
#endif
        return o;
    };
    // Whole 16-byte runs of output channels: each wave stages its 64 positions x WN channels in
    // LDS (the stages are free: every read and DMA is done), granule g of position p at slot
    // g ^ key(p), then stores each position's channels with GO lanes of 16 bytes. The key spreads
    // a store instruction's 16 positions over distinct bank slots.
    // (A wave's 96 channels of BN 192 are 12 granules a row: such tiles store directly.)
    constexpr bool STAGED = BN != 192;
    if(STAGED && (ncols & 7) == 0 && (outChans & 7) == 0 && (chanBase & 7) == 0)
    {
        constexpr int GO = STAGED ? WN * 2 / 16 : 8; // granules a staged row (16 or 8)
        constexpr int PO = WAVE / GO; // positions a store instruction
        auto key = [](int p) __attribute__((always_inline))
        {
            return GO == 16 ? (p % 8) * 2 + (p / 8) % 2 : p % 8;
        };
        Raw16* region = BN == 256 ? (wave < 2   ? sx0
                                     : wave < 4 ? sx1
                                     : wave < 6 ? sw0
                                                : sw1)
                                        + (wave % 2) * (NG / 2)
                                  : (wave < 4 ? sx0 : sx1) + (wave % 4) * (NG / 4);
        k1FlexBarrier();
        const int hfg = fg / 2, hh = (fg % 2) * 8; // granule within 16 channels, byte half
#pragma unroll
        for(int i = 0; i < FM; ++i)
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int p = i * DIM + fr16;
                *reinterpret_cast<v4*>(
                    reinterpret_cast<char*>(region + p * GO + ((2 * j + hfg) ^ key(p))) + hh)
                    = rounded(i, j);
            }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        const int gl = lane % GO, pl = lane / GO;
        const int nn = n0 + grp * WN + gl * 8;
#pragma unroll
        for(int q = 0; q < WM / PO; ++q)
        {
            const int p = q * PO + pl;
            const long long m = m0 + wx * WM + p;
            const Raw16 v = region[p * GO + (gl ^ key(p))];
            if(m < gather.rows && nn < ncols)
                *reinterpret_cast<Raw16*>(out + m * outChans + chanBase + nn) = v;
        }
        return;
    }
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
        const long long m = m0 + wx * WM + i * DIM + fr16;
        if(m >= gather.rows)
            continue;
#pragma unroll
        for(int j = 0; j < FN; ++j)
        {
            const int n = nl + j * DIM;
            const v4 o = rounded(i, j);
            T* y = out + m * outChans + chanBase + n;
            if(vec && n + 3 < ncols)
                *reinterpret_cast<v4*>(y) = o;
            else
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    if(n + e < ncols)
                        y[e] = o[e];
        }
    }
}

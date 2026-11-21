// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K1 flex tile, register-staged form (CONV_IGEMM_FLEX 2: a 16x16x16 or 16x16x4 atom): the
// flex tile's GEMM over flattened output positions for parts with 64 KiB of LDS and dword LDS DMA
// (gfx942). Included by ConvIgemm.cpp after ConvIgemmFlex.h (k1FlexBarrier, spatialOf).
// HIPRTC-compatible.
//
// GEMM and steps as in ConvIgemmFlex.h: rows m are 256 flattened output positions, columns BN
// output channels of one group, a step is one 128-byte slice of reduction channels (64 16-bit,
// 32 fp32) at one filter tap, slices outer and taps (t, r, s) inner.
//
// One LDS stage: X (256 rows) and W (BN rows) of 128-byte rows, eight 16-byte granules a row,
// granule g of row r at slot g ^ (r % 8): a fragment read of 16 consecutive rows covers the
// banks once, and a thread's stored granule is one aligned 16-byte write. The next step's tile is
// prefetched in registers (16-byte buffer loads, four X and BN / 64 W granules a thread) one step
// ahead, so a load has a whole step to land.
//
// Waves: two groups of four, the BN / 2-channel halves of the block; a group's four waves split
// the 256 positions, so a wave owns 64 positions x BN / 2 channels (the square wave tile reads
// the fewest LDS bytes a step: 64 X and 64 W rows). A step is four phases,
// each ended by a barrier, and the second group runs one phase behind the first:
//   [mem A]     every fragment read of the step (two 64-byte halves of the slice);
//   [compute A] the first half's MMAs at raised priority;
//   [mem B]     the group's share of the next step's tile from registers into the stage (the
//               other group has finished its reads of this step one phase earlier), then the
//               loads of the step after next;
//   [compute B] the second half's MMAs.
// So on every SIMD one wave multiplies while the other reads or writes LDS.
//
// A 16-byte granule holds GE = 16 / sizeof(T) consecutive channels; lane group f (lane / 16)
// reads granule 4 h + f of half h, and each of its GE / KPL MMAs takes KPL of those channels in
// both operands (the reduction's order inside a slice is the same for both, so the sum is the
// convolution's).
//
// Split-K (a.splits > 1): as the flex tile, split z runs a contiguous range of the steps and
// writes fp32 partials that ConvIgemmReduce adds in split order. A launch of the trailing row
// tiles only (the tail split) indexes its partials from row tile a.partialTile.
//
// Direct filter (WD 1, backward data): W comes from the graph's K T R S C/g filter, where a
// step's reduction channels (output channels k) are rows and the tile's columns (input
// channels) run contiguously, instead of from the transposed copy. A thread's unit is KV
// consecutive k rows (a dword: two 16-bit or one fp32 channel) of one 16-byte column granule:
// it loads the KV row granules and stores each column's dword into the column's row of the same
// stage, so the stage and the MMAs are the forward form's. The 32 units of a column granule are
// 32 lanes of one wave, so a dword store covers the 32 banks once.
#pragma once

template <class T, class Atom, int DIR, int BN, int WD = 0>
__device__ __attribute__((always_inline)) void k1ConvFlexReg(const K1Args& a CONV_EPI_FN_PARAM)
{
    constexpr int WAVE = 64, DIM = 16;
    constexpr int BM = 256;
    constexpr int THREADS = 512;
    constexpr int ES = static_cast<int>(sizeof(T));
    constexpr int GE = 16 / ES; // elements a granule
    constexpr int GR = 8; // granules a row (128 bytes)
    constexpr int KS = GR * GE; // reduction channels a step
    constexpr int NG = BM * GR, NGW = BN * GR; // granules of the X and W tiles
    constexpr int XL = NG / THREADS, WL = NGW / THREADS; // granules a thread: X, W
    constexpr int WM = BM / 4, WN = BN / 2; // a wave's output positions and channels
    constexpr int FM = WM / DIM, FN = WN / DIM; // a wave's position x channel fragments
    constexpr int MPG = GE / Atom::KPL; // MMAs a granule
    // Direct filter: KV k rows (one dword) by one column granule a unit, KQ units a granule.
    constexpr bool DIRECT = DIR == K1_BACKWARD_DATA && WD != 0;
    constexpr int KV = 4 / ES;
    constexpr int KQ = KS / KV;
    constexpr int WU = DIRECT ? KQ * (BN / GE) / THREADS : 0; // units a thread
    constexpr int WR = DIRECT ? WU * KV : WL; // W registers a thread
    typedef typename K1Raw<16>::type Raw16;
    typedef typename Atom::frag Frag;
    static_assert(Atom::WAVE == WAVE && GE % Atom::KPL == 0 && 4 * Atom::KPL == Atom::KA,
                  "16-row atoms, KPL elements a lane group");
    static_assert((BN == 256 || BN == 128) && XL == 4 && WL >= 1, "tile");
    static_assert(static_cast<int>(sizeof(Frag)) == Atom::KPL * ES, "fragment of KPL elements");
    static_assert(!DIRECT || (KQ == 32 && (KQ * (BN / GE)) % THREADS == 0 && WU >= 1),
                  "direct-filter units: a column granule's 32 dwords on 32 lanes");

    // Forward: W in two stages, alternating per step, so the second half's W fragments are read
    // in [mem B] while the next step's W goes into the other stage.
    constexpr bool WDB = DIR == K1_FORWARD;
    __shared__ Raw16 sx[NG];
    __shared__ Raw16 sw[WDB ? 2 * NGW : NGW];

    // One output tile (row tile bx, column tile by, group gz): its steps [total lo / den,
    // total hi / den) of the tile's total, into the output (slot < 0) or into fp32 partials slot
    // (indexed from row tile a.partialTile), and zeros into partials slot + 1 .. zeroTo - 1.
    auto runTile = [&](unsigned bx,
                       unsigned by,
                       int gz,
                       long long lo,
                       long long hi,
                       long long den,
                       int slot,
                       int zeroTo) __attribute__((always_inline))
    {
        K1Gather gather;
        K1TapMap map;
        int g, phase;
        k1Setup<DIR>(a, gz, gather, map, g, phase);
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

        // Steps over the depth and height taps that reach the input for some row of the tile.
        const int taps2 = gather.taps[2];
        int q0 = 0, q1 = gather.taps[0], r0 = 0, r1 = gather.taps[1];
        {
            const long long last = (m0 + BM < gather.rows ? m0 + BM : gather.rows) - 1;
            const K1Pos plane = static_cast<K1Pos>(gather.out[1]) * gather.out[2];
            const long long imgF = m0 / spatialOf(gather), imgL = last / spatialOf(gather);
            const K1Pos restF = static_cast<K1Pos>(m0 - imgF * spatialOf(gather));
            const K1Pos restL = static_cast<K1Pos>(last - imgL * spatialOf(gather));
            const int odF = static_cast<int>(restF / plane), odL = static_cast<int>(restL / plane);
            auto tapRange
                = [&](int ax, int oF, int oL, int& lo, int& hi) __attribute__((always_inline))
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
        const int vFirst = static_cast<int>(static_cast<long long>(total) * lo / den);
        const int vEnd = static_cast<int>(static_cast<long long>(total) * hi / den);

        // This thread's granules: X rows tid / 8 + 64 i, W rows tid / 8 + 64 i, slot tid % 8, which
        // holds granule (tid % 8) ^ (row % 8) of its row (the same granule for every i).
        const T* src = static_cast<const T*>(DIR == K1_FORWARD ? a.x : a.y);
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
        const long long imageElems
            = static_cast<long long>(gather.in[0]) * gather.in[1] * gather.in[2] * gather.chans;
        const long long firstImage = m0 / spatial;
        const long long blockBase = firstImage * imageElems;
        const __amdgpu_buffer_rsrc_t xr = k1Rsrc(
            src + blockBase, (a.n - firstImage) * imageElems * static_cast<long long>(sizeof(T)));
        const int rowT = tid / GR;
        const int gran = ((tid % GR) ^ (rowT % 8)) * GE; // the thread's granule, in elements
        int xOff[XL]; // the row's element offset at tap 0 (relative to the buffer base)
        unsigned xMask[XL]; // in-image taps: bits 0..9 depth, 10..19 height, 20..29 width
#pragma unroll
        for(int i = 0; i < XL; ++i)
        {
            const long long m = m0 + rowT + 64 * i;
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
                    ((((img - firstImage) * gather.in[0] + d) * gather.in[1] + h) * gather.in[2]
                     + w)
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
        const long long filterTaps = static_cast<long long>(a.ft) * a.fr * a.fs;
        const long long wTapStride = DIR == K1_FORWARD  ? cg
                                     : a.wTapStride > 0 ? a.wTapStride
                                                        : static_cast<long long>(cg) * kg;
        const long long wRowStride
            = DIR == K1_FORWARD ? filterTaps * cg : (a.wRowStride > 0 ? a.wRowStride : kg);
        const long long wGroupBase = static_cast<long long>(g) * kg * filterTaps * cg;
        const __amdgpu_buffer_rsrc_t wr = k1Rsrc(
            a.w, static_cast<long long>(a.k) * filterTaps * cg * static_cast<long long>(sizeof(T)));
        int wOff[DIRECT ? WU : WL];
        bool wOk[DIRECT ? WU : WL];
        if constexpr(DIRECT)
        {
            // Unit u = tid + THREADS i: k rows (u % KQ) KV .., column granule u / KQ; the offset of
            // its first k row of slice 0 at tap 0.
#pragma unroll
            for(int i = 0; i < WU; ++i)
            {
                const int u = tid + THREADS * i;
                const int n = n0 + (u / KQ) * GE;
                wOk[i] = n < ncols;
                wOff[i] = static_cast<int>(
                    wOk[i] ? (static_cast<long long>(g) * kg + (u % KQ) * KV) * filterTaps * cg + n
                           : 0);
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < WL; ++i)
            {
                const int n = n0 + rowT + 64 * i;
                wOk[i] = n < ncols;
                wOff[i] = static_cast<int>(wGroupBase + (wOk[i] ? n * wRowStride : 0));
            }
        }

        // The load cursor: the step the next tile load fetches (taps qt, qr, qs; slice c0).
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
        Cursor cur = cursorAt(vFirst);
        auto advance = [&]() __attribute__((always_inline))
        {
            ++cur.v;
            if(++cur.qs == taps2)
            {
                cur.qs = 0;
                if(++cur.qr == r1)
                {
                    cur.qr = r0;
                    if(++cur.qt == q1)
                    {
                        cur.qt = q0;
                        cur.c0 += KS;
                    }
                }
            }
        };
        // The cursor step's X and W granules of this thread into registers (zeros past the range).
        Raw16 rx[XL], rw[WR];
        auto load = [&]() __attribute__((always_inline))
        {
            const bool cok = cur.v < vEnd && cur.c0 + gran < red;
            const int xDelta
                = ((cur.qt * gather.step[0] * gather.in[1] + cur.qr * gather.step[1]) * gather.in[2]
                   + cur.qs * gather.step[2])
                      * gather.chans
                  + cur.c0;
            const unsigned bits = (1u << cur.qt) | (1u << (10 + cur.qr)) | (1u << (20 + cur.qs));
#pragma unroll
            for(int i = 0; i < XL; ++i)
            {
                const bool ok = cok && (xMask[i] & bits) == bits;
                rx[i] = k1BufferLoad<Raw16>(
                    xr,
                    ok ? static_cast<unsigned>((xOff[i] + xDelta + gran) * ES) : K1_OUT_OF_RANGE);
            }
            const int ft = map.first[0] + cur.qt * map.step[0];
            const int fr = map.first[1] + cur.qr * map.step[1];
            const int fs = map.first[2] + cur.qs * map.step[2];
            const int tap = (ft * a.fr + fr) * a.fs + fs;
            if constexpr(DIRECT)
            {
                // k row c0 + (u % KQ) KV + e of the unit's column granule at this tap.
                const int kRow = static_cast<int>(filterTaps) * cg;
                const int wDelta = tap * cg + cur.c0 * kRow;
#pragma unroll
                for(int i = 0; i < WU; ++i)
                {
                    const int k = cur.c0 + ((tid + THREADS * i) % KQ) * KV;
#pragma unroll
                    for(int e = 0; e < KV; ++e)
                    {
                        const bool ok = cur.v < vEnd && wOk[i] && k + e < red;
                        rw[i * KV + e] = k1BufferLoad<Raw16>(
                            wr,
                            ok ? static_cast<unsigned>((wOff[i] + wDelta + e * kRow) * ES)
                               : K1_OUT_OF_RANGE);
                    }
                }
            }
            else
            {
                const int wDelta = static_cast<int>(tap * wTapStride) + cur.c0;
#pragma unroll
                for(int i = 0; i < WL; ++i)
                {
                    const bool ok = cok && wOk[i];
                    rw[i] = k1BufferLoad<Raw16>(
                        wr,
                        ok ? static_cast<unsigned>((wOff[i] + wDelta + gran) * ES)
                           : K1_OUT_OF_RANGE);
                }
            }
            advance();
        };
        // The registers into the stage: granule slot (row, tid % 8) is index tid + 512 i; W into W
        // stage wb.
        auto store = [&](int wb) __attribute__((always_inline))
        {
#pragma unroll
            for(int i = 0; i < XL; ++i)
                sx[tid + THREADS * i] = rx[i];
            if constexpr(DIRECT)
            {
                // Column n's dword of k rows kLoc .. kLoc + KV - 1: granule kLoc / GE of row n
                // (swizzled), dword (kLoc % GE) / KV of it.
#pragma unroll
                for(int i = 0; i < WU; ++i)
                {
                    const int u = tid + THREADS * i;
                    const int kLoc = (u % KQ) * KV, nLoc = (u / KQ) * GE;
#pragma unroll
                    for(int e = 0; e < GE; ++e)
                    {
                        const int n = nLoc + e;
                        unsigned v;
                        if constexpr(KV == 2)
                        {
                            const unsigned lo = rw[i * KV][e / 2], hi = rw[i * KV + 1][e / 2];
                            v = e % 2 == 0 ? (lo & 0xFFFFu) | (hi << 16)
                                           : (lo >> 16) | (hi & 0xFFFF0000u);
                        }
                        else
                        {
                            v = rw[i][e];
                        }
                        reinterpret_cast<unsigned*>(
                            &sw[n * GR + ((kLoc / GE) ^ (n % 8))])[(kLoc % GE) / KV]
                            = v;
                    }
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < WL; ++i)
                    sw[wb * NGW + tid + THREADS * i] = rw[i];
            }
        };

        typename Atom::acc acc[FM][FN];
#pragma unroll
        for(int i = 0; i < FM; ++i)
#pragma unroll
            for(int j = 0; j < FN; ++j)
                acc[i][j] = typename Atom::acc{};

        const int fr16 = lane % DIM, fg = lane / DIM;
        const int xRow0 = wx * WM + fr16, wRow0 = grp * WN + fr16;
        auto at = [&](int r, int hf) __attribute__((always_inline))
        {
            return r * GR + ((4 * hf + fg) ^ (r % 8));
        };
        Raw16 xf[2][FM], wf[2][FN];
        // The MMAs of half hf: each granule pair (W, X) gives MPG MMAs of KPL channels.
        auto mmas = [&](int hf) __attribute__((always_inline))
        {
            __builtin_amdgcn_s_setprio(1);
#pragma unroll
            for(int e = 0; e < MPG; ++e)
#pragma unroll
                for(int j = 0; j < FN; ++j)
                {
                    Frag wv;
                    __builtin_memcpy(&wv,
                                     reinterpret_cast<const char*>(&wf[hf][j]) + e * Atom::KPL * ES,
                                     sizeof(Frag));
#pragma unroll
                    for(int i = 0; i < FM; ++i)
                    {
                        Frag xv;
                        __builtin_memcpy(&xv,
                                         reinterpret_cast<const char*>(&xf[hf][i])
                                             + e * Atom::KPL * ES,
                                         sizeof(Frag));
                        acc[i][j] = Atom::mma(wv, xv, acc[i][j]);
                    }
                }
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
        };

        // Prologue: step vFirst into the stage, step vFirst + 1 into the registers. The second group
        // starts one phase behind.
        load();
        __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
        store(0);
        load();
        __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
        k1FlexBarrier();
        if(grp == 1)
            k1FlexBarrier();
        for(int v = vFirst; v < vEnd; ++v)
        {
            // The W stage this step reads (always 0 with one W stage).
            const int wb = WDB ? (v - vFirst) & 1 : 0;
            // [mem A]: X of both halves, W of the first (of both with one W stage).
#pragma unroll
            for(int hf = 0; hf < 2; ++hf)
            {
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    xf[hf][i] = sx[at(xRow0 + i * DIM, hf)];
                if(!WDB || hf == 0)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        wf[hf][j] = sw[wb * NGW + at(wRow0 + j * DIM, hf)];
            }
            __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
            k1FlexBarrier();
            // [compute A]
            mmas(0);
            k1FlexBarrier();
            // [mem B]: the second half's W (two W stages), the next step's tile (loaded one step ago)
            // into the stages, then the loads of the step after next, so the compute phases issue
            // only MMAs.
            if constexpr(WDB)
            {
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    wf[1][j] = sw[wb * NGW + at(wRow0 + j * DIM, 1)];
            }
            __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
            store(wb ^ (WDB ? 1 : 0));
            load();
            __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
            k1FlexBarrier();
            // [compute B]: the second half.
            mmas(1);
            k1FlexBarrier();
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
        if(slot >= 0)
        {
            for(int sl = slot; sl < zeroTo || sl == slot; ++sl)
            {
                float* partial = a.partial + static_cast<long long>(sl) * a.outElems;
                const bool zero = sl != slot;
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
                        float* o = partial
                                   + (m - static_cast<long long>(a.partialTile) * BM) * outChans
                                   + chanBase + n;
                        const typename Atom::acc v = zero ? typename Atom::acc{} : acc[i][j];
                        if(vec && n + 3 < ncols)
                            *reinterpret_cast<typename Atom::acc*>(o) = v;
                        else
#pragma unroll
                            for(int e = 0; e < 4; ++e)
                                if(n + e < ncols)
                                    o[e] = v[e];
                    }
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
                v4 o;
#pragma unroll
                for(int e = 0; e < 4; ++e)
#if defined(CONV_EPI) && CONV_EPI
                    o[e] = K1Elem<T>::store(epi(acc[i][j][e], chans[j][e], 0, true));
#else
                    o[e] = K1Elem<T>::store(acc[i][j][e]);
#endif
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
    };

    unsigned bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    k1XcdOrder(bx, by, bz, a.xcds);
    if constexpr(DIR == K1_FORWARD)
    {
        if(a.streamUnits > 0)
        {
            // Stream split (one group, fewer tiles than blocks): the G = streamUnits blocks
            // share the T x KT (tile, step) pairs evenly, tiles in order (column tiles fastest),
            // KT a tile's steps before trimming out-of-image taps. Block b runs [b W / G,
            // (b + 1) W / G) of W = T KT, a step range of each tile it reaches. A tile's ranges
            // go to partials slots 0, 1, .. in block order (every tile spans two blocks or more),
            // and its last block zeros the slots up to a.splits, so ConvIgemmReduce adds a.splits
            // slots in order.
            const long long kt = static_cast<long long>(a.ft) * a.fr * a.fs * ((a.c + KS - 1) / KS);
            const long long tilesY = (a.k + BN - 1) / BN;
            const long long tiles = ((a.m + BM - 1) / BM) * tilesY;
            const long long units = a.streamUnits;
            const long long total = tiles * kt;
            const long long b = bx;
            const long long start = b * total / units, end = (b + 1) * total / units;
            // The block whose range holds pair p.
            auto blockOf = [&](long long p) __attribute__((always_inline))
            {
                return ((p + 1) * units - 1) / total;
            };
            for(long long t = start / kt; t * kt < end; ++t)
            {
                const long long t0 = t * kt;
                const long long first = blockOf(t0), last = blockOf(t0 + kt - 1);
                const long long lo = (start > t0 ? start : t0) - t0;
                const long long hi = (end < t0 + kt ? end : t0 + kt) - t0;
                runTile(static_cast<unsigned>(t / tilesY),
                        static_cast<unsigned>(t % tilesY),
                        0,
                        lo,
                        hi,
                        kt,
                        static_cast<int>(b - first),
                        b == last ? a.splits : 0);
            }
            return;
        }
    }
    const int splits = a.splits > 1 ? a.splits : 1;
    const int split = static_cast<int>(bz) % splits;
    runTile(bx,
            by,
            static_cast<int>(bz) / splits,
            split,
            split + 1,
            splits,
            splits > 1 ? split : -1,
            0);
}

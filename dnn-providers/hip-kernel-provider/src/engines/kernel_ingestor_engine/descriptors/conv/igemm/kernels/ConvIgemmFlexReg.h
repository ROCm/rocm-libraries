// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K1 flex tile, register-staged form (CONV_IGEMM_FLEX 2: a 16x16x16 or 16x16x4 atom): the
// flex tile's GEMM over flattened output positions for parts with 64 KiB of LDS and dword LDS DMA
// (gfx942). Included by ConvIgemm.cpp after ConvIgemmFlex.h (k1FlexBarrier, spatialOf).
// HIPRTC-compatible.
//
// GEMM and steps as in ConvIgemmFlex.h: rows m are 256 flattened output positions, columns BN
// output channels of one group, a step is one KB-byte slice of reduction channels at one filter
// tap, slices outer and taps (t, r, s) inner. KB is 128 (64 16-bit or 32 fp32 channels), or 64 on
// the 16-bit 256 x 256 tile, whose 64 MMAs a wave a step read a third fewer operand bytes.
//
// One LDS stage: X (256 rows) and W (BN rows) of KB-byte rows, GR = KB / 16 granules of 16 bytes
// a row, granule g of row r at slot g ^ key(r) (r % 8, or (r / 2) % 4 for four granules a row): a
// fragment read of 16 consecutive rows covers the banks once, and a thread's stored granule is one
// aligned 16-byte write. The next step's tile is prefetched in registers (16-byte buffer loads)
// one step ahead, so a load has a whole step to land.
//
// Waves: two groups of four, the BN / 2-channel halves of the block; a group's four waves split
// the 256 positions, so a wave owns 64 positions x BN / 2 channels (the square wave tile reads
// the fewest LDS bytes a step: 64 X and 64 W rows). A step is four phases,
// each ended by a barrier, and the second group runs one phase behind the first:
//   [mem A]     every fragment read of the step;
//   [compute A] the first half's MMAs at raised priority;
//   [mem B]     the group's share of the next step's tile from registers into the stage (the
//               other group has finished its reads of this step one phase earlier), then the
//               loads of the step after next (forward: their offsets);
//   [compute B] the second half's MMAs; forward issues the step after next's loads here, one
//               per five MMAs, which kept [mem B] short and measured 1.03-1.09x on MI300A.
// So on every SIMD one wave multiplies while the other reads or writes LDS.
//
// A 16-byte granule holds GE = 16 / sizeof(T) consecutive channels; lane group f (lane / 16)
// reads granule 4 h + f of granule group h (two groups a 128-byte row, one a 64-byte row), and
// each of its GE / KPL MMAs takes KPL of those channels in both operands (the reduction's order
// inside a slice is the same for both, so the sum is the convolution's). The two compute phases
// take one granule group each, or, with one group, half of each granule's MMAs.
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

// Box-ordered rows (BOX 1): row m's dx pixel. The rows run class by class (K1BoxArgs), then
// image, then the class box's (d, h, w); cls gets the class's run per axis. 32-bit arithmetic:
// the pack serves box configs only where the rows stay below 2^31.
struct K1BoxPos
{
    int pix; // linear dx pixel (n, d, h, w)
    int img, d, h, w;
};
__device__ inline K1BoxPos
    k1BoxDecode(const K1BoxArgs& box, int m, int images, int dd, int hh, int ww, int (&cls)[3])
{
    int rem = m;
    int unit = images * hh * ww; // rows a d coordinate
    int j0 = 0;
#pragma unroll
    for(int j = 0; j < K1_BOX_RUNS - 1; ++j)
    {
        const int rows = unit * box.size[0][j0];
        if(j0 + 1 < box.runs[0] && rem >= rows)
        {
            rem -= rows;
            ++j0;
        }
    }
    unit = images * box.size[0][j0] * ww; // rows an h coordinate
    int j1 = 0;
#pragma unroll
    for(int j = 0; j < K1_BOX_RUNS - 1; ++j)
    {
        const int rows = unit * box.size[1][j1];
        if(j1 + 1 < box.runs[1] && rem >= rows)
        {
            rem -= rows;
            ++j1;
        }
    }
    unit = images * box.size[0][j0] * box.size[1][j1]; // rows a w coordinate
    int j2 = 0;
#pragma unroll
    for(int j = 0; j < K1_BOX_RUNS - 1; ++j)
    {
        const int rows = unit * box.size[2][j2];
        if(j2 + 1 < box.runs[2] && rem >= rows)
        {
            rem -= rows;
            ++j2;
        }
    }
    cls[0] = j0;
    cls[1] = j1;
    cls[2] = j2;
    const int sd = box.size[0][j0], sh = box.size[1][j1], sw = box.size[2][j2];
    const int boxPix = sd * sh * sw;
    K1BoxPos p;
    p.img = rem / boxPix;
    int r = rem - p.img * boxPix;
    const int rq = r / sw;
    p.w = box.begin[2][j2] + r - rq * sw;
    const int rr = rq / sh;
    p.h = box.begin[1][j1] + rq - rr * sh;
    p.d = box.begin[0][j0] + rr;
    p.pix = ((p.img * dd + p.d) * hh + p.h) * ww + p.w;
    return p;
}

template <class T, class Atom, int DIR, int BN, int WD = 0, int KB = 128, int BOX = 0>
__device__ __attribute__((always_inline)) void k1ConvFlexReg(const K1Args& a CONV_EPI_FN_PARAM,
                                                             const K1BoxArgs* box = nullptr)
{
    constexpr int WAVE = 64, DIM = 16;
    constexpr int BM = 256;
    constexpr int THREADS = 512;
    constexpr int ES = static_cast<int>(sizeof(T));
    constexpr int GE = 16 / ES; // elements a granule
    constexpr int GR = KB / 16; // granules a row (KB bytes: 8 or 4)
    constexpr int NH = GR / 4; // granule groups of the four lane groups a row
    constexpr int RPP = THREADS / GR; // rows a pass of the block's granule loads
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
    static_assert((BN == 256 || BN == 128) && (GR == 8 || GR == 4) && XL >= 1 && WL >= 1
                      && (!DIRECT || GR == 8),
                  "tile");
    static_assert(static_cast<int>(sizeof(Frag)) == Atom::KPL * ES, "fragment of KPL elements");
    static_assert(!DIRECT || (KQ == 32 && (KQ * (BN / GE)) % THREADS == 0 && WU >= 1),
                  "direct-filter units: a column granule's 32 dwords on 32 lanes");

    __shared__ Raw16 sx[NG];
    __shared__ Raw16 sw[NGW];
    // The slot key of row r: granule g of row r sits at slot g ^ key(r), so the 16-byte reads of
    // 16 consecutive rows by one lane group cover the banks once.
    auto key = [](int r) __attribute__((always_inline))
    {
        return GR == 8 ? r % 8 : (r >> 1) % 4;
    };

    K1Gather gather;
    K1TapMap map;
    int g, phase;
    unsigned bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    k1XcdOrder(bx, by, bz, a.xcds);
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

    // Steps over the depth and height taps that reach the input for some row of the tile (box
    // order: the union of the tap ranges of the classes the tile's rows belong to, along every
    // axis; widths s0..s1).
    const int taps2 = gather.taps[2];
    int q0 = 0, q1 = gather.taps[0], r0 = 0, r1 = gather.taps[1];
    int s0 = 0, s1 = taps2;
    static_assert(BOX == 0 || DIR == K1_BACKWARD_DATA, "box rows: backward data");
    if constexpr(BOX != 0)
    {
        const long long last = (m0 + BM < gather.rows ? m0 + BM : gather.rows) - 1;
        int cf[3], cl[3];
        k1BoxDecode(
            *box, static_cast<int>(m0), a.n, gather.out[0], gather.out[1], gather.out[2], cf);
        k1BoxDecode(
            *box, static_cast<int>(last), a.n, gather.out[0], gather.out[1], gather.out[2], cl);
        int lo[3], hi[3];
#pragma unroll
        for(int ax = 0; ax < 3; ++ax)
        {
            // Classes between the first and last row's along this axis: the runs from the
            // first's to the last's where the outer axes' runs agree, every run otherwise.
            bool same = true;
#pragma unroll
            for(int o = 0; o < ax; ++o)
                same = same && cf[o] == cl[o];
            const int jf = same ? cf[ax] : 0, jl = same ? cl[ax] : box->runs[ax] - 1;
            lo[ax] = gather.taps[ax];
            hi[ax] = -1;
#pragma unroll
            for(int j = 0; j < K1_BOX_RUNS; ++j)
            {
                if(j >= jf && j <= jl)
                {
                    lo[ax] = box->lo[ax][j] < lo[ax] ? box->lo[ax][j] : lo[ax];
                    hi[ax] = box->hi[ax][j] > hi[ax] ? box->hi[ax][j] : hi[ax];
                }
            }
        }
        q0 = __builtin_amdgcn_readfirstlane(lo[0]);
        q1 = __builtin_amdgcn_readfirstlane(hi[0] + 1);
        r0 = __builtin_amdgcn_readfirstlane(lo[1]);
        r1 = __builtin_amdgcn_readfirstlane(hi[1] + 1);
        s0 = __builtin_amdgcn_readfirstlane(lo[2]);
        s1 = __builtin_amdgcn_readfirstlane(hi[2] + 1);
    }
    else
    {
        const long long last = (m0 + BM < gather.rows ? m0 + BM : gather.rows) - 1;
        const K1Pos plane = static_cast<K1Pos>(gather.out[1]) * gather.out[2];
        const long long imgF = m0 / spatialOf(gather), imgL = last / spatialOf(gather);
        const K1Pos restF = static_cast<K1Pos>(m0 - imgF * spatialOf(gather));
        const K1Pos restL = static_cast<K1Pos>(last - imgL * spatialOf(gather));
        const int odF = static_cast<int>(restF / plane), odL = static_cast<int>(restL / plane);
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
    const int nTaps = (q1 - q0) * (r1 - r0) * (BOX != 0 ? s1 - s0 : taps2);
    const int total = nTaps * ((red + KS - 1) / KS);
    auto bound = [&](int z) __attribute__((always_inline))
    {
        return z >= splits ? total : static_cast<int>(static_cast<long long>(total) * z / splits);
    };
    const int vFirst = bound(split);
    const int vEnd = bound(split + 1);

    // This thread's granules: X rows tid / 8 + 64 i, W rows tid / 8 + 64 i, slot tid % 8, which
    // holds granule (tid % 8) ^ (row % 8) of its row (the same granule for every i).
    const T* src = static_cast<const T*>(DIR == K1_FORWARD ? a.x : a.y);
    const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
    const long long imageElems
        = static_cast<long long>(gather.in[0]) * gather.in[1] * gather.in[2] * gather.chans;
    // Box rows come from any image: the buffer starts at the first.
    const long long firstImage = BOX != 0 ? 0 : m0 / spatial;
    const long long blockBase = firstImage * imageElems;
    const __amdgpu_buffer_rsrc_t xr = k1Rsrc(
        src + blockBase, (a.n - firstImage) * imageElems * static_cast<long long>(sizeof(T)));
    const int rowT = tid / GR;
    const int gran = ((tid % GR) ^ key(rowT)) * GE; // the thread's granule, in elements
    int xOff[XL]; // the row's element offset at tap 0 (relative to the buffer base)
    unsigned xMask[XL]; // in-image taps: bits 0..9 depth, 10..19 height, 20..29 width
#pragma unroll
    for(int i = 0; i < XL; ++i)
    {
        const long long m = m0 + rowT + RPP * i;
        xOff[i] = 0;
        xMask[i] = 0;
        if(m < gather.rows)
        {
            long long img;
            int od, oh, ow;
            if constexpr(BOX != 0)
            {
                int cls[3];
                const K1BoxPos p = k1BoxDecode(*box,
                                               static_cast<int>(m),
                                               a.n,
                                               gather.out[0],
                                               gather.out[1],
                                               gather.out[2],
                                               cls);
                img = p.img;
                od = p.d;
                oh = p.h;
                ow = p.w;
            }
            else
            {
                img = m / spatial;
                K1Pos rest = static_cast<K1Pos>(m - img * spatial);
                ow = static_cast<int>(rest % gather.out[2]);
                rest /= gather.out[2];
                oh = static_cast<int>(rest % gather.out[1]);
                od = static_cast<int>(rest / gather.out[1]);
            }
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
            const int n = n0 + rowT + RPP * i;
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
        if constexpr(BOX != 0)
        {
            c.qs = s0 + tap % (s1 - s0);
            tap /= s1 - s0;
        }
        else
        {
            c.qs = tap % taps2;
            tap /= taps2;
        }
        c.qr = r0 + tap % (r1 - r0);
        c.qt = q0 + tap / (r1 - r0);
        return c;
    };
    Cursor cur = cursorAt(vFirst);
    auto advance = [&]() __attribute__((always_inline))
    {
        if constexpr(BOX != 0)
        {
            ++cur.v;
            if(++cur.qs == s1)
            {
                cur.qs = s0;
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
        }
        else
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
                xr, ok ? static_cast<unsigned>((xOff[i] + xDelta + gran) * ES) : K1_OUT_OF_RANGE);
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
                    ok ? static_cast<unsigned>((wOff[i] + wDelta + gran) * ES) : K1_OUT_OF_RANGE);
            }
        }
        advance();
    };
    // The registers into the stage: granule slot (row, tid % 8) is index tid + 512 i.
    auto store = [&]() __attribute__((always_inline))
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
                sw[tid + THREADS * i] = rw[i];
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
        return r * GR + ((4 * hf + fg) ^ key(r));
    };
    Raw16 xf[NH][FM], wf[NH][FN];
    // Forward spreads the loads of the step after next over [compute B]: their offsets in
    // [mem B], then one load per PSF MMAs (backward data issues them in [mem B]).
    constexpr bool SPREAD = DIR == K1_FORWARD;
    constexpr int PSF = 5;
    unsigned off[XL + WL];
    auto offsets = [&]() __attribute__((always_inline))
    {
        const bool cok = (cur.v < vEnd) & (cur.c0 + gran < red);
        const int xDelta
            = ((cur.qt * gather.step[0] * gather.in[1] + cur.qr * gather.step[1]) * gather.in[2]
               + cur.qs * gather.step[2])
                  * gather.chans
              + cur.c0;
        const unsigned bits = (1u << cur.qt) | (1u << (10 + cur.qr)) | (1u << (20 + cur.qs));
#pragma unroll
        for(int i = 0; i < XL; ++i)
        {
            const bool ok = cok & ((xMask[i] & bits) == bits);
            off[i] = ok ? static_cast<unsigned>((xOff[i] + xDelta + gran) * ES) : K1_OUT_OF_RANGE;
        }
        const int ft = map.first[0] + cur.qt * map.step[0];
        const int fr = map.first[1] + cur.qr * map.step[1];
        const int fs = map.first[2] + cur.qs * map.step[2];
        const int tap = (ft * a.fr + fr) * a.fs + fs;
        const int wDelta = static_cast<int>(tap * wTapStride) + cur.c0;
#pragma unroll
        for(int i = 0; i < WL; ++i)
        {
            const bool ok = cok & wOk[i];
            off[XL + i]
                = ok ? static_cast<unsigned>((wOff[i] + wDelta + gran) * ES) : K1_OUT_OF_RANGE;
        }
        advance();
    };
    auto issue = [&](int q) __attribute__((always_inline))
    {
        if(q < XL)
            rx[q] = k1BufferLoad<Raw16>(xr, off[q]);
        else
            rw[q - XL] = k1BufferLoad<Raw16>(wr, off[q]);
    };
    // The MMAs of compute phase p: granule group p (two groups a row) or the p-th half of each
    // granule's MPG MMAs (one group); each granule pair (W, X) gives MMAs of KPL channels. With
    // loads, one per PSF MMAs.
    auto mmas = [&](int p, bool loads) __attribute__((always_inline))
    {
        __builtin_amdgcn_s_setprio(1);
        const int hf = NH == 2 ? p : 0;
        constexpr int EN = NH == 2 ? MPG : MPG / 2;
        const int e0 = NH == 2 ? 0 : p * EN;
        int n = 0, q = 0;
#pragma unroll
        for(int e = e0; e < e0 + EN; ++e)
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
                                     reinterpret_cast<const char*>(&xf[hf][i]) + e * Atom::KPL * ES,
                                     sizeof(Frag));
                    acc[i][j] = Atom::mma(wv, xv, acc[i][j]);
                    if(loads && ++n % PSF == 0 && q < XL + WL)
                    {
                        __builtin_amdgcn_sched_barrier(0);
                        issue(q++);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                }
            }
        if(loads)
        {
#pragma unroll
            for(; q < XL + WL; ++q)
                issue(q);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(0);
    };

    // Prologue: step vFirst into the stage, step vFirst + 1 into the registers. The second group
    // starts one phase behind.
    load();
    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
    store();
    load();
    __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
    k1FlexBarrier();
    if(grp == 1)
        k1FlexBarrier();
    for(int v = vFirst; v < vEnd; ++v)
    {
        // [mem A]
#pragma unroll
        for(int hf = 0; hf < NH; ++hf)
        {
#pragma unroll
            for(int i = 0; i < FM; ++i)
                xf[hf][i] = sx[at(xRow0 + i * DIM, hf)];
#pragma unroll
            for(int j = 0; j < FN; ++j)
                wf[hf][j] = sw[at(wRow0 + j * DIM, hf)];
        }
        __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
        k1FlexBarrier();
        // [compute A]
        mmas(0, false);
        k1FlexBarrier();
        // [mem B]: the next step's tile (loaded one step ago) into the stage, then the loads of the
        // step after next (forward: their offsets).
        __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
        store();
        if constexpr(SPREAD)
            offsets();
        else
            load();
        __builtin_amdgcn_s_waitcnt(0xC07F); // lgkmcnt(0)
        k1FlexBarrier();
        // [compute B]: the second half (forward: with the loads of the step after next).
        mmas(1, SPREAD);
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
    if(splits > 1)
    {
        float* partial = a.partial + static_cast<long long>(split) * a.outElems;
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
            const long long m = m0 + wx * WM + i * DIM + fr16;
            if(m >= gather.rows)
                continue;
            // Box rows: the partials are indexed by dx pixel, as the output (no tail split).
            long long pm = m - static_cast<long long>(a.partialTile) * BM;
            if constexpr(BOX != 0)
            {
                int cls[3];
                pm = k1BoxDecode(*box,
                                 static_cast<int>(m),
                                 a.n,
                                 gather.out[0],
                                 gather.out[1],
                                 gather.out[2],
                                 cls)
                         .pix;
            }
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int n = nl + j * DIM;
                float* o = partial + pm * outChans + chanBase + n;
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
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
        const long long m = m0 + wx * WM + i * DIM + fr16;
        if(m >= gather.rows)
            continue;
        long long pm = m;
        if constexpr(BOX != 0)
        {
            int cls[3];
            pm = k1BoxDecode(*box,
                             static_cast<int>(m),
                             a.n,
                             gather.out[0],
                             gather.out[1],
                             gather.out[2],
                             cls)
                     .pix;
        }
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
            T* y = out + pm * outChans + chanBase + n;
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

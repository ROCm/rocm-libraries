// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The direct loop (loop 5): gfx950, 16-bit, channels-last, unit height and width stride and
// unit height dilation (hipconv's direct_wgrad design, written for K3):
// - GEMM: M = input channels (x, the A operand), N = output channels (dy, B), reduction = 32
//   output columns per mfma_f32_16x16x32; one iteration per input row, the x row read at every
//   filter-width shift (DW apart), the filter height's dy rows held in a register ring.
// - An item is one image plane's 32-column strip over a row segment (whole height unless the
//   planner splits it; a segment's first R - 1 iterations only fill the ring). A workgroup of 8
//   waves owns a channel tile (group, filter depth tap, c, k) and walks its items WQ at a time:
//   waves_q, the item's waves split the channels, the WQ items run side by side and their sums
//   meet in item order through LDS.
// - Rows reach LDS by 16-byte DMA into P + 1 ring slots, each its own __shared__ object (so the
//   transpose reads' waits stay exact). Channel groups are XOR-rotated by column (ConvWrwGeom.h)
//   so the ds_read_b64_tr_b16 reads are conflict-free; the DMA fetch address is the rotation's
//   inverse.
// - PP: an 8-wave ping-pong, two wavegroups a half step apart; a memory phase is transpose reads
//   and DMA issue only, a compute phase MFMAs only (s_setprio, raw s_barrier).
// - A persistent grid of about one workgroup per CU walks (tile, item) units, in cells (tile
//   fastest, for few tiles) or in equal runs (for many), XCD-contiguous.
// fp32 partials [split][g][tap][k][c] as the other loops, reduced by ConvWrwReduce.
#pragma once

/// A direct-loop row of C channels read at column shifts 0 .. SHIFTS - 1 (ConvWrwGeom.h's
/// rotation; none when SWZ is 0, the ablation's plain rows).
template <int C, int SHIFTS, int SWZ>
struct K3DSwz
{
    static constexpr int C4 = C / 4;
    static constexpr int ROT = SWZ ? k3dRotation(C, SHIFTS) : -1;
    static_assert(!SWZ || ROT >= 0, "K3 direct: no conflict-free rotation for this row");
    __device__ static constexpr int g(int x)
    {
        return ROT < 0 ? 0 : k3dRot(ROT, C, x);
    }
    /// Element offset of column x, channel group c4 (4 channels) in a row.
    __device__ static constexpr int at(int x, int c4)
    {
        return 4 * (x * C4 + (c4 ^ (4 * g(x))));
    }
};

template <int BC, int BK, int WQ, int S, int DW>
struct K3DGeom
{
    static constexpr int ROUND = 512; // one wave's 16-byte DMA: 64 lanes x 8 elements
    static constexpr int WIN = 32 + (S - 1) * DW; // x columns a strip's taps reach
    static constexpr int SCOLS = k3dXCols(BC, S, DW);
    static constexpr int S_ELEMS = SCOLS * BC;
    static constexpr int D_ELEMS = 32 * BK;
    static constexpr int S_ROUNDS = S_ELEMS / ROUND, D_ROUNDS = D_ELEMS / ROUND;
    static constexpr int NW = 8 / WQ; // waves per item
    static constexpr int S_LOADS = (S_ROUNDS + NW - 1) / NW;
    static constexpr int D_LOADS = (D_ROUNDS + NW - 1) / NW;
    static constexpr int LOADS = S_LOADS + D_LOADS;
    static constexpr int D_BASE = WQ * S_ELEMS;
    static constexpr int DRAIN = WQ * (S_ELEMS + D_ELEMS);
    static constexpr int SLOT = DRAIN + ROUND;
    static_assert(S_ELEMS % ROUND == 0 && D_ELEMS % ROUND == 0, "K3 direct rounds");
};

#if defined(__gfx950__)
#define K3D_BARRIER()                      \
    do                                     \
    {                                      \
        __builtin_amdgcn_s_barrier();      \
        __builtin_amdgcn_sched_barrier(0); \
    } while(0)
#else
#define K3D_BARRIER() __builtin_trap()
#endif

constexpr int k3dLcm(int a, int b)
{
    int x = a, y = b;
    while(y != 0)
    {
        const int t = x % y;
        x = y;
        y = t;
    }
    return a / x * b;
}

template <int N>
__device__ inline void k3dWaitVm()
{
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}
__device__ inline void k3dWaitLds()
{
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

/// Two transpose reads: lane l gets channel (l % 16) of the fragment's 16 at 8 positions.
template <class Atom, class T>
__device__ inline typename Atom::frag k3dTr(const T* p0, const T* p1)
{
#if defined(__gfx950__)
    typedef short s4 __attribute__((ext_vector_type(4)));
    typedef __attribute__((address_space(3))) s4 lds_s4;
    const s4 lo = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p0));
    const s4 hi = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p1));
    return __builtin_bit_cast(typename Atom::frag,
                              __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
#else
    __builtin_trap();
    return typename Atom::frag{};
#endif
}

template <class T,
          class Atom,
          int BC,
          int BK,
          int WQ,
          int FC,
          int FK,
          int R,
          int S,
          int DW,
          int P,
          int PP,
          int SWZ,
          int XCD>
__device__ void k3WrwDirect(const K3Args& a)
{
    constexpr int WAVES = 8;
    constexpr int WC = BC / (16 * FC), WK = BK / (16 * FK);
    static_assert(WC * WK * WQ == WAVES, "K3 direct: 8 waves");
    static_assert(sizeof(T) == 2 && Atom::KA == 32, "K3 direct: 16-bit, 16x16x32");
    constexpr int NSLOT = P + 1;
    typedef K3DGeom<BC, BK, WQ, S, DW> G;
    typedef K3DSwz<BC, (S - 1) * DW + 1, SWZ> SX;
    typedef K3DSwz<BK, 1, SWZ> DX;
    constexpr int KEEP = (P - 1) * G::LOADS;
    static_assert(G::LOADS == k3dLoads(BC, BK, WQ, S, DW) && KEEP <= 63, "K3 direct loads");
    static_assert(k3LdsBytes(5, 2, 2, BK, BC, 32, R, S, 1, DW, WK, WC, 0, 0, 8, 0, NSLOT, WQ)
                      == static_cast<long long>(NSLOT) * G::SLOT * 2
                             + (WQ > 1 ? (WQ - 1) * (WAVES / WQ) * S * FC * FK * 64 * 16 : 0),
                  "K3 LDS arithmetic (ConvWrwGeom.h)");
    static_assert(NSLOT >= 2 && NSLOT <= 5 && BC <= 512 && BK <= 512, "K3 direct ring");

    __shared__ __attribute__((aligned(16))) T slot0[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot1[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot2[NSLOT > 2 ? G::SLOT : 1];
    __shared__ __attribute__((aligned(16))) T slot3[NSLOT > 3 ? G::SLOT : 1];
    __shared__ __attribute__((aligned(16))) T slot4[NSLOT > 4 ? G::SLOT : 1];
    auto slotPtr = [&](auto sc) -> T* {
        constexpr int I = decltype(sc)::value % NSLOT;
        if constexpr(I == 0)
            return slot0;
        else if constexpr(I == 1)
            return slot1;
        else if constexpr(I == 2)
            return slot2;
        else if constexpr(I == 3)
            return slot3;
        else
            return slot4;
    };

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int item = wave % WQ;
    const int rest = wave / WQ; // load wave within the item
    const int wc = rest / WK, wk = rest % WK;
    const int cBase = wc * FC * 16, kBase = wk * FK * 16;
    const int wg = PP ? wave / 4 : 0;

    // Channels: C and K over all groups (the row strides), cg and kg within one.
    const int C = a.c, K = a.k;
    const int cg = C / a.groups, kg = K / a.groups;
    const int H = a.hi, W = a.wi, Ho = a.ho, Wo = a.wo;
    // Tile: (group, filter depth tap, c tile, k tile), k fastest.
    const int cTiles = (cg + BC - 1) / BC, kTiles = (kg + BK - 1) / BK;
    const int tiles = a.groups * a.ft * cTiles * kTiles;
    const int colBlocks = a.chunks;
    const int items = static_cast<int>(a.steps);
    const int segs = a.segs, segRows = a.segRows;
    const int iters = segRows + R - 1;
    const int sRowBytes = W * C * 2, dRowBytes = Wo * K * 2;

    // Work, two walks over (channel tile, item) units, `per` items to a piece:
    // - cells (a.walk 0): cell = split x tiles + tile, the tile fastest, so the workgroups that
    //   run at once on an XCD share rows in its L2; each tile has a.splits pieces; workgroups
    //   stride over the cells.
    // - stream (a.walk 1): units tile-major, workgroup w takes units [w per, w per + per); a
    //   tile's pieces are the workgroups its units reach, and its last piece zeroes the partials
    //   slots past its pieces (a.splits is the most pieces any tile has). For many tiles.
    // Logical workgroups are XCD-contiguous.
    const int nwg = gridDim.x;
    int w = blockIdx.x;
    if(XCD && nwg % 8 == 0)
        w = (blockIdx.x % 8) * (nwg / 8) + blockIdx.x / 8;
    const long long per = a.stepsPerSplit;
    const long long units = static_cast<long long>(tiles) * items;
    const bool stream = a.walk != 0;
    long long u = stream ? w * per : w;
    const long long u1
        = stream ? (u + per < units ? u + per : units) : static_cast<long long>(tiles) * a.splits;

    // Per-lane LDS offsets of the transpose reads (row-independent).
    const int xl = (lane / 16) * 8 + (lane % 16) / 4;
    const int c4l = lane % 4;
    int sOff[S][FC][2], dOff[FK][2];
#pragma unroll
    for(int s = 0; s < S; ++s)
#pragma unroll
        for(int f = 0; f < FC; ++f)
#pragma unroll
            for(int rd = 0; rd < 2; ++rd)
                sOff[s][f][rd]
                    = item * G::S_ELEMS + SX::at(xl + rd * 4 + s * DW, (cBase + f * 16) / 4 + c4l);
#pragma unroll
    for(int f = 0; f < FK; ++f)
#pragma unroll
        for(int rd = 0; rd < 2; ++rd)
            dOff[f][rd]
                = G::D_BASE + item * G::D_ELEMS + DX::at(xl + rd * 4, (kBase + f * 16) / 4 + c4l);

    // DMA rounds this wave issues: destination (wave-uniform) and, per lane, the source column
    // and channel of its 16 bytes (the swizzle's inverse), decoded where an item starts.
    auto sRound = [&](int i) { return rest * G::S_LOADS + i; };
    auto dRound = [&](int i) { return rest * G::D_LOADS + i; };
    auto sDst = [&](int i) {
        return sRound(i) < G::S_ROUNDS ? item * G::S_ELEMS + sRound(i) * G::ROUND : G::DRAIN;
    };
    auto dDst = [&](int i) {
        return dRound(i) < G::D_ROUNDS ? G::D_BASE + item * G::D_ELEMS + dRound(i) * G::ROUND
                                       : G::DRAIN;
    };

    typename Atom::acc acc[R][S][FC][FK];
    typename Atom::frag ring[R][FK];

    while(u < u1)
    {
        int tile, ib, ie, split, lastSplit;
        if(stream)
        {
            tile = static_cast<int>(u / items);
            ib = static_cast<int>(u - static_cast<long long>(tile) * items);
            ie = static_cast<long long>(items - ib) < u1 - u ? items
                                                             : ib + static_cast<int>(u1 - u);
            u += ie - ib;
            const long long tileFirst = static_cast<long long>(tile) * items;
            const int firstW = static_cast<int>(tileFirst / per);
            split = w - firstW;
            lastSplit = static_cast<int>((tileFirst + items - 1) / per) - firstW;
        }
        else
        {
            tile = static_cast<int>(u % tiles);
            split = static_cast<int>(u / tiles);
            lastSplit = a.splits - 1;
            ib = static_cast<int>(split * per);
            ie = ib + per < items ? ib + static_cast<int>(per) : items;
            u += nwg;
        }
        const int k0 = (tile % kTiles) * BK;
        const int c0 = (tile / kTiles % cTiles) * BC;
        const int t = tile / (kTiles * cTiles) % a.ft;
        const int g = tile / (kTiles * cTiles * a.ft);
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < FC; ++x)
#pragma unroll
                    for(int y = 0; y < FK; ++y)
                        acc[r][s][x][y] = typename Atom::acc{};

        for(int it0 = ib; it0 < ie; it0 += WQ)
        {
            const int my = it0 + item;
            const bool valid = my < ie;
            // Item: (image, row segment, 32-column block), the block fastest.
            const int cb = valid ? my % colBlocks : 0;
            const int ns = valid ? my / colBlocks : 0;
            // Image: (n, output depth od); the x plane is od sd + t dd - pd.
            const int nod = ns / segs;
            const int n = nod / a.dout, od = nod % a.dout;
            const int id = od * a.sd + t * a.dd - a.pd;
            const bool live = valid && id >= 0 && id < a.di;
            const long long xPlane = static_cast<long long>(H) * W * C;
            const long long yPlane = static_cast<long long>(Ho) * Wo * K;
            const int h0 = (ns % segs) * segRows;
            const int h1 = h0 + segRows < H ? h0 + segRows : H;
            const int q0 = cb * 32;
            const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc(
                const_cast<T*>(static_cast<const T*>(a.x)) + n * a.xImage + (live ? id : 0) * xPlane
                    + g * cg,
                static_cast<short>(0),
                live ? static_cast<int>(xPlane * 2 - g * cg * 2) : 0,
                K1_RSRC_FLAGS);
            const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
                const_cast<T*>(static_cast<const T*>(a.dy)) + n * a.yImage + od * yPlane + g * kg,
                static_cast<short>(0),
                live ? static_cast<int>(yPlane * 2 - g * kg * 2) : 0,
                K1_RSRC_FLAGS);
            // Column parts of the source offsets (out-of-range when the column or the channel
            // is outside the image).
            int sV[G::S_LOADS], dV[G::D_LOADS];
#pragma unroll
            for(int i = 0; i < G::S_LOADS; ++i)
            {
                const int u2 = (sRound(i) * G::ROUND + lane * 8) / 4;
                const int x = u2 / SX::C4;
                const int chan = 4 * ((u2 % SX::C4) ^ (4 * SX::g(x)));
                const int col = q0 - a.pw + x;
                const bool ok = sRound(i) < G::S_ROUNDS && x < G::WIN && col >= 0 && col < W
                                && c0 + chan < cg;
                sV[i] = ok ? (col * C + c0 + chan) * 2 : static_cast<int>(K1_OUT_OF_RANGE);
            }
#pragma unroll
            for(int i = 0; i < G::D_LOADS; ++i)
            {
                const int u2 = (dRound(i) * G::ROUND + lane * 8) / 4;
                const int x = u2 / DX::C4;
                const int chan = 4 * ((u2 % DX::C4) ^ (4 * DX::g(x)));
                const int col = q0 + x;
                const bool ok = dRound(i) < G::D_ROUNDS && col < Wo && k0 + chan < kg;
                dV[i] = ok ? (col * K + k0 + chan) * 2 : static_cast<int>(K1_OUT_OF_RANGE);
            }
            // Iteration i: x row i - (R - 1), dy row i - (R - 1) + ph.
            auto issue = [&](auto sc, int i) {
                T* base = slotPtr(sc);
                // x rows outside the segment read zero (the ring's first R - 1 dy rows and
                // the padding iterations multiply nothing).
                const int h = h0 + i - (R - 1);
                const int oh = h + a.ph;
                const bool sOk = h >= h0 && h < h1;
                const bool dOk = oh >= 0 && oh < Ho;
                const int sS = sOk ? h * sRowBytes : 0;
                const int dS = dOk ? oh * dRowBytes : 0;
#pragma unroll
                for(int j = 0; j < G::D_LOADS; ++j)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        yr,
                        (__attribute__((address_space(3))) void*)(base + dDst(j)),
                        16,
                        dOk ? dV[j] : static_cast<int>(K1_OUT_OF_RANGE),
                        dS,
                        0,
                        0);
#pragma unroll
                for(int j = 0; j < G::S_LOADS; ++j)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(
                        xr,
                        (__attribute__((address_space(3))) void*)(base + sDst(j)),
                        16,
                        sOk ? sV[j] : static_cast<int>(K1_OUT_OF_RANGE),
                        sS,
                        0,
                        0);
            };
            // Prologue: rows 0 .. P - 1 into slots 0 .. P - 1.
            k3ForEach<P>([&](auto sc) { issue(sc, decltype(sc)::value); });
#pragma unroll
            for(int r = 0; r < R; ++r)
#pragma unroll
                for(int f = 0; f < FK; ++f)
                    ring[r][f] = typename Atom::frag{};
            // The stagger: pong takes one barrier more, a half step behind ping from here on. Ping
            // reads row 0 after it, so pong's share of row 0 lands first.
            if(PP && wg != 0)
            {
                k3dWaitVm<KEEP>();
                K3D_BARRIER();
            }

            constexpr int UNROLL = k3dLcm(NSLOT, R);
            auto step = [&](auto uc, int base) {
                constexpr int U = decltype(uc)::value;
                constexpr int RS = U % NSLOT;
                constexpr int IS = (U + P) % NSLOT;
                const int i = base + U;
                if(!PP || wg == 0)
                {
                    k3dWaitVm<KEEP>();
                    k3dWaitLds();
                }
                K3D_BARRIER();
                if(PP)
                    __builtin_amdgcn_s_setprio(0);
                const T* rs = slotPtr(K3Int<RS>{});
                typename Atom::frag sf[S][FC];
#pragma unroll
                for(int s = 0; s < S; ++s)
#pragma unroll
                    for(int f = 0; f < FC; ++f)
                        sf[s][f] = k3dTr<Atom, T>(rs + sOff[s][f][0], rs + sOff[s][f][1]);
                constexpr int WR = (U + R - 1) % R;
#pragma unroll
                for(int f = 0; f < FK; ++f)
                    ring[WR][f] = k3dTr<Atom, T>(rs + dOff[f][0], rs + dOff[f][1]);
                issue(K3Int<IS>{}, i + P);
                if(PP && wg != 0)
                {
                    k3dWaitVm<KEEP>();
                    k3dWaitLds();
                }
                if(PP)
                {
                    K3D_BARRIER();
                    __builtin_amdgcn_s_setprio(1);
                }
                // The first R - 1 iterations multiply zero x rows (above the image), and so do
                // the padding iterations past the last row: no branch in the loop.
                {
#pragma unroll
                    for(int j = 0; j < R; ++j)
                    {
                        const int r = R - 1 - j;
                        const int sl = (U + j) % R;
#pragma unroll
                        for(int s = 0; s < S; ++s)
#pragma unroll
                            for(int x = 0; x < FC; ++x)
#pragma unroll
                                for(int y = 0; y < FK; ++y)
                                    acc[r][s][x][y]
                                        = Atom::mma(sf[s][x], ring[sl][y], acc[r][s][x][y]);
                    }
                }
            };
            for(int base = 0; base < iters; base += UNROLL)
                k3ForEach<UNROLL>([&](auto uc) { step(uc, base); });
            // Drain: every DMA landed and every read retired before the next item's prologue.
            k3dWaitVm<0>();
            k3dWaitLds();
            if(PP && wg == 0)
                K3D_BARRIER();
            K3D_BARRIER();
            if(PP)
                __builtin_amdgcn_s_setprio(0);
        }

        // waves_q: the item waves of one channel slice add into item 0's, in item order.
        if constexpr(WQ > 1)
        {
            constexpr int NW = WAVES / WQ;
            constexpr int PER = S * FC * FK;
            typedef typename Atom::acc A4;
            __shared__ A4 red[(WQ - 1) * NW * PER * 64];
#pragma unroll
            for(int r = 0; r < R; ++r)
            {
                if(item != 0)
                {
                    A4* o = red + (((item - 1) * NW + rest) * PER) * 64 + lane;
#pragma unroll
                    for(int s = 0; s < S; ++s)
#pragma unroll
                        for(int x = 0; x < FC; ++x)
#pragma unroll
                            for(int y = 0; y < FK; ++y)
                                o[((s * FC + x) * FK + y) * 64] = acc[r][s][x][y];
                }
                __syncthreads();
                if(item == 0)
                {
                    for(int q = 1; q < WQ; ++q)
                    {
                        const A4* o = red + (((q - 1) * NW + rest) * PER) * 64 + lane;
#pragma unroll
                        for(int s = 0; s < S; ++s)
#pragma unroll
                            for(int x = 0; x < FC; ++x)
#pragma unroll
                                for(int y = 0; y < FK; ++y)
                                    acc[r][s][x][y] += o[((s * FC + x) * FK + y) * 64];
                    }
                }
                __syncthreads();
            }
        }
        // Partials: lane holds c = cBase + x 16 + (lane / 16) 4 + e, k = kBase + y 16 + lane % 16.
        // [split][g][tap][k][c] within the group; tap (t R + r) S + s.
        const long long kc = static_cast<long long>(kg) * cg;
        float* out = a.partial + static_cast<long long>(split) * a.partialElems
                     + (static_cast<long long>(g) * a.ft + t) * R * S * kc;
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < FC; ++x)
#pragma unroll
                    for(int y = 0; y < FK; ++y)
                    {
                        const int c = c0 + cBase + x * 16 + (lane / 16) * 4;
                        const int k = k0 + kBase + y * 16 + lane % 16;
                        if(item == 0 && c < cg && k < kg)
                        {
                            float* o = out + (r * S + s) * kc + static_cast<long long>(k) * cg + c;
                            *reinterpret_cast<typename Atom::acc*>(o) = acc[r][s][x][y];
                            for(int z = split + 1; split == lastSplit && z < a.splits; ++z)
                                *reinterpret_cast<typename Atom::acc*>(
                                    o + static_cast<long long>(z - split) * a.partialElems)
                                    = typename Atom::acc{};
                        }
                    }
    }
}

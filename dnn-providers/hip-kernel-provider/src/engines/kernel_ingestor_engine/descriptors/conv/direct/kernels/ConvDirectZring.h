// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K2 z-ring tiles (CONV_DIRECT_FWD and CONV_DIRECT_BWD, ring 2): stride-1 3-D direct convolution
// on 16 x 16 MFMA atoms (wave64), channels-last, with the depth ring in the accumulators. Backward
// data runs as the forward form (dy as the input, the rotated filter with its channel roles
// swapped, padding F - 1 - p). HIPRTC-compatible: no standard headers.
//
// Work. Output planes are numbered in the linear order (tile, z), a tile being TY x 16 output
// positions (y, x) x BN output channels of one image, the channel tile fastest. The launch has
// one workgroup per compute unit (gridDim.x, chosen at launch, so one code object serves every
// product); workgroup b takes the b-th of gridDim.x contiguous ranges of that order, their
// lengths within one plane of each other, as one pencil per tile it touches: L consecutive
// output planes from zs. A pencil walks the input planes zs - pd + t, t = 0 .. L + KD - 2
// ("steps"). Input plane zs - pd + t feeds output plane zs + t - dz through depth tap dz.
//
// The ring: KD accumulator slots, slot dz holding output plane zs + t - dz at step t. At the end
// of step t slot KD - 1 is complete and stored; the slots shift by one and slot 0 restarts. Each
// input plane is staged in LDS once per pencil (in BC-channel chunks, "stages") and applied to
// all KD live output planes, so an input element is fetched about (TY + R - 1)(16 + S - 1) /
// (16 TY) times instead of once per depth tap.
//
// A stage is the tile plus its halo, (TY + R - 1) x (16 + S - 1) positions x BC channels, in LDS
// chunk-major: [16-byte channel chunk][position], each chunk's sub-plane padded to the target's
// conflict-free period for 128-bit reads (128 bytes on gfx942, 256 on gfx950), so a 16-lane
// group's fragment read covers 16 consecutive 16-byte slots of one chunk. Double buffered: the
// next stage's loads go out during this stage's MMAs and land in the other buffer after them,
// one barrier per stage.
//
// Waves: WM along the tile's rows, WN along the output channels. A stage is a sequence of items
// (dy, dx, channel block, dz): a wave reads its A fragments of (dy, dx, block) from LDS once
// (16 bytes a lane: 16 / (KPL x element) atoms) and uses them for all KD depth taps; the weight
// fragments come straight from global memory in the formatted order (k2FormatWeights, ka = 64
// bytes of channels: each load reads whole 128-byte lines), K2Z_PREFETCH items ahead of their
// MMAs, carried across stages. The MMA takes the weights as its first operand, so a result lane
// holds 4 consecutive output channels of one position and stores them as one vector.
//
// Prologue steps (the first KD - 1 of a pencil), steady steps and epilogue steps are three loops:
// the edge steps guard each depth tap's MMAs with a uniform branch (and mask its weight loads),
// the steady ones run every tap. The order of every sum is fixed (no atomics): deterministic.
//
// Edges (K2Z_EDGES 1, set by the pack when an output plane is not whole tiles). The rem columns
// past the last whole 16-column strip are served by narrow tiles: a fragment's 16 lanes hold
// RF = 16 / rem rows x rem columns of that strip (lane l: row l / rem, column l % rem; lanes past
// RF rem idle), so a tile is ROWS RF rows, staged as those rows plus R - 1, rem + S - 1 positions
// each, and computed by wave row 0 (the other wave rows skip their MMAs and stores). In the last
// tile row of a strip, a wave row whose rows all lie past the output skips its MMAs too.
#pragma once

#include "ConvDirectCore.h"

// Per target: the LDS sub-plane period of conflict-free 128-bit reads (bytes) and the weight
// prefetch depth (items ahead of their MMAs) the register budget holds without spills.
#if defined(__gfx950__)
#define K2Z_PERIOD_BYTES 256
#ifndef K2Z_PREFETCH
#define K2Z_PREFETCH 8
#endif
#else
#define K2Z_PERIOD_BYTES 128
#ifndef K2Z_PREFETCH
#define K2Z_PREFETCH 5
#endif
#endif

/// The machine scheduler moves nothing across this point (it orders the item's loads around its
/// MMAs: vector loads complete in order, so where a prefetch sits sets what a wait covers).
__device__ inline void k2zFence()
{
    __builtin_amdgcn_sched_barrier(0);
}

/// Reads @p v in an empty side-effecting asm: instruction selection otherwise sinks unchained
/// MMAs past later fences toward their uses. No instruction, no copy.
template <class V>
__device__ inline void k2zPin(const V& v)
{
    asm volatile("" ::"v"(v));
}

/// An opaque copy of a uniform value: recomputed where it is used instead of being held in
/// registers across the loops.
__device__ inline unsigned k2zOpaque(unsigned v)
{
    asm volatile("" : "+s"(v));
    return v;
}

/// A uniform value the backend computed on the vector unit (integer division expands to VALU
/// code), back in a scalar register: a buffer offset in a vector register costs a waterfall loop
/// around every load.
__device__ inline unsigned k2zUniform(unsigned v)
{
    return __builtin_amdgcn_readfirstlane(v);
}

/// The output element of an fp32 result: bf16 through the compiler's conversion (round to
/// nearest even, infinities kept, NaN quiet), half the instructions of the bit-level form.
template <class TO>
__device__ inline TO k2zStore(float v)
{
    if constexpr(sizeof(TO) == 2 && !__is_same(TO, _Float16))
        return __builtin_bit_cast(unsigned short, static_cast<__bf16>(v));
    else
        return K1Elem<TO>::store(v);
}

#ifndef K2Z_EDGES
#define K2Z_EDGES 0
#endif

/// The most stage positions of a narrow tile of `frags` fragments over 1 to 15 columns.
constexpr int k2zNarrowSlots(int frags, int r, int s)
{
    int best = 0;
    for(int w = 1; w < 16; ++w)
    {
        const int slots = (frags * (16 / w) + r - 1) * (w + s - 1);
        best = slots > best ? slots : best;
    }
    return best;
}

template <class T,
          class Atom,
          int KD,
          int R,
          int S,
          int BM,
          int BN,
          int BC,
          int WM,
          int WN,
          class TO = T>
__device__ __attribute__((always_inline)) void k2ConvZring(const K2Args& a CONV_EPI_FN_PARAM)
{
    static_assert(Atom::WAVE == 64 && Atom::DIM == 16 && Atom::ACC == 4, "16 x 16 MFMA atoms");
    constexpr int TX = 16; // one M block per tile row
    constexpr int TY = BM / TX;
    constexpr int ROWS = TY / WM; // tile rows (M blocks) per wave
    constexpr int NBW = BN / 16 / WN; // 16-channel N blocks per wave
    constexpr int THREADS = WM * WN * 64;
    constexpr int ES = static_cast<int>(sizeof(T));
    constexpr int VEC = 16 / ES; // channels per 16-byte chunk
    constexpr int CB = 4 * VEC; // channels per block: four lane groups' chunks
    constexpr int JB = BC / CB; // blocks per stage
    constexpr int NSL = VEC / Atom::KPL; // atoms per 16-byte fragment
    constexpr int PY = TY + R - 1, PX = TX + S - 1, PP = PY * PX;
    constexpr bool EDGES = K2Z_EDGES != 0;
    constexpr int PPN = EDGES ? k2zNarrowSlots(ROWS, R, S) : 0; // a narrow tile's stage, at most
    constexpr int PPM = PPN > PP ? PPN : PP;
    static_assert(!EDGES || ROWS % WM == 0, "narrow tiles deal their fragments to the wave rows");
    constexpr int PER = K2Z_PERIOD_BYTES / 16;
    constexpr int PPA = (PPM + PER - 1) / PER * PER; // positions per chunk sub-plane
    constexpr int NQ = BC / VEC; // chunks per stage
    constexpr int SUB = PPA * VEC; // elements per chunk sub-plane
    constexpr int PLANE = NQ * SUB; // elements per stage
    constexpr int BUFS = PLANE + VEC; // a stage and the dummy slot of the copy lanes past it
    constexpr int TC = PPA * NQ; // 16-byte slots per stage
    constexpr int RC = (TC + THREADS - 1) / THREADS;
    constexpr int NG = R * S * JB; // groups (dy, dx, block)
    constexpr int NI = NG * KD; // items (group, dz)
    constexpr int PF = K2Z_PREFETCH < NI ? K2Z_PREFETCH : NI;
    static_assert(BM % TX == 0 && TY % WM == 0 && BN % (16 * WN) == 0 && BC % CB == 0
                      && VEC % Atom::KPL == 0 && PER % 8 == 0,
                  "z-ring tile");
    typedef typename K1Raw<16>::type Raw;
    typedef typename Atom::frag Frag;
    typedef typename Atom::acc Acc;
    struct Frags
    {
        Frag f[NSL];
    };
    static_assert(sizeof(Frags) == 16, "16-byte fragments");

    __shared__ __attribute__((aligned(16))) T lds[2 * BUFS];

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = wave % WM, wn = wave / WM;
    const int lm = lane % 16, lg = lane / 16;

    // Work ranges over the linear order (tile, z).
    const unsigned ntk = static_cast<unsigned>(a.kpad / BN);
    const unsigned tilesX = static_cast<unsigned>(a.tilesW);
    const unsigned tilesY = static_cast<unsigned>(a.tilesH);
    const unsigned dOut = static_cast<unsigned>(a.dout);
    // Edges: tilesX whole strips (a.tilesW) x tilesY tile rows, then a.tilesD narrow tiles, per
    // image and channel tile.
    const unsigned tilesPer = tilesX * tilesY + (EDGES ? static_cast<unsigned>(a.tilesD) : 0u);
    const unsigned total = EDGES ? ntk * tilesPer * static_cast<unsigned>(a.n) * dOut
                                 : ntk * tilesX * tilesY * static_cast<unsigned>(a.n) * dOut;
    const unsigned groups = gridDim.x, bid = blockIdx.x;
    const unsigned share = k2zUniform(total / groups);
    const unsigned extra = total - share * groups;
    const unsigned u0 = bid * share + (bid < extra ? bid : extra);
    const unsigned u1 = u0 + share + (bid < extra ? 1u : 0u);
    if(u1 <= u0)
        return;

    const int hwc = a.hi * a.wi * a.c;
    // Formatted weights wf[((tap * ca + c / CB) * kpad + n) * CB + c % CB]: bytes per CB-channel
    // block of every column (KB) and per tap (TAPB).
    const unsigned KB = static_cast<unsigned>(a.kpad) * CB * ES;
    const unsigned TAPB = static_cast<unsigned>(a.ca) * KB;
    const __amdgpu_buffer_rsrc_t wRsrc
        = k1Rsrc(a.w, static_cast<long long>(KD) * R * S * static_cast<long long>(TAPB));
    // B: output channel 16 (wn NBW + j) + lm of the tile, channels VEC lg .. + VEC - 1 of a block.
    unsigned bLane[NBW];
#pragma unroll
    for(int j = 0; j < NBW; ++j)
        bLane[j] = static_cast<unsigned>(((16 * (wn * NBW + j) + lm) * CB + VEC * lg) * ES);
    // A: chunk lg of a block, position lm of the wave's first row (elements).
    const int aLane = lg * SUB + (wm * ROWS * PX + lm) * VEC;

    Acc acc[KD][ROWS][NBW];
    Raw carry[PF][NBW];

    auto loadA = [&](const T* ab, auto G, Raw(&af)[ROWS]) {
        constexpr int g = decltype(G)::value;
        constexpr int jb = g % JB, dx = (g / JB) % S, dy = g / JB / S;
#pragma unroll
        for(int i = 0; i < ROWS; ++i)
            af[i] = *reinterpret_cast<const Raw*>(ab + 4 * jb * SUB + ((i + dy) * PX + dx) * VEC);
    };
    // Weight fragments of item k at stage base sb (bytes); masked off when !ok.
    auto loadB = [&](unsigned sb, auto KI, bool ok, Raw(&bf)[NBW]) {
        constexpr int k = decltype(KI)::value;
        constexpr int g = k / KD, dz = k % KD;
        constexpr int jb = g % JB, dx = (g / JB) % S, dy = g / JB / S;
        constexpr int tap = (dz * R + dy) * S + dx;
        const int so = static_cast<int>(sb + tap * TAPB + jb * KB);
#pragma unroll
        for(int j = 0; j < NBW; ++j)
            bf[j] = k2BufferLoadS<Raw>(wRsrc, ok ? bLane[j] : K1_OUT_OF_RANGE, so);
    };
    auto mmas = [&](auto DZ, const Raw(&af)[ROWS], const Raw(&bf)[NBW]) {
        constexpr int dz = decltype(DZ)::value;
        Frags fb[NBW];
#pragma unroll
        for(int j = 0; j < NBW; ++j)
            fb[j] = __builtin_bit_cast(Frags, bf[j]);
#pragma unroll
        for(int s = 0; s < NSL; ++s)
#pragma unroll
            for(int i = 0; i < ROWS; ++i)
            {
                const Frags fa = __builtin_bit_cast(Frags, af[i]);
#pragma unroll
                for(int j = 0; j < NBW; ++j)
                    acc[dz][i][j] = Atom::mma(fb[j].f[s], fa.f[s], acc[dz][i][j]);
            }
#pragma unroll
        for(int i = 0; i < ROWS; ++i)
#pragma unroll
            for(int j = 0; j < NBW; ++j)
                k2zPin(acc[dz][i][j]);
    };
#if K2Z_EDGES
    // Edge tiles' steps: only the wave's first `live` fragments (scalar branches; the rest of the
    // tile lies past the output).
    auto mmasLive = [&](auto DZ, const Raw(&af)[ROWS], const Raw(&bf)[NBW], int live) {
        constexpr int dz = decltype(DZ)::value;
        Frags fb[NBW];
#pragma unroll
        for(int j = 0; j < NBW; ++j)
            fb[j] = __builtin_bit_cast(Frags, bf[j]);
#pragma unroll
        for(int s = 0; s < NSL; ++s)
#pragma unroll
            for(int i = 0; i < ROWS; ++i)
            {
                if(i >= live)
                    break;
                const Frags fa = __builtin_bit_cast(Frags, af[i]);
#pragma unroll
                for(int j = 0; j < NBW; ++j)
                    acc[dz][i][j] = Atom::mma(fb[j].f[s], fa.f[s], acc[dz][i][j]);
            }
#pragma unroll
        for(int i = 0; i < ROWS; ++i)
#pragma unroll
            for(int j = 0; j < NBW; ++j)
                k2zPin(acc[dz][i][j]);
    };
#endif

    const unsigned firstTile = k2zUniform(u0 / dOut), lastTile = k2zUniform((u1 - 1) / dOut);
    // The first tile's (channel tile, x tile, y tile, image); the next pencils' tiles follow it
    // in that order, channel tile fastest.
    unsigned kt, xt, yt, img;
    if constexpr(!EDGES)
    {
        const unsigned r1 = k2zUniform(firstTile / ntk);
        kt = firstTile - r1 * ntk;
        const unsigned r2 = k2zUniform(r1 / tilesX);
        xt = r1 - r2 * tilesX;
        img = k2zUniform(r2 / tilesY);
        yt = r2 - img * tilesY;
    }
    for(unsigned tile = firstTile; tile <= lastTile; ++tile)
    {
        const unsigned tb = tile * dOut;
        const int zs = static_cast<int>(k2zUniform((u0 > tb ? u0 : tb) - tb));
        const int L = static_cast<int>(k2zUniform((u1 < tb + dOut ? u1 : tb + dOut) - tb)) - zs;
        // Edges: the tile's (channel tile, place in the image, image), decoded per tile; a narrow
        // tile's index nq among the narrow tiles.
        unsigned nq = 0;
        bool narrow = false;
        if constexpr(EDGES)
        {
            const unsigned r1 = k2zUniform(tile / ntk);
            kt = tile - r1 * ntk;
            img = k2zUniform(r1 / tilesPer);
            const unsigned loc = r1 - img * tilesPer;
            narrow = loc >= tilesX * tilesY;
            nq = narrow ? loc - tilesX * tilesY : 0u;
            yt = narrow ? 0u : k2zUniform(loc / tilesX);
            xt = narrow ? tilesX : loc - yt * tilesX;
        }
#if K2Z_EDGES
        auto pencil = [&](auto NW) {
            constexpr bool NARROW = decltype(NW)::value != 0;
#else
        {
            constexpr bool NARROW = false;
#endif
            const int x0 = static_cast<int>(xt) * TX, y0 = static_cast<int>(yt) * TY;
            const int k0 = static_cast<int>(kt) * BN;
            const unsigned kofs = static_cast<unsigned>(k0 * CB * ES);
            const __amdgpu_buffer_rsrc_t xRsrc
                = k1Rsrc(static_cast<const T*>(a.x) + static_cast<long long>(img) * a.di * hwc,
                         static_cast<long long>(a.di) * hwc * ES);
            const long long yImage = static_cast<long long>(a.dout) * a.ho * a.wo * a.k;
            TO* const yImg = static_cast<TO*>(a.y) + static_cast<long long>(img) * yImage;
            const __amdgpu_buffer_rsrc_t yRsrc
                = k1Rsrc(yImg, yImage * static_cast<long long>(sizeof(TO)));

            // Narrow tile nq: rows yA .. yA + ROWS rf - 1 of the last strip (rem columns), staged from
            // row yA, pxr positions a row; lane (lr, lc) of every fragment. Its ROWS fragments are
            // dealt to the wave rows, ROWS / WM each: wave row wm's fragment i is the tile's
            // fragment wm ROWS / WM + i, rows (wm ROWS / WM + i) rf on. `live`: the fragments a
            // wave computes (all of them in a whole tile; in the last tile row those with rows in
            // the output; ROWS / WM in a narrow tile).
            const int rem = a.wo - static_cast<int>(tilesX) * TX;
            const int rf = NARROW ? 16 / rem : 1;
            const int lr = NARROW ? lm / rem : 0, lc = NARROW ? lm - lr * rem : 0;
            const int yA = NARROW ? static_cast<int>(nq) * ROWS * rf : y0;
            const int pxr = NARROW ? rem + S - 1 : PX;
            const int ppr = NARROW ? (ROWS * rf + R - 1) * pxr : PP;
            const int rowsLeft = a.ho - y0 - wm * ROWS;
            const int live = !EDGES   ? ROWS
                             : NARROW ? ROWS / WM
                                      : (rowsLeft < 0 ? 0 : (rowsLeft < ROWS ? rowsLeft : ROWS));
            const bool waveLive = live > 0;
            // A narrow tile's lane slot (row wm ROWS / WM rf + lr, column lc of the stage): the
            // wave's and lane's part of every A read.
            const int laneSlot = lg * SUB + ((wm * (ROWS / WM) * rf + lr) * pxr + lc) * VEC;
            // Stage copy: 16-byte slot i = tid + r THREADS is octet o = i / 8 -> chunk o % NQ,
            // positions 8 (o / NQ) .. + 7, one per lane of the octet.
            int tidp = tid;
            asm volatile("" : "+v"(tidp));
            unsigned cpOff[RC];
            int cpLds[RC];
#pragma unroll
            for(int r = 0; r < RC; ++r)
            {
                const int i = tidp + r * THREADS;
                const int o = i / 8, e = i % 8;
                const int qq = o % NQ, p = (o / NQ) * 8 + e;
                const int py = p / pxr, px = p - py * pxr;
                const int iy = yA - a.ph + py, ix = x0 - a.pw + px;
                const bool in = i < TC && p < ppr
                                && static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi)
                                && static_cast<unsigned>(ix) < static_cast<unsigned>(a.wi);
                cpOff[r] = in ? static_cast<unsigned>(((iy * a.wi + ix) * a.c + VEC * qq) * ES)
                              : K1_OUT_OF_RANGE;
                cpLds[r] = i < TC ? qq * SUB + p * VEC : PLANE;
            }
            auto loadPlane = [&](int zin, int c0, Raw(&pl)[RC]) {
                const bool zok = static_cast<unsigned>(zin) < static_cast<unsigned>(a.di);
                const int so = ((zok ? zin : 0) * hwc + c0) * ES;
#pragma unroll
                for(int r = 0; r < RC; ++r)
                    pl[r] = k2BufferLoadS<Raw>(xRsrc, zok ? cpOff[r] : K1_OUT_OF_RANGE, so);
            };
            auto storePlane = [&](int p, const Raw(&pl)[RC]) {
#pragma unroll
                for(int r = 0; r < RC; ++r)
                    *reinterpret_cast<Raw*>(lds + p * BUFS + cpLds[r]) = pl[r];
            };

            // Output plane zo from slot KD - 1: lane holds channels 16 (wn NBW + j) + 4 lg .. + 3 of
            // column lm of each of the wave's rows (K is a multiple of 4: one vector a lane). The row
            // offset is wave-uniform (scalar offset, a scalar branch past the output's rows); the lane
            // part and its mask are recomputed per store from opaque copies of the sizes instead of
            // being held across the steps.
            auto storeSlot = [&](int zo) {
                const int ho = static_cast<int>(k2zOpaque(static_cast<unsigned>(a.ho)));
                const int wo = static_cast<int>(k2zOpaque(static_cast<unsigned>(a.wo)));
                const int kk = static_cast<int>(k2zOpaque(static_cast<unsigned>(a.k)));
                typedef typename K1Raw<4 * static_cast<int>(sizeof(TO))>::type Out;
                if constexpr(NARROW)
                {
                    // Fragment i: row yA + (wm ROWS / WM + i) rf + lr, column x0 + lc.
                    const bool laneOk = lr < rf;
#pragma unroll
                    for(int i = 0; i < ROWS / WM; ++i)
                    {
                        const int yy = yA + (wm * (ROWS / WM) + i) * rf + lr;
                        const bool posOk = laneOk && yy < ho;
                        const int pixBase = ((zo * ho + yy) * wo + x0 + lc) * kk;
#pragma unroll
                        for(int j = 0; j < NBW; ++j)
                        {
                            const int kc = k0 + 16 * (wn * NBW + j) + 4 * lg;
                            TO v[4];
#pragma unroll
                            for(int e = 0; e < 4; ++e)
                            {
#if defined(CONV_EPI) && CONV_EPI
                                v[e] = k2zStore<TO>(
                                    epi(acc[KD - 1][i][j][e],
                                        epi.chan(kc + e, kc + e < kk),
                                        static_cast<long long>(img) * a.dout * a.ho * a.wo * kk
                                            + pixBase + kc + e,
                                        posOk && kc + e < kk));
#else
                                v[e] = k2zStore<TO>(acc[KD - 1][i][j][e]);
#endif
                            }
                            Out w;
                            __builtin_memcpy(&w, v, sizeof(Out));
                            const unsigned off
                                = posOk && kc < kk
                                      ? static_cast<unsigned>((pixBase + kc)
                                                              * static_cast<int>(sizeof(TO)))
                                      : K1_OUT_OF_RANGE;
                            if constexpr(sizeof(TO) == 2)
                                __builtin_amdgcn_raw_buffer_store_b64(w, yRsrc, off, 0, 0);
                            else
                                __builtin_amdgcn_raw_buffer_store_b128(w, yRsrc, off, 0, 0);
                        }
                    }
                    return;
                }
                const bool colOk = x0 + lm < wo;
                unsigned laneOff[NBW];
#pragma unroll
                for(int j = 0; j < NBW; ++j)
                {
                    const int kc = k0 + 16 * (wn * NBW + j) + 4 * lg;
                    laneOff[j]
                        = colOk && kc < kk
                              ? static_cast<unsigned>((lm * kk + kc) * static_cast<int>(sizeof(TO)))
                              : K1_OUT_OF_RANGE;
                }
#pragma unroll
                for(int i = 0; i < ROWS; ++i)
                {
                    const int yy = y0 + wm * ROWS + i;
                    if(yy >= ho)
                        break;
                    // The row's first output element (x0, k = 0).
                    const int rowBase = ((zo * ho + yy) * wo + x0) * kk;
#pragma unroll
                    for(int j = 0; j < NBW; ++j)
                    {
                        TO v[4];
#pragma unroll
                        for(int e = 0; e < 4; ++e)
                        {
#if defined(CONV_EPI) && CONV_EPI
                            // The channel's operands per element: no registers held across the rows.
                            const int kc = k0 + 16 * (wn * NBW + j) + 4 * lg + e;
                            v[e] = k2zStore<TO>(
                                epi(acc[KD - 1][i][j][e],
                                    epi.chan(kc, kc < kk),
                                    static_cast<long long>(img) * a.dout * a.ho * a.wo * kk
                                        + rowBase + lm * kk + kc,
                                    colOk && kc < kk));
#else
                            v[e] = k2zStore<TO>(acc[KD - 1][i][j][e]);
#endif
                        }
                        Out w;
                        __builtin_memcpy(&w, v, sizeof(Out));
                        const int so = rowBase * static_cast<int>(sizeof(TO));
                        if constexpr(sizeof(TO) == 2)
                            __builtin_amdgcn_raw_buffer_store_b64(w, yRsrc, laneOff[j], so, 0);
                        else
                            __builtin_amdgcn_raw_buffer_store_b128(w, yRsrc, laneOff[j], so, 0);
                    }
                }
            };

            // One stage on buffer p: items in order; the weights PF items ahead, on into the next
            // stage's first items (carry); the A fragments one group ahead; the next stage's input
            // loads after the first item's weight loads (they then have PF items of cover).
            auto stage = [&](auto E,
                             int p,
                             unsigned sb,
                             unsigned sbn,
                             const bool(&valid)[KD],
                             int zinN,
                             int cN,
                             Raw(&pl)[RC]) {
                constexpr bool EDGE = decltype(E)::value != 0;
                const T* ab = lds + p * BUFS + (NARROW ? laneSlot : aLane);
                Raw bq[NI + PF][NBW];
                Raw aq[2][ROWS];
#pragma unroll
                for(int k = 0; k < PF; ++k)
#pragma unroll
                    for(int j = 0; j < NBW; ++j)
                        bq[k][j] = carry[k][j];
                // Narrow tiles: fragment i's lane reads staged row i rf + lr + dy, column lc + dx (the
                // lane's part is in ab, the rest is wave-uniform).
                auto loadAt = [&](auto G, Raw(&af)[ROWS]) {
                    if constexpr(NARROW)
                    {
                        constexpr int g = decltype(G)::value;
                        constexpr int jb = g % JB, dx = (g / JB) % S, dy = g / JB / S;
#pragma unroll
                        for(int i = 0; i < ROWS / WM; ++i)
                            af[i] = *reinterpret_cast<const Raw*>(
                                ab + 4 * jb * SUB + ((i * rf + dy) * pxr + dx) * VEC);
                    }
                    else
                        loadA(ab, G, af);
                };
                loadAt(K2Int<0>{}, aq[0]);
                k2StaticFor<0, NI>([&](auto KI) {
                    constexpr int k = decltype(KI)::value;
                    constexpr int g = k / KD, dz = k % KD;
                    if constexpr(dz == 0 && g + 1 < NG)
                        loadAt(K2Int<g + 1>{}, aq[(g + 1) % 2]);
                    constexpr int kn = k + PF;
                    if constexpr(kn < NI)
                        loadB(sb, K2Int<kn>{}, !EDGE || valid[kn % KD], bq[kn]);
                    else
                        loadB(sbn, K2Int<kn - NI>{}, true, bq[kn]);
                    if constexpr(k == 0)
                        loadPlane(zinN, cN, pl);
                    k2zFence();
#if K2Z_EDGES
                    if(!EDGE || valid[dz])
                    {
                        if constexpr(EDGE)
                            mmasLive(K2Int<dz>{}, aq[g % 2], bq[k], live);
                        else
                            mmas(K2Int<dz>{}, aq[g % 2], bq[k]);
                    }
#else
                    if(!EDGE || valid[dz])
                        mmas(K2Int<dz>{}, aq[g % 2], bq[k]);
#endif
                    k2zFence();
                });
#pragma unroll
                for(int k = 0; k < PF; ++k)
#pragma unroll
                    for(int j = 0; j < NBW; ++j)
                        carry[k][j] = bq[NI + k][j];
            };

            // Pencil prologue: stage (step 0, chunk 0) into buffer 0, its first weight items.
            int p = 0;
            {
                Raw pl[RC];
                loadPlane(zs - a.pd, 0, pl);
                k2StaticFor<0, PF>(
                    [&](auto KI) { loadB(kofs, KI, true, carry[decltype(KI)::value]); });
                storePlane(0, pl);
            }
            __syncthreads();
#pragma unroll
            for(int s = 0; s < KD; ++s)
#pragma unroll
                for(int i = 0; i < ROWS; ++i)
#pragma unroll
                    for(int j = 0; j < NBW; ++j)
                        acc[s][i][j] = Acc{};

            // Steps [t0, t1): every stage, then (ST) the finished plane's store, then the ring shift.
            auto steps = [&](auto E, auto ST, int t0, int t1) {
                constexpr bool EDGE = decltype(E)::value != 0;
                for(int t = t0; t < t1; ++t)
                {
                    bool valid[KD];
                    const int zin = zs - a.pd + t;
#pragma unroll
                    for(int dz = 0; dz < KD; ++dz)
                        valid[dz] = static_cast<unsigned>(zin) < static_cast<unsigned>(a.di)
                                    && t - dz >= 0 && t - dz < L && waveLive;
                    (void)valid;
                    unsigned sb = kofs;
                    for(int c0 = 0; c0 < a.c; c0 += BC)
                    {
                        const bool lastC = c0 + BC >= a.c;
                        const int cN = lastC ? 0 : c0 + BC;
                        const int zinN = lastC ? zin + 1 : zin;
                        const unsigned sbn = lastC ? kofs : sb + JB * KB;
                        Raw pl[RC];
                        stage(E, p, sb, sbn, valid, zinN, cN, pl);
                        storePlane(p ^ 1, pl);
                        __syncthreads();
                        p ^= 1;
                        sb = sbn;
                    }
                    if constexpr(decltype(ST)::value != 0)
                        storeSlot(zs + t - (KD - 1));
#pragma unroll
                    for(int i = 0; i < ROWS; ++i)
#pragma unroll
                        for(int j = 0; j < NBW; ++j)
                        {
#pragma unroll
                            for(int s = KD - 1; s > 0; --s)
                                acc[s][i][j] = acc[s - 1][i][j];
                            acc[0][i][j] = Acc{};
                        }
                    (void)EDGE;
                }
            };
            const int tS = L > KD - 1 ? L : KD - 1;
            steps(K2Int<1>{}, K2Int<0>{}, 0, KD - 1);
            if(live == ROWS)
            {
                steps(K2Int<0>{}, K2Int<1>{}, KD - 1, tS);
                steps(K2Int<1>{}, K2Int<1>{}, tS, L + KD - 1);
            }
            else
            {
                // A wave with part of the tile or none: the same stages and barriers, its live
                // fragments' MMAs only.
                steps(K2Int<1>{}, K2Int<1>{}, KD - 1, L + KD - 1);
            }
#if K2Z_EDGES
        };
        if(narrow)
            pencil(K2Int<1>{});
        else
            pencil(K2Int<0>{});
        __syncthreads();
        continue;
#else
        }
        __syncthreads();
#endif
        if(++kt == ntk)
        {
            kt = 0;
            if(++xt == tilesX)
            {
                xt = 0;
                if(++yt == tilesY)
                {
                    yt = 0;
                    ++img;
                }
            }
        }
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K3: backward weights as a row-direct implicit GEMM on matrix cores, x and dy read in place
// (packed channels-first or channels-last), every filter-width tap of a block's filter rows
// accumulated in registers, the reduction over images and output positions split over blocks
// into fp32 partials that ConvWrwReduce adds in split order (deterministic, no atomics).
// HIPRTC-compatible: no standard headers.
//
// Per group, filter row (t, r) and tap s:
//   dw[k][c][t][r][s] = sum over output rows (n, od, oh) and positions q of
//                       dy[k][od][oh][q] * x[c][id][ih][q * SW + s * DW - pw]
// with id = od sd + t dd - pd and ih = oh sh + r dh - ph. A block owns BK output channels, BC
// input channels and a set of filter rows with all S taps of each. Per step it stages a dy row
// chunk (BK x BQ) and, per filter row, the input window the chunk's taps reach (BC x WIN with
// WIN = (BQ - 1) SW + (S - 1) DW + 1), split into SW stride phases, so tap s reads phase
// (s DW) % SW shifted by (s DW) / SW: the window is staged once for all S taps.
//
// Two loops:
// - k3WrwRows (any stride): a step is one output row chunk with the block's FR filter rows'
//   windows.
// - k3WrwRing (height stride 1): a step is one input row; the last R dy rows' fragments stay in
//   registers, so each window serves all R x S taps of one depth tap.
//
// The matrix-core step is K1's atom: D[k][c] += A[k][q] B[c][q], A = dy, B = x, the reduction
// over positions. Both LDS tiles are position-contiguous rows; a channels-last operand is
// transposed on its way into LDS (KV positions per stored word).
#pragma once

#include "ConvIgemmCore.h"
#include "ConvWrwArgs.h"
#include "ConvWrwGeom.h"

#define K3_NCHW 0
#define K3_NHWC 1
// Channels-last kept channels-last in LDS ([position][channel] rows), the matrix-core
// fragments read with the LDS transpose read (gfx950, 16-bit): a tap's shift is a row offset.
#define K3_NHWC_TR 2

// CONV_K3_DWS 1: the epilogues can also write dw itself (K3Args::dwStore, one split); 0: they
// write fp32 partials only, compiled exactly as before the direct store (the config knob keeps
// it out of tiles it would cost).
#ifndef CONV_K3_DWS
#define CONV_K3_DWS 0
#endif
// CONV_K3_PB 1 (channels-first K3Tiles): the x and dy buffers start at the block's first
// channel, not at the image, and span its bk (dy) or bc (x) channel planes, so an image may
// exceed the 32-bit buffer range as long as a block's channel planes do not.
#ifndef CONV_K3_PB
#define CONV_K3_PB 0
#endif
// The loops are inlined into the kernel entry; with the direct store's second epilogue the
// compiler may otherwise outline the larger ones into a called function.
#if CONV_K3_DWS
#define K3_LOOP_INLINE __attribute__((always_inline))
#else
#define K3_LOOP_INLINE
#endif

/// Compile-time loop: f(K3Int<0>{}) .. f(K3Int<N - 1>{}).
template <int V>
struct K3Int
{
    static constexpr int value = V;
};
template <int I, int N, class F>
__device__ inline void k3ForEachFrom(F&& f)
{
    if constexpr(I < N)
    {
        f(K3Int<I>{});
        k3ForEachFrom<I + 1, N>(f);
    }
}
template <int N, class F>
__device__ inline void k3ForEach(F&& f)
{
    k3ForEachFrom<0, N>(f);
}

/// Buffer loads issued as inline asm, so the kernel, not the compiler, places the waits: the
/// compiler's wait insertion goes conservative (vmcnt(0)) across the loop's control flow and
/// would drain the prefetched step. Each step issues a fixed number of loads (out-of-range
/// ones read zeros), and k3WaitLoads waits until at most N are in flight, then pins the
/// registers it covers so no use moves above the wait.
typedef int k3i4 __attribute__((ext_vector_type(4)));
__device__ inline k3i4 k3Rsrc4(const void* base, long long bytes)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    k3i4 r;
    r[0] = __builtin_amdgcn_readfirstlane(static_cast<int>(p & 0xffffffffu));
    r[1] = __builtin_amdgcn_readfirstlane(static_cast<int>((p >> 32) & 0xffffu));
    r[2] = __builtin_amdgcn_readfirstlane(bytes < 0x7fffffffLL ? static_cast<int>(bytes)
                                                               : 0x7fffffff);
    r[3] = K1_RSRC_FLAGS;
    return r;
}
template <class Raw>
__device__ inline void k3Load(Raw& v, k3i4 rsrc, unsigned offset)
{
    static_assert(sizeof(Raw) == 16 || sizeof(Raw) == 8 || sizeof(Raw) == 4 || sizeof(Raw) == 2,
                  "K3 load size");
    const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>(
            static_cast<unsigned long long>(static_cast<unsigned>(rsrc[0]))
            | (static_cast<unsigned long long>(static_cast<unsigned>(rsrc[1])) << 32)),
        static_cast<short>(0),
        rsrc[2],
        K1_RSRC_FLAGS);
    v = k1BufferLoad<Raw>(r, offset);
}
/// Loads (per k3UaLoads) of one 16-byte unit at any element offset (align 2: channels-last
/// units of channel counts that are not whole vectors): dwords one by one, so each is bounds
/// checked alone (a unit reaching past the buffer keeps its in-range channels), 16-bit units at
/// an odd element shifted into place. A dword is checked whole, so a buffer ending mid-dword
/// (16-bit, an odd element count) would drop its last element: a 2-byte load fetches it.
template <class T>
constexpr int k3UaLoads()
{
    return sizeof(T) == 2 ? 6 : 4;
}
template <class T, class Raw>
__device__ inline void k3LoadUa(Raw& v, k3i4 rsrc, unsigned offset)
{
    static_assert(sizeof(Raw) == 16, "K3 unaligned units are 16 bytes");
    constexpr int N = sizeof(T) == 2 ? 5 : 4;
    const bool out = offset == K1_OUT_OF_RANGE;
    const unsigned base = offset & ~3u;
    unsigned d[N];
#pragma unroll
    for(int i = 0; i < N; ++i)
        k3Load(d[i], rsrc, out ? K1_OUT_OF_RANGE : base + 4u * i);
    unsigned w[4];
    if constexpr(N == 5)
    {
        const unsigned rec = static_cast<unsigned>(rsrc[2]);
        const bool half = (rec & 2u) != 0;
        unsigned short tail;
        k3Load(tail, rsrc, out || !half ? K1_OUT_OF_RANGE : rec - 2u);
#pragma unroll
        for(int i = 0; i < N; ++i)
            d[i] = half && base + 4u * i == (rec & ~3u) ? static_cast<unsigned>(tail) : d[i];
        const unsigned shift = offset & 2u;
#pragma unroll
        for(int i = 0; i < 4; ++i)
            w[i] = __builtin_amdgcn_alignbyte(d[i + 1], d[i], shift);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
            w[i] = d[i];
    }
    v = __builtin_bit_cast(Raw, w);
}
/// Element e of a loaded run (by value: indexing registers through a pointer keeps them in
/// memory).
template <class T, int VEC, class Raw>
__device__ inline T k3Elem(const Raw& v, int e)
{
    if constexpr(sizeof(Raw) == sizeof(T))
    {
        return __builtin_bit_cast(T, v);
    }
    else if constexpr(sizeof(T) == 4)
    {
        // Select the dword by comparison (a runtime vector subscript miscompiled here).
        T r = T(0);
#pragma unroll
        for(int i = 0; i < static_cast<int>(sizeof(Raw) / 4); ++i)
            r = i == e ? __builtin_bit_cast(T, static_cast<unsigned>(v[i])) : r;
        return r;
    }
    else
    {
        // Select the dword by comparison (a runtime vector subscript miscompiled here).
        unsigned d = 0;
        if constexpr(sizeof(Raw) == 4)
        {
            d = __builtin_bit_cast(unsigned, v);
        }
        else
        {
#pragma unroll
            for(int i = 0; i < static_cast<int>(sizeof(Raw) / 4); ++i)
                d = i == e / 2 ? static_cast<unsigned>(v[i]) : d;
        }
        return __builtin_bit_cast(T, static_cast<unsigned short>(d >> (16 * (e % 2))));
    }
}

template <int N>
__device__ inline void k3WaitLoads()
{
    // An ISA-encoding split, so the whole major family: every gfx12 target (gfx12-generic
    // included) has s_wait_loadcnt and no s_waitcnt vmcnt.
#if defined(__GFX12__)
    asm volatile("s_wait_loadcnt %0" ::"n"(N) : "memory");
#else
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
#endif
}
template <class Raw>
__device__ inline void k3Pin(Raw& v)
{
    asm volatile("" : "+v"(v));
}

/// KPL consecutive elements of a position-contiguous LDS row starting OFF elements past p,
/// where p is aligned to the fragment's size. A whole aligned read when OFF is a multiple of
/// KPL; else aligned dword reads funnel-shifted into place (an unaligned LDS vector read is
/// serialised).
template <class Atom, class T, int OFF>
__device__ inline typename Atom::frag k3FragAt(const T* p)
{
    typedef typename Atom::frag F;
    constexpr int KPL = Atom::KPL;
    constexpr int BYTES = KPL * static_cast<int>(sizeof(T));
    if constexpr((KPL == 1 || OFF % KPL == 0) && BYTES <= 16)
    {
        typedef typename K1Raw<BYTES>::type Raw;
        return __builtin_bit_cast(F, *reinterpret_cast<const Raw*>(p + OFF));
    }
    else if constexpr(OFF % KPL == 0)
    {
        // 32-byte fragments (gfx11 WMMA): two aligned 16-byte reads.
        typedef typename K1Raw<16>::type Raw;
        Raw v[BYTES / 16];
#pragma unroll
        for(int i = 0; i < BYTES / 16; ++i)
            v[i] = reinterpret_cast<const Raw*>(p + OFF)[i];
        return __builtin_bit_cast(F, v);
    }
    else if constexpr(sizeof(T) == 2 && (OFF % 2) == 0)
    {
        constexpr int ND = BYTES / 4;
        const unsigned* d = reinterpret_cast<const unsigned*>(p + OFF);
        unsigned v[ND];
#pragma unroll
        for(int i = 0; i < ND; ++i)
            v[i] = d[i];
        return __builtin_bit_cast(F, v);
    }
    else
    {
        static_assert(sizeof(T) == 2, "only 16-bit fragments can be element-shifted");
        constexpr int ND = BYTES / 4;
        const unsigned* d = reinterpret_cast<const unsigned*>(p + (OFF - 1));
        unsigned w[ND + 1];
#pragma unroll
        for(int i = 0; i <= ND; ++i)
            w[i] = d[i];
        unsigned v[ND];
#pragma unroll
        for(int i = 0; i < ND; ++i)
            v[i] = __builtin_amdgcn_alignbyte(w[i + 1], w[i], 2);
        return __builtin_bit_cast(F, v);
    }
}

/// TR layout: per-lane element offset, within a [position][channel] tile of pitch LD, of the
/// fragment of channel lane % DIM, positions kBase(lane) .. + KPL - 1, read by two
/// ds_read_b64_tr_b16 (each gives a lane four positions). In a 16-lane group, source lane s
/// reads four channels 4 (s % 4) .. at position s / 4; the read hands output lane j channel j
/// of the group at those four positions.
template <class Atom, int LD>
__device__ inline int k3TrLane(int lane)
{
    const int s16 = lane % 16;
    const int group = (lane % Atom::DIM) - s16;
    return (Atom::template kBase<Atom::KPL>(lane) + s16 / 4) * LD + group + 4 * (s16 % 4);
}
template <class Atom, class T, int LD>
__device__ inline typename Atom::frag k3FragTr(const T* p)
{
    static_assert(Atom::KPL == 8 && sizeof(T) == 2, "K3 transpose reads: 8 x 16-bit per lane");
#if defined(__gfx950__)
    typedef short s4 __attribute__((ext_vector_type(4)));
    typedef __attribute__((address_space(3))) s4 lds_s4;
    const s4 lo = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p));
    const s4 hi = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(p + 4 * LD));
    return __builtin_bit_cast(typename Atom::frag,
                              __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
#else
    __builtin_trap();
    return typename Atom::frag{};
#endif
}

/// A 16-byte LDS store at an address aligned only to the element: one unaligned ds write
/// (the compiler would split it into element stores).
template <class T, class Raw>
__device__ inline void k3StoreUnaligned(T* dst, const Raw& v)
{
    const unsigned addr = static_cast<unsigned>(
        reinterpret_cast<unsigned long long>((__attribute__((address_space(3))) T*)dst));
#if K1_HAS_WMMA11 || K1_HAS_WMMA12
    asm volatile("ds_store_b128 %0, %1" ::"v"(addr), "v"(v) : "memory");
#else
    asm volatile("ds_write_b128 %0, %1" ::"v"(addr), "v"(v) : "memory");
#endif
}

/// Keep-mask of a 16-byte run: dword i keeps element e when bit e of `mask` is set.
template <class T, int VEC>
__device__ inline void k3MaskRun(typename K1Raw<VEC * sizeof(T)>::type& v, int mask)
{
    unsigned* d = reinterpret_cast<unsigned*>(&v);
    constexpr int ND = VEC * static_cast<int>(sizeof(T)) / 4;
#pragma unroll
    for(int i = 0; i < ND; ++i)
    {
        if constexpr(sizeof(T) == 4)
        {
            d[i] = ((mask >> i) & 1) ? d[i] : 0u;
        }
        else
        {
            const unsigned lo = ((mask >> (2 * i)) & 1) ? 0x0000ffffu : 0u;
            const unsigned hi = ((mask >> (2 * i + 1)) & 1) ? 0xffff0000u : 0u;
            d[i] &= lo | hi;
        }
    }
}

/// Global loads and LDS stores of one reduction step.
///
/// A (dy): BK rows (output channels) by BQ positions; LDS As[k][q], pitch LDA.
/// B (x): per filter row and stride phase, BC rows (input channels) by PH phase positions;
/// LDS Bs[(f SW + phase) BC + c][j], pitch LDB, window position p = phase + SW j.
///
/// Channels-first: a unit is VEC consecutive positions of one row (positions fastest across
/// lanes); channels-last: KV positions (stored as one word per channel) by VEC channels,
/// positions fastest. Offsets are elements within one image (the launcher keeps an image
/// within 2^31 bytes); each thread's offset parts and LDS addresses are fixed per block. The
/// loads are straight-line (no branch around a load, so the compiler counts loads in flight
/// exactly) and every step issues the same number (out-of-range ones read zeros). A
/// channels-first run is loaded whole from the nearest in-image offset; a step whose window or
/// chunk leaves its rows (uniform) zeros the out-of-row elements, and shifts a run clamped at
/// the image's ends, when it stores.
template <class T,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int FR,
          int S,
          int SW,
          int DW,
          int VEC,
          int THREADS,
          int PF = 1,
          int ALIGN = 0>
struct K3Tiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr bool TR = LAYOUT == K3_NHWC_TR;
    static constexpr int KV = LAYOUT == K3_NHWC ? 4 / static_cast<int>(sizeof(T)) : 1;
    static constexpr int WIN = (BQ - 1) * SW + (S - 1) * DW + 1;
    // ALIGN 1 (channels-first, width stride 1, rows a whole number of VEC): units start at
    // multiples of VEC within the row, so each is wholly in or out of it (out: read as
    // zeros), and the store shifts the window into place with an unaligned LDS write.
    // ALIGN 2 (channels-last, transposed into LDS): dy units start at any element, for output
    // channel counts that are not whole vectors; the channels a unit reads past the group land
    // in tile rows the partials drop.
    static_assert(ALIGN == 0 || (ALIGN == 1 && LAYOUT == K3_NCHW && SW == 1)
                      || (ALIGN == 2 && LAYOUT == K3_NHWC && VEC * sizeof(T) == 16),
                  "K3 aligned units");
    // Channels-first: VEC-position units over the window; each phase gets VEC / SW of a unit.
    static constexpr int NBV = (WIN + (ALIGN == 1 ? VEC - 1 : 0) + VEC - 1) / VEC;
    // Phase length: the window's positions in a phase, rounded so units and words fit, plus
    // two elements of slack for the funnel-shifted fragment reads.
    static constexpr int PH_RAW = LAYOUT == K3_NCHW ? NBV * VEC / SW : (WIN + SW - 1) / SW;
    static constexpr int PH = (PH_RAW + KV - 1) / KV * KV;
    static constexpr int SKEW = 16 / static_cast<int>(sizeof(T));
    // TR: rows are positions of BK (dy) or BC (x) channels, pitch padded by 16 bytes; the x
    // tile holds PH rows per phase. Else rows are channels of positions.
    static constexpr int LDA = TR ? k3TrPitch(BK) : BQ + SKEW;
    static constexpr int LDB = TR ? k3TrPitch(BC) : (PH + 2 + SKEW - 1) / SKEW * SKEW + SKEW;
    // ALIGN: a guard of VEC elements ahead of the x tile takes the first row's shifted spill.
    static constexpr int A_ELEMS = (TR ? BQ : BK) * LDA + (ALIGN == 1 ? VEC : 0);
    static constexpr int B_ELEMS = FR * SW * (TR ? PH : BC) * LDB;
    static constexpr int NJ = PH / KV;
    // Window positions the loads cover (the interior test).
    static constexpr int SPAN = LAYOUT == K3_NCHW ? NBV * VEC : SW * PH;
    static constexpr int A_UNITS = LAYOUT == K3_NCHW ? BK * (BQ / VEC) : (BQ / KV) * (BK / VEC);
    static constexpr int B_UNITS = LAYOUT == K3_NCHW ? FR * BC * NBV : FR * SW * NJ * (BC / VEC);
    static_assert(!TR || sizeof(T) == 2, "K3 transpose reads take 16-bit data");
    static constexpr int A_PASSES = (A_UNITS + THREADS - 1) / THREADS;
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static constexpr int A_REGS = LAYOUT == K3_NCHW ? A_PASSES : A_PASSES * KV;
    static constexpr int B_REGS = LAYOUT == K3_NCHW ? B_PASSES : B_PASSES * KV;
    /// Loads one step issues.
    static constexpr int LOADS = A_REGS * (ALIGN == 2 ? k3UaLoads<T>() : 1) + B_REGS;
    static_assert(LAYOUT == K3_NHWC || (BQ % VEC == 0 && VEC % SW == 0), "K3 channels-first units");
    static_assert(LAYOUT == K3_NCHW || (BK % VEC == 0 && BC % VEC == 0 && BQ % KV == 0),
                  "K3 channels-last units");
    static_assert(ALIGN == 0 || !TR, "K3 aligned units are channels-first");
    static_assert(A_PASSES <= 32 && B_PASSES <= 32, "K3 pass masks");

    /// One loaded step: its registers and what the store needs to fix an edge step.
    struct Step
    {
        Raw ra[A_REGS], rb[B_REGS];
        int q0, yRow, xRow[FR], shiftX;
        bool yOk, xOk[FR], edge;
    };
    Step st[PF];
    // Per pass: offset part (channel, position in chunk / window), LDS element offset, filter
    // row (FR > 1), and whether the unit exists with its channel in range.
    int aOff[A_PASSES], aLds[A_PASSES], aPos[A_PASSES];
    int bOff[B_PASSES], bLds[B_PASSES], bPos[B_PASSES], bF[B_PASSES];
    unsigned aOk, bOk; // bit p: pass p in range
    /// Block-based buffers (CONV_K3_PB): the dy and x elements before the block's first
    /// channel within an image, and the elements its buffers span.
    static constexpr bool PB = CONV_K3_PB != 0;
    static_assert(!PB || LAYOUT == K3_NCHW, "K3 block-based buffers are channels-first");
    long long yBase, xBase;
    int yWin, xWin;

    __device__ void init(const K3Args& a, int tid, int g, int k0, int c0, int kg, int cg)
    {
        aOk = 0;
        bOk = 0;
        if constexpr(PB)
        {
            yBase = static_cast<long long>(g * kg + k0) * a.ys[1];
            xBase = static_cast<long long>(g * cg + c0) * a.xs[1];
            const long long yLeft = a.yImage - yBase, xLeft = a.xImage - xBase;
            yWin = static_cast<int>(yLeft < BK * a.ys[1] ? yLeft : BK * a.ys[1]);
            xWin = static_cast<int>(xLeft < BC * a.xs[1] ? xLeft : BC * a.xs[1]);
        }
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            int k;
            if(LAYOUT == K3_NCHW)
            {
                const int row = u / (BQ / VEC), qv = u % (BQ / VEC);
                k = k0 + row;
                aPos[p] = qv * VEC;
                aOff[p] = (PB ? row : g * kg + k) * static_cast<int>(a.ys[1]) + aPos[p];
                aLds[p] = row * LDA + qv * VEC;
            }
            else if constexpr(TR)
            {
                // Channels fastest across lanes: a wave reads whole channel runs.
                const int kv = u % (BK / VEC), qp = u / (BK / VEC);
                k = k0 + kv * VEC;
                aPos[p] = qp;
                aOff[p] = aPos[p] * static_cast<int>(a.ys[4]) + g * kg + k;
                aLds[p] = qp * LDA + kv * VEC;
            }
            else
            {
                const int qp = u % (BQ / KV), kv = u / (BQ / KV);
                k = k0 + kv * VEC;
                aPos[p] = qp * KV;
                aOff[p] = aPos[p] * static_cast<int>(a.ys[4]) + g * kg + k;
                aLds[p] = (kv * VEC) * LDA + qp * KV;
            }
            if(u < A_UNITS && k < kg)
                aOk |= 1u << p;
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            int c;
            if(LAYOUT == K3_NCHW)
            {
                const int v = u % NBV, rowc = u / NBV;
                const int cl = rowc % BC;
                c = c0 + cl;
                bF[p] = rowc / BC;
                bPos[p] = v * VEC;
                bOff[p] = (PB ? cl : g * cg + c) * static_cast<int>(a.xs[1]) + bPos[p];
                bLds[p] = ((bF[p] * SW) * BC + cl) * LDB + v * (VEC / SW);
            }
            else if constexpr(TR)
            {
                // Channels fastest across lanes, then the phase's positions.
                const int cv = u % (BC / VEC);
                const int rest = u / (BC / VEC);
                const int jp = rest % NJ;
                const int rest2 = rest / NJ;
                const int phase = rest2 % SW;
                c = c0 + cv * VEC;
                bF[p] = rest2 / SW;
                bPos[p] = phase + SW * jp;
                bOff[p] = bPos[p] * static_cast<int>(a.xs[4]) + g * cg + c;
                bLds[p] = ((bF[p] * SW + phase) * PH + jp) * LDB + cv * VEC;
            }
            else
            {
                const int jp = u % NJ;
                const int rest = u / NJ;
                const int cv = rest % (BC / VEC);
                const int rest2 = rest / (BC / VEC);
                const int phase = rest2 % SW;
                c = c0 + cv * VEC;
                bF[p] = rest2 / SW;
                bPos[p] = phase + SW * jp * KV;
                bOff[p] = bPos[p] * static_cast<int>(a.xs[4]) + g * cg + c;
                bLds[p] = ((bF[p] * SW + phase) * BC + cv * VEC) * LDB + jp * KV;
            }
            if(u < B_UNITS && c < cg)
                bOk |= 1u << p;
        }
    }

    /// Pass p's x row: its filter row's validity and offset.
    __device__ void
        rowOf(int p, const bool (&xOk)[FR], const int (&xRow)[FR], bool& rowOk, int& row) const
    {
        rowOk = (bOk >> p) & 1u;
        row = xRow[0];
        if constexpr(FR == 1)
        {
            rowOk = rowOk && xOk[0];
        }
        else
        {
            bool any = false;
#pragma unroll
            for(int f = 0; f < FR; ++f)
                if(bF[p] == f)
                {
                    any = xOk[f];
                    row = xRow[f];
                }
            rowOk = rowOk && any;
        }
    }

    /// Issues the global loads of one step into register set P: the dy row (yOk, element offset
    /// yRow of its depth and height in the image) from position q0, and per filter row f the
    /// x row (xOk[f], xRow[f]) window from position q0 SW - pw, all of image n.
    template <int P = 0>
    __device__ void load(const K3Args& a,
                         int n,
                         int q0,
                         bool yOk,
                         int yRow,
                         const bool (&xOk)[FR],
                         const int (&xRow)[FR])
    {
        Step& s = st[P];
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * a.yImage + (PB ? yBase : 0),
                                (PB ? yWin : a.yImage) * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * a.xImage + (PB ? xBase : 0),
                                (PB ? xWin : a.xImage) * sizeof(T));
        const int iw0 = q0 * SW - a.pw;
        const int yElems = PB ? yWin : static_cast<int>(a.yImage);
        const int xElems = PB ? xWin : static_cast<int>(a.xImage);
        const int ys4 = static_cast<int>(a.ys[4]), xs4 = static_cast<int>(a.xs[4]);
        s.q0 = q0;
        s.yRow = yRow;
        s.yOk = yOk;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            s.xRow[f] = xRow[f];
            s.xOk[f] = xOk[f];
        }
        s.edge
            = LAYOUT == K3_NCHW && ALIGN != 1 && (q0 + BQ > a.wo || iw0 < 0 || iw0 + SPAN > a.wi);
        // ALIGN: the units start at iwA, the multiple of VEC at or below iw0.
        const int iwA = iw0 >= 0 ? iw0 / VEC * VEC : -((-iw0 + VEC - 1) / VEC) * VEC;
        s.shiftX = iwA - iw0;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const bool ok = yOk && ((aOk >> p) & 1u);
            if(LAYOUT == K3_NCHW)
            {
                const int off = aOff[p] + yRow + q0;
                if constexpr(ALIGN == 1)
                {
                    const bool in = ok && q0 + aPos[p] < a.wo;
                    k3Load(
                        s.ra[p], yr, in ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                }
                else
                {
                    const int lo = off < 0 ? 0 : (off > yElems - VEC ? yElems - VEC : off);
                    k3Load(
                        s.ra[p], yr, ok ? static_cast<unsigned>(lo) * sizeof(T) : K1_OUT_OF_RANGE);
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const bool okq = ok && q0 + aPos[p] + j < a.wo;
                    const int off = aOff[p] + yRow + q0 * ys4 + j * ys4;
                    const unsigned at
                        = okq ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE;
                    if constexpr(ALIGN == 2)
                        k3LoadUa<T>(s.ra[p * KV + j], yr, at);
                    else
                        k3Load(s.ra[p * KV + j], yr, at);
                }
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            bool rowOk;
            int row;
            rowOf(p, xOk, xRow, rowOk, row);
            if(LAYOUT == K3_NCHW)
            {
                if constexpr(ALIGN == 1)
                {
                    const int iw = iwA + bPos[p];
                    const bool in = rowOk && iw >= 0 && iw < a.wi;
                    const int off = bOff[p] + row + iwA;
                    k3Load(
                        s.rb[p], xr, in ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                }
                else
                {
                    const int off = bOff[p] + row + iw0;
                    const int lo = off < 0 ? 0 : (off > xElems - VEC ? xElems - VEC : off);
                    k3Load(s.rb[p],
                           xr,
                           rowOk ? static_cast<unsigned>(lo) * sizeof(T) : K1_OUT_OF_RANGE);
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const int iw = iw0 + bPos[p] + SW * j;
                    const bool ok = rowOk && iw >= 0 && iw < a.wi;
                    const int off = bOff[p] + row + iw0 * xs4 + j * SW * xs4;
                    k3Load(s.rb[p * KV + j],
                           xr,
                           ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                }
            }
        }
    }

    /// Waits for register set P's loads with at most N loads (newer steps') still in flight.
    template <int P, int N>
    __device__ void wait()
    {
        k3WaitLoads<N>();
#pragma unroll
        for(int i = 0; i < A_REGS; ++i)
            k3Pin(st[P].ra[i]);
#pragma unroll
        for(int i = 0; i < B_REGS; ++i)
            k3Pin(st[P].rb[i]);
    }

    /// A channels-first run's fix-up on an edge step: keep-mask of in-row elements and the
    /// shift of a run clamped at the image's ends (loaded index = stored index + shift).
    __device__ static void fixOf(int off, int pos, int limit, int elems, int& mask, int& shift)
    {
        const int lo = off < 0 ? 0 : (off > elems - VEC ? elems - VEC : off);
        shift = off - lo;
        const int first = pos < 0 ? (-pos < VEC ? -pos : VEC) : 0;
        const int room = limit - pos;
        const int stop = room < VEC ? (room > 0 ? room : 0) : VEC;
        mask = first >= stop ? 0 : ((1 << stop) - 1) & ~((1 << first) - 1);
    }

    /// Stores run v at dst with its fix-up.
    __device__ static void storeFixed(T* dst, Raw v, int mask, int shift)
    {
        if(shift == 0)
        {
            k3MaskRun<T, VEC>(v, mask);
            *reinterpret_cast<Raw*>(dst) = v;
            return;
        }
        T tmp[VEC];
        const T* e = reinterpret_cast<const T*>(&v);
#pragma unroll
        for(int j = 0; j < VEC; ++j)
            tmp[j] = T(0);
#pragma unroll
        for(int i = 0; i < VEC; ++i)
        {
            const int j = i - shift;
#pragma unroll
            for(int jj = 0; jj < VEC; ++jj)
                if(jj == j && ((mask >> jj) & 1))
                    tmp[jj] = e[i];
        }
        *reinterpret_cast<Raw*>(dst) = __builtin_bit_cast(Raw, tmp);
    }

    /// Stores the loaded step (register set P) into LDS.
    template <int P = 0>
    __device__ void store(const K3Args& a, T* As, T* Bs, int tid) const
    {
        const Step& s = st[P];
        if(LAYOUT == K3_NCHW && s.edge)
        {
            storeEdge<P>(a, As, Bs, tid);
            return;
        }
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                continue;
            if(LAYOUT == K3_NCHW || TR)
                *reinterpret_cast<Raw*>(As + aLds[p]) = s.ra[p * KV];
            else
                storeWords(As + aLds[p], LDA, &s.ra[p * KV]);
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            if(LAYOUT == K3_NCHW && ALIGN)
                k3StoreUnaligned(Bs + bLds[p] + s.shiftX, s.rb[p]);
            else if(LAYOUT == K3_NCHW)
                storePhases(Bs + bLds[p], s.rb[p]);
            else if(TR)
                *reinterpret_cast<Raw*>(Bs + bLds[p]) = s.rb[p];
            else
                storeWords(Bs + bLds[p], LDB, &s.rb[p * KV]);
        }
    }

    template <int P>
    __device__ void storeEdge(const K3Args& a, T* As, T* Bs, int tid) const
    {
        const Step& s = st[P];
        const int yElems = PB ? yWin : static_cast<int>(a.yImage);
        const int xElems = PB ? xWin : static_cast<int>(a.xImage);
        const int iw0 = s.q0 * SW - a.pw;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                continue;
            int mask, shift;
            fixOf(aOff[p] + s.yRow + s.q0, s.q0 + aPos[p], a.wo, yElems, mask, shift);
            storeFixed(As + aLds[p], s.ra[p], mask, shift);
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            bool rowOk;
            int row;
            rowOf(p, s.xOk, s.xRow, rowOk, row);
            int mask, shift;
            fixOf(bOff[p] + row + iw0, iw0 + bPos[p], a.wi, xElems, mask, shift);
            if(SW == 1)
            {
                storeFixed(Bs + bLds[p], s.rb[p], mask, shift);
            }
            else
            {
                T tmp[VEC];
                storeFixed(tmp, s.rb[p], mask, shift);
                storePhases(Bs + bLds[p], __builtin_bit_cast(Raw, tmp));
            }
        }
    }

    /// A channels-first window run split into its SW stride phases.
    __device__ static void storePhases(T* dst, const Raw& run)
    {
        if constexpr(SW == 1)
        {
            *reinterpret_cast<Raw*>(dst) = run;
        }
        else
        {
            const T* e = reinterpret_cast<const T*>(&run);
#pragma unroll
            for(int ph = 0; ph < SW; ++ph)
#pragma unroll
                for(int i = 0; i < VEC / SW; ++i)
                    dst[ph * BC * LDB + i] = e[ph + SW * i];
        }
    }

    /// Channels-last transpose: KV loaded vectors (one per position) of VEC channels become VEC
    /// words of KV positions, one per channel row of pitch LD.
    /// The words are built from the loaded vectors' dwords by value: indexing the registers
    /// through a pointer made the compiler keep them in memory (LDS) instead.
    __device__ static void storeWords(T* dst, int ld, const Raw* src)
    {
        static_assert(sizeof(Raw) == 16, "K3 channels-last units are 16 bytes");
#pragma unroll
        for(int i = 0; i < VEC; ++i)
        {
            if constexpr(KV == 1)
            {
                const unsigned d = src[0][i];
                *reinterpret_cast<unsigned*>(dst + i * ld) = d;
            }
            else
            {
                const int sh = 16 * (i % 2);
                const unsigned lo = (src[0][i / 2] >> sh) & 0xffffu;
                const unsigned hi = (src[1][i / 2] >> sh) & 0xffffu;
                *reinterpret_cast<unsigned*>(dst + i * ld) = lo | (hi << 16);
            }
        }
    }
};

/// Packed output rows (loop 7, channels-first): one step's BQ positions hold several output
/// rows of one (image, depth) plane, row r of the step at positions r WV .. r WV + wo - 1 of a
/// virtual row of WV = a.segs >= wo + (S - 1) DW / SW columns, a.segRows = floor(BQ / WV) rows a
/// step. The x tile holds, per filter row, each step row's input row at window positions
/// r WV SW + pw + iw, so window position j SW + s DW is x for position j and tap s as on one row
/// and the LDS tiles and multiply are K3Tiles' channels-first ones. Positions past a row's wo
/// (and rows past the plane) hold zero dy; every staged position is written each step (zero
/// outside the image), so stale LDS never meets a product. Units are elements, the position
/// fastest. A dy unit's position is the thread's own (THREADS a multiple of BQ), its row and
/// column computed once; an x unit's step row and input column (runtime divisors) are computed
/// once per pass and kept packed in one register.
template <class T, int BK, int BC, int BQ, int FR, int S, int SW, int DW, int VEC, int THREADS>
struct K3PackTiles
{
    typedef K3Tiles<T, K3_NCHW, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS> Geo;
    typedef typename K1Raw<sizeof(T)>::type RawE;
    static constexpr bool TR = false;
    static constexpr int LDA = Geo::LDA, LDB = Geo::LDB, PH = Geo::PH;
    static constexpr int A_ELEMS = Geo::A_ELEMS, B_ELEMS = Geo::B_ELEMS;
    static constexpr int SPAN = Geo::SPAN;
    static constexpr int A_PASSES = BK * BQ / THREADS;
    static constexpr int B_UNITS = FR * BC * SPAN;
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static_assert(THREADS % BQ == 0 && A_PASSES * THREADS == BK * BQ,
                  "K3 packed rows: a thread keeps one dy position");
    RawE ra[A_PASSES], rb[B_PASSES];
    /// An x unit's step row (low 16 bits) and input column (high 16 bits, below SPAN), or -1
    /// where it stages zero (past the step's rows or outside the input row).
    int bInfo[B_PASSES];
    int jA, rA, colA; ///< the thread's dy position: step row and column
    int kA0;

    __device__ static void unitOf(int u, int& f, int& pos, int& c)
    {
        pos = u % SPAN;
        const int rest = u / SPAN;
        c = rest % BC;
        f = rest / BC;
    }

    __device__ void init(const K3Args& a, int tid, int /*g*/, int /*k0*/, int /*c0*/, int, int)
    {
        const int wv = a.segs, rows = a.segRows;
        jA = tid % BQ;
        rA = jA / wv;
        colA = jA - rA * wv;
        if(rA >= rows)
            colA = a.wo; // past every row: zero dy
        kA0 = tid / BQ;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            int f, pos, c;
            unitOf(tid + p * THREADS, f, pos, c);
            const int r = pos / (wv * SW);
            const int iw = pos - r * wv * SW - a.pw;
            const bool ok = r < rows && iw >= 0 && iw < a.wi && tid + p * THREADS < B_UNITS;
            bInfo[p] = ok ? r | (iw << 16) : -1;
        }
    }

    /// The global loads of one step: rows oh0 .. oh0 + segRows - 1 of plane (n, od). Filter row
    /// f: fOk[f] (the row exists and its depth tap is in the image), xBase[f] its depth offset,
    /// ihOf[f] its height offset (oh sh + ihOf[f] is the input row).
    __device__ void load(const K3Args& a,
                         int n,
                         int g,
                         int k0,
                         int c0,
                         int od,
                         int oh0,
                         bool live,
                         const bool (&fOk)[FR],
                         const int (&xBase)[FR],
                         const int (&ihOf)[FR])
    {
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * a.yImage, a.yImage * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * a.xImage, a.xImage * sizeof(T));
        const int kg = a.k / a.groups, cg = a.c / a.groups;
        const int ys1 = static_cast<int>(a.ys[1]), ys2 = static_cast<int>(a.ys[2]),
                  ys3 = static_cast<int>(a.ys[3]);
        const int xs1 = static_cast<int>(a.xs[1]), xs3 = static_cast<int>(a.xs[3]);
        const int ohA = oh0 + rA;
        const bool rowA = live && colA < a.wo && ohA < a.ho;
        const int yBase = od * ys2 + ohA * ys3 + colA;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int k = k0 + kA0 + p * (THREADS / BQ);
            const bool ok = rowA && k < kg;
            const int off = (g * kg + k) * ys1 + yBase;
            k3Load(ra[p], yr, ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
        }
        const int tid = threadIdx.x;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            int f, pos, c;
            unitOf(tid + p * THREADS, f, pos, c);
            const int r = bInfo[p] & 0xffff;
            const int iw = bInfo[p] >> 16;
            bool fok = false;
            int base = 0, ihOff = 0;
#pragma unroll
            for(int ff = 0; ff < FR; ++ff)
                if(ff == f)
                {
                    fok = fOk[ff];
                    base = xBase[ff];
                    ihOff = ihOf[ff];
                }
            const int oh = oh0 + r;
            const int ih = oh * a.sh + ihOff;
            const bool ok
                = live && bInfo[p] >= 0 && oh < a.ho && fok && c0 + c < cg && ih >= 0 && ih < a.hi;
            const int off = (g * cg + c0 + c) * xs1 + base + ih * xs3 + iw;
            k3Load(rb[p], xr, ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
        }
    }

    template <int P, int N>
    __device__ void wait()
    {
        k3WaitLoads<N>();
#pragma unroll
        for(int i = 0; i < A_PASSES; ++i)
            k3Pin(ra[i]);
#pragma unroll
        for(int i = 0; i < B_PASSES; ++i)
            k3Pin(rb[i]);
    }

    template <int P = 0>
    __device__ void store(const K3Args& /*a*/, T* As, T* Bs, int tid) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            As[(kA0 + p * (THREADS / BQ)) * LDA + jA] = __builtin_bit_cast(T, ra[p]);
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            int f, pos, c;
            unitOf(u, f, pos, c);
            Bs[((f * SW + pos % SW) * BC + c) * LDB + pos / SW] = __builtin_bit_cast(T, rb[p]);
        }
    }
};

/// The block's (tile, row set, split) indices. XCD 1: logical blocks run contiguously per XCD
/// (gfx942 / gfx950 dispatch round-robin over 8), so the channel tiles of one split, which
/// read the same rows, share an L2.
template <int XCD>
__device__ inline void k3BlockIds(unsigned& bx, unsigned& by, unsigned& bz)
{
    bx = blockIdx.x;
    by = blockIdx.y;
    bz = blockIdx.z;
    if constexpr(XCD != 0)
        k1XcdOrder(bx, by, bz);
}

/// One step's accumulation from LDS: per k-atom, filter row f's window (fStride elements per
/// row; 0 when every row shares one window) against dy fragments dyf[f], all S taps, into
/// acc[f S + s].
template <class Atom,
          class T,
          int FR,
          int S,
          int SW,
          int DW,
          int FK,
          int FC,
          int BC,
          int LDB,
          int KK,
          int KB = 0,
          int KE = KK,
          int TR = 0,
          int PHR = 0>
__device__ inline void k3Multiply(typename Atom::acc (&acc)[FR * S][FK][FC],
                                  const typename Atom::frag (&dyf)[FR][KK][FK],
                                  const T* bBase,
                                  int fStride)
{
    constexpr int DIM = Atom::DIM, KA = Atom::KA;
    // TR: bBase is the lane's k3TrLane offset into the [row][channel] tile, PHR rows a phase.
#pragma unroll
    for(int kk = KB; kk < KE; ++kk)
    {
        k3ForEach<S>([&](auto sc) {
            constexpr int s = decltype(sc)::value;
            constexpr int ph = (s * DW) % SW, off = (s * DW) / SW;
#pragma unroll
            for(int y = 0; y < FC; ++y)
            {
                typename Atom::frag bf;
#pragma unroll
                for(int f = 0; f < FR; ++f)
                {
                    if(f == 0 || fStride != 0)
                    {
                        if constexpr(TR != 0)
                            bf = k3FragTr<Atom, T, LDB>(
                                bBase + f * fStride + (ph * PHR + kk * KA + off) * LDB + y * DIM);
                        else
                            bf = k3FragAt<Atom, T, off>(bBase + f * fStride
                                                        + (ph * BC + y * DIM) * LDB + kk * KA);
                    }
#pragma unroll
                    for(int x = 0; x < FK; ++x)
                    {
                        acc[f * S + s][x][y] = Atom::mma(dyf[f][kk][x], bf, acc[f * S + s][x][y]);
                    }
                }
            }
        });
    }
}

/// k3Multiply over only the tap rows whose bit is set in `rows` (a track's first and last R - 1
/// steps pair the window with dy rows outside the image; skipping them saves the multiplies).
template <class Atom,
          class T,
          int R,
          int S,
          int SW,
          int DW,
          int FK,
          int FC,
          int BC,
          int LDB,
          int KK,
          int TR = 0,
          int PHR = 0>
__device__ inline void k3MultiplyRows(typename Atom::acc (&acc)[R * S][FK][FC],
                                      const typename Atom::frag (&dyf)[R][KK][FK],
                                      const T* bBase,
                                      unsigned rows)
{
    if(rows == (1u << R) - 1u)
    {
        k3Multiply<Atom, T, R, S, SW, DW, FK, FC, BC, LDB, KK, 0, KK, TR, PHR>(acc, dyf, bBase, 0);
        return;
    }
    k3ForEach<R>([&](auto rc) {
        constexpr int r = decltype(rc)::value;
        if((rows >> r) & 1u)
        {
            typedef typename Atom::acc AccRow[S][FK][FC];
            typedef const typename Atom::frag DyRow[1][KK][FK];
            k3Multiply<Atom, T, 1, S, SW, DW, FK, FC, BC, LDB, KK, 0, KK, TR, PHR>(
                *reinterpret_cast<AccRow*>(&acc[r * S]),
                *reinterpret_cast<DyRow*>(&dyf[r]),
                bBase,
                0);
        }
    });
}

/// Tap rows r whose dy row, m - r of a track with U rows, exists.
__device__ inline unsigned k3RowsOf(int m, int u, int r)
{
    unsigned rows = 0;
    for(int i = 0; i < r; ++i)
        rows |= (m - i >= 0 && m - i < u) ? 1u << i : 0u;
    return rows;
}

/// Where a kernel's sums of group g go: element (tap, k, c) (tap of `taps`, k and c within the
/// group) at base + tap ts + k ks + c. The fp32 partials of `split` ([split][g][tap][k][c]), or
/// under a.dwStore dw itself in T (taps reflected under flip: a negative tap stride from the
/// last tap). Channels are contiguous in both, so the epilogues share their addressing.
struct K3Out
{
    void* base;
    long long ks;
    long long ts;
};

template <class T>
__device__ inline K3Out k3OutOf(const K3Args& a, int g, int taps, unsigned split)
{
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    if(a.dwStore != 0)
    {
        T* dw = static_cast<T*>(a.dwOut) + static_cast<long long>(g) * kg * a.dwk
                + (a.flip ? static_cast<long long>(taps - 1) * a.dwtap : 0);
        return {dw, a.dwk, a.flip ? -a.dwtap : a.dwtap};
    }
    const long long kc = static_cast<long long>(kg) * cg;
    return {a.partial + split * a.partialElems + static_cast<long long>(g) * taps * kc, cg, kc};
}

template <class P>
struct K3PointeeOf;
template <class E>
struct K3PointeeOf<E*>
{
    typedef E type;
};

/// Runs store(p), p o's base typed for its elements: float* for the partials (and an fp32 dw),
/// T* for a 16-bit dw. The branch is taken once, outside the store's unrolled loops (a branch
/// per element made the 16-bit epilogues twice as long).
template <class T, class Store>
__device__ inline void k3WithOut(const K3Args& a, const K3Out& o, Store&& store)
{
    if constexpr(sizeof(T) == sizeof(float))
        store(static_cast<float*>(o.base));
    else if(a.dwStore != 0)
        store(static_cast<T*>(o.base));
    else
        store(static_cast<float*>(o.base));
}

/// The sums, lanes along c: acc[f S + s] holds tap (tapRow0 + f) S + s; filter rows at or past
/// `frows` are skipped. Into fp32 partials [split][g][tap][k][c], or dw (k3OutOf).
template <class T, class Atom, int FR, int S, int FK, int FC, int WTK, int WTC>
__device__ inline void k3StorePartials(const K3Args& a,
                                       const typename Atom::acc (&acc)[FR * S][FK][FC],
                                       int g,
                                       int tapRow0,
                                       int frows,
                                       int k0,
                                       int c0,
                                       int wk,
                                       int wc,
                                       int lane,
                                       unsigned split,
                                       int rowStep = 1,
                                       int rowEnd = 0x7fffffff)
{
    constexpr int DIM = Atom::DIM;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    if constexpr(!CONV_K3_DWS)
    {
        const long long kc = static_cast<long long>(kg) * cg;
        const int taps = frows * S;
        float* out = a.partial + split * a.partialElems + static_cast<long long>(g) * taps * kc;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            // Filter row tapRow0 + f rowStep (the ring's height phases step by the stride).
            const int row = tapRow0 + f * rowStep;
            if(row >= frows || row >= rowEnd)
                continue;
#pragma unroll
            for(int s = 0; s < S; ++s)
            {
                float* o = out + static_cast<long long>(row * S + s) * kc;
#pragma unroll
                for(int x = 0; x < FK; ++x)
#pragma unroll
                    for(int y = 0; y < FC; ++y)
                    {
                        const int c = c0 + wc * WTC + y * DIM + lane % DIM;
#pragma unroll
                        for(int e = 0; e < Atom::ACC; ++e)
                        {
                            const int k = k0 + wk * WTK + x * DIM + Atom::dRow(lane, e);
                            if(k < kg && c < cg)
                                o[static_cast<long long>(k) * cg + c] = acc[f * S + s][x][y][e];
                        }
                    }
            }
        }
        return;
    }
    const K3Out out = k3OutOf<T>(a, g, frows * S, split);
    k3WithOut<T>(a, out, [&](auto* base) {
        typedef typename K3PointeeOf<decltype(base)>::type E;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            // Filter row tapRow0 + f rowStep (the ring's height phases step by the stride).
            const int row = tapRow0 + f * rowStep;
            if(row >= frows || row >= rowEnd)
                continue;
#pragma unroll
            for(int s = 0; s < S; ++s)
            {
                E* o = base + static_cast<long long>(row * S + s) * out.ts;
#pragma unroll
                for(int x = 0; x < FK; ++x)
#pragma unroll
                    for(int y = 0; y < FC; ++y)
                    {
                        const int c = c0 + wc * WTC + y * DIM + lane % DIM;
#pragma unroll
                        for(int e = 0; e < Atom::ACC; ++e)
                        {
                            const int k = k0 + wk * WTK + x * DIM + Atom::dRow(lane, e);
                            if(k < kg && c < cg)
                                o[k * out.ks + c] = K1Elem<E>::store(acc[f * S + s][x][y][e]);
                        }
                    }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Output-row kernel (any stride). Block: WK x WC waves over output and input channels, tile
// BK x BC, FR filter rows (of the ft x fr flattened rows) by S taps, BQ positions per step.
// Grid: (k tiles x c tiles, groups x filter-row groups, splits). Steps: (n, od, oh, chunk),
// stepsPerSplit per split. Two LDS buffers, one barrier per step.
template <class T,
          class Atom,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int FR,
          int S,
          int SW,
          int DW,
          int WK,
          int WC,
          int VEC,
          int XCD = 0,
          int PACK = 0>
__device__ K3_LOOP_INLINE void k3WrwRows(const K3Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WK * WC * WAVE;
    constexpr int DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int WTK = BK / WK, WTC = BC / WC;
    constexpr int FK = WTK / DIM, FC = WTC / DIM;
    constexpr int KK = BQ / KA;
    static_assert(FK * DIM * WK == BK && FC * DIM * WC == BC && BQ % KA == 0, "K3 tile");
    static_assert(PACK == 0 || LAYOUT == K3_NCHW, "K3 packed rows are channels-first");
    typedef typename K1If<PACK != 0,
                          K3PackTiles<T, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS>,
                          K3Tiles<T, LAYOUT, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS>>::type Tiles;
    static_assert(
        k3LdsBytes(0, LAYOUT, sizeof(T), BK, BC, BQ, FR, S, SW, DW, WK, WC, 0, 0, VEC, 0, 0)
            == 2 * (Tiles::A_ELEMS + Tiles::B_ELEMS) * sizeof(T),
        "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int LDA = Tiles::LDA, LDB = Tiles::LDB;
    constexpr int BUF = Tiles::A_ELEMS + Tiles::B_ELEMS;

    __shared__ __attribute__((aligned(16))) T lds[2 * BUF];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wk = wave % WK, wc = wave / WK;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    const int frows = a.ft * a.fr;
    const int frGroups = (frows + FR - 1) / FR;
    const int g = by / frGroups;
    const int fr0 = (by % frGroups) * FR;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;

    typename Atom::acc acc[FR * S][FK][FC];
#pragma unroll
    for(int i = 0; i < FR * S; ++i)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FC; ++y)
                acc[i][x][y] = typename Atom::acc{};

    // Filter rows' (t, r): fixed per block.
    int ft[FR], frr[FR];
#pragma unroll
    for(int f = 0; f < FR; ++f)
    {
        ft[f] = (fr0 + f) / a.fr;
        frr[f] = (fr0 + f) - ft[f] * a.fr;
    }
    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    // Cursor: (n, od, oh, chunk), chunk fastest. Packed rows: (n, od, chunk), chunk the plane's
    // group of segRows rows, oh unused.
    constexpr int ROWS_HO = PACK != 0 ? 0 : 1;
    int n = 0, od = 0, oh = 0, chunk = 0;
    if(begin < end)
    {
        const long long row = begin / a.chunks;
        chunk = static_cast<int>(begin - row * a.chunks);
        const long long r2 = ROWS_HO ? row / a.ho : row;
        if constexpr(ROWS_HO != 0)
            oh = static_cast<int>(row % a.ho);
        od = static_cast<int>(r2 % a.dout);
        n = static_cast<int>(r2 / a.dout);
    }
    Tiles tiles;
    tiles.init(a, tid, g, k0, c0, kg, cg);
    auto issue = [&](bool live) {
        live = live && n < a.n;
        if constexpr(PACK != 0)
        {
            bool fOk[FR];
            int xBase[FR], ihOf[FR];
#pragma unroll
            for(int f = 0; f < FR; ++f)
            {
                const int id = od * a.sd + ft[f] * a.dd - a.pd;
                fOk[f] = fr0 + f < frows && id >= 0 && id < a.di;
                xBase[f] = id * xs2;
                ihOf[f] = frr[f] * a.dh - a.ph;
            }
            tiles.load(a, live ? n : 0, g, k0, c0, od, chunk * a.segRows, live, fOk, xBase, ihOf);
        }
        else
        {
            bool xOk[FR];
            int xRow[FR];
#pragma unroll
            for(int f = 0; f < FR; ++f)
            {
                const int id = od * a.sd + ft[f] * a.dd - a.pd;
                const int ih = oh * a.sh + frr[f] * a.dh - a.ph;
                xOk[f] = live && fr0 + f < frows && id >= 0 && id < a.di && ih >= 0 && ih < a.hi;
                xRow[f] = id * xs2 + ih * xs3;
            }
            tiles.load(a, live ? n : 0, chunk * BQ, live, od * ys2 + oh * ys3, xOk, xRow);
        }
    };
    auto advance = [&]() {
        if(++chunk == a.chunks)
        {
            chunk = 0;
            if(ROWS_HO == 0 || ++oh == a.ho)
            {
                oh = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++n;
                }
            }
        }
    };

    if(begin < end)
    {
        issue(true);
        tiles.template wait<0, 0>();
        tiles.store(a, lds, lds + Tiles::A_ELEMS, tid);
        __syncthreads();
    }
    const int kb = Atom::template kBase<KPL>(lane);
    int buf = 0;
    for(long long step = begin; step < end; ++step)
    {
        advance();
        issue(step + 1 < end);
        const T* As = lds + buf * BUF;
        const T* Bs = As + Tiles::A_ELEMS;
        typename Atom::frag dyf[FR][KK][FK];
#pragma unroll
        for(int kk = 0; kk < KK; ++kk)
#pragma unroll
            for(int x = 0; x < FK; ++x)
            {
                if constexpr(Tiles::TR)
                    dyf[0][kk][x] = k3FragTr<Atom, T, LDA>(As + wk * WTK + k3TrLane<Atom, LDA>(lane)
                                                           + kk * KA * LDA + x * DIM);
                else
                    dyf[0][kk][x] = k3FragAt<Atom, T, 0>(As + (wk * WTK + lane % DIM) * LDA + kb
                                                         + x * DIM * LDA + kk * KA);
#pragma unroll
                for(int f = 1; f < FR; ++f)
                    dyf[f][kk][x] = dyf[0][kk][x];
            }
        constexpr int TRB = Tiles::TR ? 1 : 0;
        if constexpr(Tiles::TR)
            k3Multiply<Atom, T, FR, S, SW, DW, FK, FC, BC, LDB, KK, 0, KK, TRB, Tiles::PH>(
                acc, dyf, Bs + wc * WTC + k3TrLane<Atom, LDB>(lane), SW * Tiles::PH * LDB);
        else
            k3Multiply<Atom, T, FR, S, SW, DW, FK, FC, BC, LDB, KK>(
                acc, dyf, Bs + (wc * WTC + lane % DIM) * LDB + kb, SW * BC * LDB);
        tiles.template wait<0, 0>();
        if(step + 1 < end)
        {
            T* nA = lds + (buf ^ 1) * BUF;
            tiles.store(a, nA, nA + Tiles::A_ELEMS, tid);
        }
        __syncthreads();
        buf ^= 1;
    }
    k3StorePartials<T, Atom, FR, S, FK, FC, WTK, WTC>(
        a, acc, g, fr0, frows, k0, c0, wk, wc, lane, bz);
}

// ---------------------------------------------------------------------------------------------
// Generic taps (loop 8), the coverage tier: every geometry field is a runtime argument (filter
// extents, strides, dilations, padding) and x and dy are read through their element strides,
// so one compiled config serves any filter and layout. A block owns a bk x bc channel tile and
// TB consecutive taps of the flattened (t, r, s) filter; per step (an output row's BQ-position
// chunk of one image) it stages the dy chunk once and, per tap, the x positions that tap reads
// (a strided gather), and multiplies each tap's x tile against the dy chunk into its own
// accumulators. Channels-first (LAYOUT 0): element units, a thread keeping one position.
// Channels-last (LAYOUT 1): units of VEC contiguous channels at one position, scattered into
// the channel rows. Two LDS buffers, one barrier per step. Admitted only where no specialized
// config is (the host's two-tier rule).
// Below.
template <class Atom, class T>
__device__ inline typename Atom::frag k3FragAtRt(const T* p, int off);
template <int TC>
__device__ inline void k3StoreFmaRun(float* o, int c, int cg, const float (&v)[TC]);
template <class E, int TC>
__device__ inline void k3StoreRun(E* o, int c, int cg, const float (&v)[TC]);

/// The generic loop's FMA form, in place of a matrix atom (exact fp32 where no matrix core takes
/// the element): each of (BK / TK) x (BC / TC) threads keeps TK output by TC input channels of
/// every tap of the block. The matrix fields are placeholders.
template <int TK_, int TC_>
struct K3FmaGeneric
{
    static constexpr int TK = TK_, TC = TC_;
    static constexpr int WAVE = 64, DIM = 1, KA = 1, KPL = 1;
    typedef float acc;
    typedef float frag;
};

/// TK and TC of the FMA form; 0 for a matrix atom.
template <class Atom>
struct K3GenericFma
{
    static constexpr int TK = 0, TC = 0;
};
template <int TK_, int TC_>
struct K3GenericFma<K3FmaGeneric<TK_, TC_>>
{
    static constexpr int TK = TK_, TC = TC_;
};

// WX > 0 (channels-last, unit width stride): the block's TB taps are consecutive width taps of
// one filter row, and one x window of BQ + WX positions per channel serves all of them, tap t
// read at offset t x dw (the host keeps (TB - 1) dw within WX).
template <class T,
          class Atom,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int TB,
          int WK,
          int WC,
          int VEC,
          int WX = 0,
          int XCD = 0>
__device__ K3_LOOP_INLINE void k3WrwGeneric(const K3Args& a)
{
    // The FMA form (K3FmaGeneric): (BK / TKF) x (BC / TCF) threads, no fragments.
    constexpr int TKF = K3GenericFma<Atom>::TK, TCF = K3GenericFma<Atom>::TC;
    constexpr bool FMA = TKF > 0;
    static_assert(!FMA || (BK % TKF == 0 && BC % TCF == 0 && WK == 1 && WC == 1),
                  "K3 generic FMA: whole thread tiles");
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = FMA ? (BK / (FMA ? TKF : 1)) * (BC / (FMA ? TCF : 1)) : WK * WC * WAVE;
    constexpr int DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int WTK = BK / WK, WTC = BC / WC;
    // (The FMA form keeps no fragments: one-element placeholders.)
    constexpr int FK = FMA ? 1 : WTK / DIM, FC = FMA ? 1 : WTC / DIM;
    constexpr int KK = FMA ? 1 : BQ / KA;
    static_assert(FMA || (FK * DIM * WK == BK && FC * DIM * WC == BC && BQ % KA == 0), "K3 tile");
    static_assert(THREADS % BQ == 0, "K3 generic: a thread keeps one position");
    constexpr int ROWS = THREADS / BQ; // channel rows a pass covers
    static_assert(BK % ROWS == 0 && BC % ROWS == 0, "K3 generic: whole channel rows a pass");
    constexpr int SKEW = 16 / static_cast<int>(sizeof(T));
    constexpr int LD = BQ + SKEW;
    constexpr bool WIN = WX > 0;
    static_assert(!WIN || (LAYOUT != 0 && WX % 8 == 0), "K3 generic window: channels-last");
    constexpr int WPOS = BQ + WX; // window positions
    constexpr int LDW = k3WinPitch(WPOS, static_cast<int>(sizeof(T)));
    // The FMA form keeps the tiles [position][channel] (a thread's channels one vector read):
    // dy rows of BK + SKEW, x rows of BC + SKEW, per tap or (window) per window position.
    constexpr int LDAT = BK + SKEW, LDBT = BC + SKEW;
    constexpr int A_ELEMS = FMA ? BQ * LDAT : BK * LD;
    constexpr int B_ELEMS = FMA ? (WIN ? WPOS : TB * BQ) * LDBT : WIN ? BC * LDW : TB * BC * LD;
    constexpr int BUF = A_ELEMS + B_ELEMS;
    constexpr int A_PASSES = BK / ROWS, B_PASSES = TB * BC / ROWS;
    static_assert(k3LdsBytes(8, 0, sizeof(T), BK, BC, BQ, TB, 1, 1, WX, WK, WC, TKF, TCF, 1, 0, 0)
                      == 2 * BUF * static_cast<long long>(sizeof(T)),
                  "K3 LDS arithmetic (ConvWrwGeom.h)");
    typedef typename K1Raw<sizeof(T)>::type RawE;

    __shared__ __attribute__((aligned(16))) T lds[2 * BUF];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wk = wave % WK, wc = wave / WK;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    const int taps = a.ft * a.fr * a.fs;
    // Window: groups of TB width taps within each filter row; else TB consecutive taps.
    const int sGroups = WIN ? (a.fs + TB - 1) / TB : 1;
    const int tapGroups = WIN ? a.ft * a.fr * sGroups : (taps + TB - 1) / TB;
    const int g = by / tapGroups;
    const int group = by % tapGroups;
    const int rowTap = WIN ? (group / sGroups) * a.fs : 0;
    const int tap0 = WIN ? rowTap + (group % sGroups) * TB : group * TB;
    const int tapLimit = WIN ? rowTap + a.fs : taps;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;

    // The block's taps: depth, height and width offsets of the input position they read.
    int tapD[TB], tapH[TB], tapW[TB];
    bool tapOk[TB];
#pragma unroll
    for(int t = 0; t < TB; ++t)
    {
        const int tap = tap0 + t;
        const int s = tap % a.fs, rt = tap / a.fs;
        const int r = rt % a.fr, ft = rt / a.fr;
        tapOk[t] = tap < tapLimit;
        tapD[t] = ft * a.dd - a.pd;
        tapH[t] = r * a.dh - a.ph;
        tapW[t] = s * a.dw - a.pw;
    }

    typename Atom::acc acc[TB][FK][FC];
#pragma unroll
    for(int t = 0; t < TB; ++t)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FC; ++y)
                acc[t][x][y] = typename Atom::acc{};
    // The FMA form's sums: TKF output by TCF input channels of each tap.
    constexpr int FTK = FMA ? TKF : 1, FTC = FMA ? TCF : 1;
    float facc[FMA ? TB : 1][FTK][FTC];
#pragma unroll
    for(int t = 0; t < (FMA ? TB : 1); ++t)
#pragma unroll
        for(int x = 0; x < FTK; ++x)
#pragma unroll
            for(int y = 0; y < FTC; ++y)
                facc[t][x][y] = 0.0f;
    const int ftc = tid % (BC / FTC), ftk = tid / (BC / FTC);

    // The thread's position in a chunk and first channel row; passes step ROWS rows.
    const int j = tid % BQ, row0 = tid / BQ;
    const long long ys0 = a.ys[0], ys1 = a.ys[1], ys2 = a.ys[2], ys3 = a.ys[3], ys4 = a.ys[4];
    const long long xs0 = a.xs[0], xs1 = a.xs[1], xs2 = a.xs[2], xs3 = a.xs[3], xs4 = a.xs[4];
    // Channels-last units: VEC channels at KW consecutive positions (a 32-bit LDS word of each
    // channel row), the channel vector fastest.
    constexpr bool CL = LAYOUT != 0;
    constexpr int KW = CL ? 4 / static_cast<int>(sizeof(T)) : 1;
    constexpr int KVN = BK / VEC, CVN = BC / VEC;
    constexpr int A_UNITS = KVN * (BQ / KW);
    constexpr int B_UNITS = WIN ? CVN * (WPOS / KW) : TB * CVN * (BQ / KW);
    constexpr int AU = (A_UNITS + THREADS - 1) / THREADS, BU = (B_UNITS + THREADS - 1) / THREADS;
    static_assert(!CL || (BK % VEC == 0 && BC % VEC == 0 && BQ % KW == 0),
                  "K3 generic: whole channel vectors and position words");
    typedef typename K1If<CL, typename K1Raw<16>::type, RawE>::type Unit;
    constexpr int NA = CL ? AU * KW : A_PASSES;
    constexpr int NB = CL ? BU * KW : B_PASSES;
    Unit ra[NA], rb[NB];
    // Channel e's KW positions of units v[0..KW): one LDS word.
    auto wordOf = [](const auto* v, int e) {
        if constexpr(sizeof(T) == 4)
            return static_cast<unsigned>(v[0][e]);
        else
        {
            const int sh = 16 * (e % 2);
            const unsigned lo = (static_cast<unsigned>(v[0][e / 2]) >> sh) & 0xffffu;
            const unsigned hi = (static_cast<unsigned>(v[1][e / 2]) >> sh) & 0xffffu;
            return lo | (hi << 16);
        }
    };
    int n = 0, od = 0, oh = 0, chunk = 0;
    if(begin < end)
    {
        const long long rowIndex = begin / a.chunks;
        chunk = static_cast<int>(begin - rowIndex * a.chunks);
        oh = static_cast<int>(rowIndex % a.ho);
        const long long r2 = rowIndex / a.ho;
        od = static_cast<int>(r2 % a.dout);
        n = static_cast<int>(r2 / a.dout);
    }
    auto tapOf = [&](int t, bool& tok, int& dD, int& dH, int& dW) {
        tok = false;
        dD = dH = dW = 0;
#pragma unroll
        for(int tt = 0; tt < TB; ++tt)
            if(tt == t)
            {
                tok = tapOk[tt];
                dD = tapD[tt];
                dH = tapH[tt];
                dW = tapW[tt];
            }
    };
    auto issue = [&](bool live) {
        live = live && n < a.n;
        // Image n's buffers: their batch strides bound them (32-bit offsets within). With
        // block-based buffers (CONV_K3_PB, images past the 32-bit range) each step's buffers
        // start at the first element the block can read in it (its first channel, the output
        // row's first reachable x row and column) and span the host's window (xImage, yImage).
        long long yOrg = 0, xOrg = 0;
        if constexpr(CONV_K3_PB != 0)
        {
            const int id0 = max(od * a.sd - a.pd, 0), ih0 = max(oh * a.sh - a.ph, 0);
            const int iw0 = max(chunk * BQ * a.sw - a.pw, 0);
            yOrg = (g * kg + k0) * ys1 + od * ys2 + oh * ys3 + chunk * BQ * ys4;
            xOrg = (g * cg + c0) * xs1 + id0 * xs2 + ih0 * xs3 + iw0 * xs4;
        }
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * ys0 + yOrg, a.yImage * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * xs0 + xOrg, a.xImage * sizeof(T));
        if constexpr(CL)
        {
#pragma unroll
            for(int p = 0; p < AU; ++p)
            {
                const int u = tid + p * THREADS;
                const int kv = u % KVN, qg = u / KVN;
                const int k = k0 + kv * VEC;
#pragma unroll
                for(int w = 0; w < KW; ++w)
                {
                    const int ow = chunk * BQ + qg * KW + w;
                    const bool ok = live && u < A_UNITS && ow < a.wo && k < kg;
                    const long long off
                        = (g * kg + k) * ys1 + od * ys2 + oh * ys3 + ow * ys4 - yOrg;
                    k3Load(ra[p * KW + w],
                           yr,
                           ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                }
            }
#pragma unroll
            for(int p = 0; p < BU; ++p)
            {
                const int u = tid + p * THREADS;
                if constexpr(WIN)
                {
                    // The window: positions chunk BQ + q, read at the first tap's offsets (unit
                    // width stride); every tap of the block reads within it.
                    const int cv = u % CVN, qg = u / CVN;
                    const int c = c0 + cv * VEC;
                    const int id = od * a.sd + tapD[0], ih = oh * a.sh + tapH[0];
                    const bool rowOk = live && u < B_UNITS && c < cg && id >= 0 && id < a.di
                                       && ih >= 0 && ih < a.hi;
#pragma unroll
                    for(int w = 0; w < KW; ++w)
                    {
                        const int iw = chunk * BQ + qg * KW + w + tapW[0];
                        const bool ok = rowOk && iw >= 0 && iw < a.wi;
                        const long long off
                            = (g * cg + c) * xs1 + id * xs2 + ih * xs3 + iw * xs4 - xOrg;
                        k3Load(rb[p * KW + w],
                               xr,
                               ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                    }
                    continue;
                }
                const int cv = u % CVN, rest = u / CVN;
                const int qg = rest % (BQ / KW), t = rest / (BQ / KW);
                bool tok;
                int dD, dH, dW;
                tapOf(t, tok, dD, dH, dW);
                const int c = c0 + cv * VEC;
                const int id = od * a.sd + dD, ih = oh * a.sh + dH;
                const bool rowOk = live && u < B_UNITS && tok && c < cg && id >= 0 && id < a.di
                                   && ih >= 0 && ih < a.hi;
#pragma unroll
                for(int w = 0; w < KW; ++w)
                {
                    const int ow = chunk * BQ + qg * KW + w;
                    const int iw = ow * a.sw + dW;
                    const bool ok = rowOk && ow < a.wo && iw >= 0 && iw < a.wi;
                    const long long off
                        = (g * cg + c) * xs1 + id * xs2 + ih * xs3 + iw * xs4 - xOrg;
                    k3Load(rb[p * KW + w],
                           xr,
                           ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
                }
            }
            return;
        }
        const int ow = chunk * BQ + j;
        const bool qOk = live && ow < a.wo;
        const long long yPos = od * ys2 + oh * ys3 + ow * ys4;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int k = k0 + row0 + p * ROWS;
            const bool ok = qOk && k < kg;
            const long long off = (g * kg + k) * ys1 + yPos - yOrg;
            k3Load(ra[p], yr, ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int rowAll = row0 + p * ROWS;
            const int t = rowAll / BC, c = c0 + rowAll % BC;
            bool tok;
            int dD, dH, dW;
            tapOf(t, tok, dD, dH, dW);
            const int id = od * a.sd + dD, ih = oh * a.sh + dH, iw = ow * a.sw + dW;
            const bool ok = qOk && tok && c < cg && id >= 0 && id < a.di && ih >= 0 && ih < a.hi
                            && iw >= 0 && iw < a.wi;
            const long long off = (g * cg + c) * xs1 + id * xs2 + ih * xs3 + iw * xs4 - xOrg;
            k3Load(rb[p], xr, ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
        }
    };
    auto store = [&](T* As, T* Bs) {
        if constexpr(CL)
        {
#pragma unroll
            for(int p = 0; p < AU; ++p)
            {
                const int u = tid + p * THREADS;
                if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                    continue;
                const int kv = u % KVN, qg = u / KVN;
                if constexpr(FMA)
                {
                    // [position][channel]: each loaded vector lands whole.
#pragma unroll
                    for(int w = 0; w < KW; ++w)
                        *reinterpret_cast<Unit*>(As + (qg * KW + w) * LDAT + kv * VEC)
                            = ra[p * KW + w];
                    continue;
                }
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    *reinterpret_cast<unsigned*>(As + (kv * VEC + e) * LD + qg * KW)
                        = wordOf(&ra[p * KW], e);
            }
#pragma unroll
            for(int p = 0; p < BU; ++p)
            {
                const int u = tid + p * THREADS;
                if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                    continue;
                if constexpr(WIN)
                {
                    const int cv = u % CVN, qg = u / CVN;
                    if constexpr(FMA)
                    {
#pragma unroll
                        for(int w = 0; w < KW; ++w)
                            *reinterpret_cast<Unit*>(Bs + (qg * KW + w) * LDBT + cv * VEC)
                                = rb[p * KW + w];
                        continue;
                    }
#pragma unroll
                    for(int e = 0; e < VEC; ++e)
                        *reinterpret_cast<unsigned*>(Bs + (cv * VEC + e) * LDW + qg * KW)
                            = wordOf(&rb[p * KW], e);
                    continue;
                }
                const int cv = u % CVN, rest = u / CVN;
                const int qg = rest % (BQ / KW), t = rest / (BQ / KW);
                if constexpr(FMA)
                {
#pragma unroll
                    for(int w = 0; w < KW; ++w)
                        *reinterpret_cast<Unit*>(Bs + (t * BQ + qg * KW + w) * LDBT + cv * VEC)
                            = rb[p * KW + w];
                    continue;
                }
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    *reinterpret_cast<unsigned*>(Bs + (t * BC + cv * VEC + e) * LD + qg * KW)
                        = wordOf(&rb[p * KW], e);
            }
        }
        else if constexpr(FMA)
        {
            // [position][channel]: position j's channel rows.
#pragma unroll
            for(int p = 0; p < A_PASSES; ++p)
                As[j * LDAT + row0 + p * ROWS] = __builtin_bit_cast(T, ra[p]);
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
                const int rowAll = row0 + p * ROWS;
                Bs[((rowAll / BC) * BQ + j) * LDBT + rowAll % BC] = __builtin_bit_cast(T, rb[p]);
            }
        }
        else
        {
#pragma unroll
            for(int p = 0; p < A_PASSES; ++p)
                As[(row0 + p * ROWS) * LD + j] = __builtin_bit_cast(T, ra[p]);
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
                Bs[(row0 + p * ROWS) * LD + j] = __builtin_bit_cast(T, rb[p]);
        }
    };
    auto advance = [&]() {
        if(++chunk == a.chunks)
        {
            chunk = 0;
            if(++oh == a.ho)
            {
                oh = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++n;
                }
            }
        }
    };
    auto waitLoads = [&]() {
        k3WaitLoads<0>();
#pragma unroll
        for(int i = 0; i < NA; ++i)
            k3Pin(ra[i]);
#pragma unroll
        for(int i = 0; i < NB; ++i)
            k3Pin(rb[i]);
    };

    if(begin < end)
    {
        issue(true);
        waitLoads();
        store(lds, lds + A_ELEMS);
        __syncthreads();
    }
    int kb = 0;
    if constexpr(!FMA)
        kb = Atom::template kBase<KPL>(lane);
    int buf = 0;
    for(long long step = begin; step < end; ++step)
    {
        advance();
        issue(step + 1 < end);
        const T* As = lds + buf * BUF;
        const T* Bs = As + A_ELEMS;
        typename Atom::frag dyf[1][KK][FK];
        if constexpr(!FMA)
        {
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    dyf[0][kk][x] = k3FragAt<Atom, T, 0>(As + (wk * WTK + lane % DIM) * LD + kb
                                                         + x * DIM * LD + kk * KA);
        }
        if constexpr(FMA)
        {
            // Per position: the thread's TKF dy values against each tap's TCF x values. Taps
            // past the block's (or row's) last are computed and not written.
            const T* aCol = As + ftk * FTK;
            const T* bCol = Bs + ftc * FTC;
#pragma unroll 2
            for(int q = 0; q < BQ; ++q)
            {
                float av[FTK];
#pragma unroll
                for(int x = 0; x < FTK; ++x)
                    av[x] = K1Elem<T>::load(aCol[q * LDAT + x]);
#pragma unroll
                for(int t = 0; t < TB; ++t)
                {
                    const T* bp
                        = bCol + (WIN ? q + t * a.dw : t * BQ + q) * static_cast<long>(LDBT);
                    float bv[FTC];
#pragma unroll
                    for(int y = 0; y < FTC; ++y)
                        bv[y] = K1Elem<T>::load(bp[y]);
#pragma unroll
                    for(int x = 0; x < FTK; ++x)
#pragma unroll
                        for(int y = 0; y < FTC; ++y)
                            facc[t][x][y] = __builtin_fmaf(av[x], bv[y], facc[t][x][y]);
                }
            }
        }
        else if constexpr(WIN)
        {
            // Tap t reads the window t dw positions on (a block-uniform runtime offset).
            const T* bRow = Bs + (wc * WTC + lane % DIM) * LDW + kb;
            k3ForEach<TB>([&](auto tc) {
                constexpr int t = decltype(tc)::value;
                if(tapOk[t])
                {
                    const int off = t * a.dw;
#pragma unroll
                    for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                        for(int y = 0; y < FC; ++y)
                        {
                            const typename Atom::frag bf
                                = k3FragAtRt<Atom, T>(bRow + y * DIM * LDW + kk * KA, off);
#pragma unroll
                            for(int x = 0; x < FK; ++x)
                                acc[t][x][y] = Atom::mma(dyf[0][kk][x], bf, acc[t][x][y]);
                        }
                }
            });
        }
        else
            k3ForEach<TB>([&](auto tc) {
                constexpr int t = decltype(tc)::value;
                typedef typename Atom::acc AccRow[1][FK][FC];
                k3Multiply<Atom, T, 1, 1, 1, 1, FK, FC, BC, LD, KK>(
                    *reinterpret_cast<AccRow*>(&acc[t]),
                    dyf,
                    Bs + t * BC * LD + (wc * WTC + lane % DIM) * LD + kb,
                    0);
            });
        waitLoads();
        if(step + 1 < end)
        {
            T* nA = lds + (buf ^ 1) * BUF;
            store(nA, nA + A_ELEMS);
        }
        __syncthreads();
        buf ^= 1;
    }
    if constexpr(FMA)
    {
        // The FMA form's sums: runs of TCF channels of output channel k for each tap.
        if constexpr(!CONV_K3_DWS)
        {
            const long long kc = static_cast<long long>(kg) * cg;
            float* out = a.partial + bz * a.partialElems + static_cast<long long>(g) * taps * kc;
#pragma unroll
            for(int t = 0; t < TB; ++t)
            {
                const int tap = tap0 + t;
                if(tap >= tapLimit)
                    continue;
#pragma unroll
                for(int x = 0; x < FTK; ++x)
                {
                    const int k = k0 + ftk * FTK + x;
                    if(k < kg)
                        k3StoreFmaRun<FTC>(out + tap * kc + static_cast<long long>(k) * cg,
                                           c0 + ftc * FTC,
                                           cg,
                                           facc[t][x]);
                }
            }
            return;
        }
        const K3Out out = k3OutOf<T>(a, g, taps, bz);
        k3WithOut<T>(a, out, [&](auto* base) {
            typedef typename K3PointeeOf<decltype(base)>::type E;
#pragma unroll
            for(int t = 0; t < TB; ++t)
            {
                const int tap = tap0 + t;
                if(tap >= tapLimit)
                    continue;
#pragma unroll
                for(int x = 0; x < FTK; ++x)
                {
                    const int k = k0 + ftk * FTK + x;
                    if(k < kg)
                        k3StoreRun<E, FTC>(
                            base + tap * out.ts + k * out.ks, c0 + ftc * FTC, cg, facc[t][x]);
                }
            }
        });
    }
    else
    {
        // The taps' sums: acc[t] holds tap tap0 + t, written as filter rows of one tap (S = 1).
        k3StorePartials<T, Atom, TB, 1, FK, FC, WTK, WTC>(
            a, acc, g, tap0, taps, k0, c0, wk, wc, lane, bz, 1, tapLimit);
    }
}

/// KPL consecutive elements of an LDS row starting OFF elements past p, OFF a per-lane runtime
/// value (p aligned to 4 bytes): dwords from the floor, shifted into place for an odd 16-bit
/// start.
template <class Atom, class T>
__device__ inline typename Atom::frag k3FragAtRt(const T* p, int off)
{
    typedef typename Atom::frag F;
    constexpr int KPL = Atom::KPL;
    if constexpr(sizeof(T) == 4)
    {
        F v;
#pragma unroll
        for(int i = 0; i < KPL; ++i)
            reinterpret_cast<unsigned*>(&v)[i] = reinterpret_cast<const unsigned*>(p + off)[i];
        return v;
    }
    else
    {
        constexpr int ND = KPL / 2;
        const unsigned* d = reinterpret_cast<const unsigned*>(p + (off & ~1));
        unsigned w[ND + 1];
#pragma unroll
        for(int i = 0; i <= ND; ++i)
            w[i] = d[i];
        const unsigned sh = (off & 1) * 2;
        unsigned v[ND];
#pragma unroll
        for(int i = 0; i < ND; ++i)
            v[i] = __builtin_amdgcn_alignbyte(w[i + 1], w[i], sh);
        return __builtin_bit_cast(F, v);
    }
}

/// The small-C rows' tiles from channels-last x and dy (stems stored N H W C): the LDS tiles of
/// K3Tiles<K3_NCHW> (so the small-C loop reads them unchanged and the LDS arithmetic is the
/// channels-first one), loaded from channels-last operands.
///
/// A (dy): a unit is KV positions by VEC output channels (16 bytes each, the output channels per
/// group whole vectors), transposed into As[k][q] by words, as K3Tiles<K3_NHWC> does. B (x): a
/// unit is one element (filter row, window position, channel), the channel fastest, so a wave
/// reads the window's run of C-channel pixels in order however few channels there are; it is
/// stored into its phase row Bs[(f SW + p % SW) BC + c][p / SW]. Elements outside the image,
/// the row or the group read zeros. The B unit's indices are recomputed from the thread and pass
/// (constant divisors), not kept in registers: there can be ~30 passes.
template <class T, int BK, int BC, int BQ, int FR, int S, int SW, int DW, int VEC, int THREADS>
struct K3SmallClTiles
{
    typedef K3Tiles<T, K3_NCHW, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS> Geo;
    typedef typename K1Raw<16>::type Raw;
    typedef typename K1Raw<sizeof(T)>::type RawE;
    static constexpr int KV = 4 / static_cast<int>(sizeof(T));
    static constexpr int LDA = Geo::LDA, LDB = Geo::LDB;
    static constexpr int A_ELEMS = Geo::A_ELEMS, B_ELEMS = Geo::B_ELEMS;
    /// Window positions staged per filter row: every phase row's PH positions.
    static constexpr int SPAN = SW * Geo::PH;
    static constexpr int A_UNITS = (BQ / KV) * (BK / VEC);
    static constexpr int B_UNITS = FR * SPAN * BC;
    static constexpr int A_PASSES = (A_UNITS + THREADS - 1) / THREADS;
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static_assert(VEC * sizeof(T) == 16 && BK % VEC == 0 && BQ % KV == 0,
                  "K3 small-C channels-last dy units");
    static_assert(A_PASSES <= 32 && B_PASSES <= 32, "K3 pass masks");

    Raw ra[A_PASSES * KV];
    RawE rb[B_PASSES];
    int aOff[A_PASSES], aLds[A_PASSES], aPos[A_PASSES];
    unsigned aOk;
    int gc0; ///< the block's first input channel, all groups

    __device__ void init(const K3Args& a, int tid, int g, int k0, int c0, int kg, int cg)
    {
        aOk = 0;
        gc0 = g * cg + c0;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            const int qp = u % (BQ / KV), kv = u / (BQ / KV);
            const int k = k0 + kv * VEC;
            aPos[p] = qp * KV;
            aOff[p] = aPos[p] * static_cast<int>(a.ys[4]) + g * kg + k;
            aLds[p] = (kv * VEC) * LDA + qp * KV;
            if(u < A_UNITS && k < kg)
                aOk |= 1u << p;
        }
    }

    /// B unit u: filter row f, window position pos, channel c of the tile.
    __device__ static void unitOf(int u, int& f, int& pos, int& c)
    {
        c = u % BC;
        const int rest = u / BC;
        pos = rest % SPAN;
        f = rest / SPAN;
    }

    /// The global loads of one step: the dy row (yOk, element offset yRow) from position q0,
    /// and per filter row f the x row (xOk[f], xRow[f]) window from position q0 SW - pw.
    template <int P = 0>
    __device__ void load(const K3Args& a,
                         int n,
                         int q0,
                         bool yOk,
                         int yRow,
                         const bool (&xOk)[FR],
                         const int (&xRow)[FR])
    {
        static_assert(P == 0, "K3 small-C channels-last tiles hold one step");
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * a.yImage, a.yImage * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * a.xImage, a.xImage * sizeof(T));
        const int ys4 = static_cast<int>(a.ys[4]), xs4 = static_cast<int>(a.xs[4]);
        const int cg = a.c / a.groups;
        const int c0 = gc0 % cg;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const bool ok = yOk && ((aOk >> p) & 1u);
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const bool okq = ok && q0 + aPos[p] + j < a.wo;
                const int off = aOff[p] + yRow + (q0 + j) * ys4;
                k3Load(ra[p * KV + j],
                       yr,
                       okq ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
            }
        }
        const int iw0 = q0 * SW - a.pw;
        const int tid = threadIdx.x;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            int f, pos, c;
            unitOf(u, f, pos, c);
            bool rowOk = false;
            int row = 0;
#pragma unroll
            for(int ff = 0; ff < FR; ++ff)
                if(ff == f)
                {
                    rowOk = xOk[ff];
                    row = xRow[ff];
                }
            const int iw = iw0 + pos;
            const bool ok = u < B_UNITS && rowOk && c0 + c < cg && iw >= 0 && iw < a.wi;
            const int off = row + iw * xs4 + gc0 + c;
            k3Load(rb[p], xr, ok ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
        }
    }

    template <int P, int N>
    __device__ void wait()
    {
        k3WaitLoads<N>();
#pragma unroll
        for(int i = 0; i < A_PASSES * KV; ++i)
            k3Pin(ra[i]);
#pragma unroll
        for(int i = 0; i < B_PASSES; ++i)
            k3Pin(rb[i]);
    }

    template <int P = 0>
    __device__ void store(const K3Args& /*a*/, T* As, T* Bs, int tid) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                continue;
            T* dst = As + aLds[p];
            // KV loaded vectors (one per position) of VEC channels become VEC words of KV
            // positions, one per channel row (by value: see K3Tiles::storeWords).
#pragma unroll
            for(int i = 0; i < VEC; ++i)
            {
                if constexpr(KV == 1)
                {
                    *reinterpret_cast<unsigned*>(dst + i * LDA) = ra[p][i];
                }
                else
                {
                    const int sh = 16 * (i % 2);
                    const unsigned lo = (ra[p * KV][i / 2] >> sh) & 0xffffu;
                    const unsigned hi = (ra[p * KV + 1][i / 2] >> sh) & 0xffffu;
                    *reinterpret_cast<unsigned*>(dst + i * LDA) = lo | (hi << 16);
                }
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            int f, pos, c;
            unitOf(u, f, pos, c);
            Bs[((f * SW + pos % SW) * BC + c) * LDB + pos / SW] = __builtin_bit_cast(T, rb[p]);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Small-C output rows (stems: few input channels). The output-rows kernel with the GEMM's N
// axis the (input channel, filter-width tap) pairs instead of the input channels: lane column n
// of a fragment is channel n / S, tap n % S, read from the channel's window at the tap's stride
// phase and offset, so BC S columns fill whole fragments where BC channels alone would leave
// most of them idle. Waves split the output channels only (WC = 1). Channels-first operands
// load through K3Tiles; channels-last ones (LAYOUT K3_NHWC) through K3SmallClTiles, into the
// same LDS tiles.
template <class T,
          class Atom,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int FR,
          int S,
          int SW,
          int DW,
          int WK,
          int VEC,
          int XCD = 0>
__device__ K3_LOOP_INLINE void k3WrwRowsSmallC(const K3Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WK * WAVE;
    constexpr int DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int WTK = BK / WK;
    constexpr int FK = WTK / DIM;
    constexpr int FN = (BC * S + DIM - 1) / DIM; // (channel, tap) fragments
    constexpr int KK = BQ / KA;
    static_assert(FK * DIM * WK == BK && BQ % KA == 0, "K3 small-C tile");
    static_assert(LAYOUT == K3_NCHW || LAYOUT == K3_NHWC, "K3 small-C layouts");
    typedef typename K1If<LAYOUT == K3_NCHW,
                          K3Tiles<T, K3_NCHW, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS>,
                          K3SmallClTiles<T, BK, BC, BQ, FR, S, SW, DW, VEC, THREADS>>::type Tiles;
    static_assert(
        k3LdsBytes(6, K3_NCHW, sizeof(T), BK, BC, BQ, FR, S, SW, DW, WK, 1, 0, 0, VEC, 0, 0)
            == 2 * (Tiles::A_ELEMS + Tiles::B_ELEMS) * sizeof(T),
        "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int LDA = Tiles::LDA, LDB = Tiles::LDB;
    constexpr int BUF = Tiles::A_ELEMS + Tiles::B_ELEMS;
    static_assert(LDB % 2 == 0, "K3 small-C rows dword aligned");

    __shared__ __attribute__((aligned(16))) T lds[2 * BUF];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wk = tid / WAVE;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    const int frows = a.ft * a.fr;
    const int frGroups = (frows + FR - 1) / FR;
    const int g = by / frGroups;
    const int fr0 = (by % frGroups) * FR;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;

    typename Atom::acc acc[FR][FK][FN];
#pragma unroll
    for(int f = 0; f < FR; ++f)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FN; ++y)
                acc[f][x][y] = typename Atom::acc{};

    // The lane's column per fragment: channel, tap, its phase row and offset in the B tile.
    int nOff[FN];
    bool nOk[FN];
#pragma unroll
    for(int y = 0; y < FN; ++y)
    {
        const int n = y * DIM + lane % DIM;
        const int c = n / S, s = n % S;
        nOk[y] = c < BC;
        const int ph = (s * DW) % SW, off = (s * DW) / SW;
        nOff[y] = nOk[y] ? (ph * BC + c) * LDB + off : 0;
    }
    int ft[FR], frr[FR];
#pragma unroll
    for(int f = 0; f < FR; ++f)
    {
        ft[f] = (fr0 + f) / a.fr;
        frr[f] = (fr0 + f) - ft[f] * a.fr;
    }
    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    int n = 0, od = 0, oh = 0, chunk = 0;
    if(begin < end)
    {
        const long long row = begin / a.chunks;
        chunk = static_cast<int>(begin - row * a.chunks);
        oh = static_cast<int>(row % a.ho);
        const long long r2 = row / a.ho;
        od = static_cast<int>(r2 % a.dout);
        n = static_cast<int>(r2 / a.dout);
    }
    Tiles tiles;
    tiles.init(a, tid, g, k0, c0, kg, cg);
    auto issue = [&](bool live) {
        live = live && n < a.n;
        bool xOk[FR];
        int xRow[FR];
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            const int id = od * a.sd + ft[f] * a.dd - a.pd;
            const int ih = oh * a.sh + frr[f] * a.dh - a.ph;
            xOk[f] = live && fr0 + f < frows && id >= 0 && id < a.di && ih >= 0 && ih < a.hi;
            xRow[f] = id * xs2 + ih * xs3;
        }
        tiles.load(a, live ? n : 0, chunk * BQ, live, od * ys2 + oh * ys3, xOk, xRow);
    };
    auto advance = [&]() {
        if(++chunk == a.chunks)
        {
            chunk = 0;
            if(++oh == a.ho)
            {
                oh = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++n;
                }
            }
        }
    };

    if(begin < end)
    {
        issue(true);
        tiles.template wait<0, 0>();
        tiles.store(a, lds, lds + Tiles::A_ELEMS, tid);
        __syncthreads();
    }
    const int kb = Atom::template kBase<KPL>(lane);
    int buf = 0;
    for(long long step = begin; step < end; ++step)
    {
        advance();
        issue(step + 1 < end);
        const T* As = lds + buf * BUF;
        const T* Bs = As + Tiles::A_ELEMS;
#pragma unroll
        for(int kk = 0; kk < KK; ++kk)
        {
            typename Atom::frag dyf[FK];
#pragma unroll
            for(int x = 0; x < FK; ++x)
                dyf[x] = k3FragAt<Atom, T, 0>(As + (wk * WTK + lane % DIM) * LDA + kb
                                              + x * DIM * LDA + kk * KA);
#pragma unroll
            for(int f = 0; f < FR; ++f)
#pragma unroll
                for(int y = 0; y < FN; ++y)
                {
                    typename Atom::frag bf
                        = k3FragAtRt<Atom, T>(Bs + f * SW * BC * LDB + kk * KA + kb, nOff[y]);
                    bf = nOk[y] ? bf : typename Atom::frag{};
#pragma unroll
                    for(int x = 0; x < FK; ++x)
                        acc[f][x][y] = Atom::mma(dyf[x], bf, acc[f][x][y]);
                }
        }
        tiles.template wait<0, 0>();
        if(step + 1 < end)
        {
            T* nA = lds + (buf ^ 1) * BUF;
            tiles.store(a, nA, nA + Tiles::A_ELEMS, tid);
        }
        __syncthreads();
        buf ^= 1;
    }

    // Partials [split][g][tap][k][c] (or dw, k3OutOf): column n is channel c0 + n / S, tap row
    // fr0 + f, tap s.
    if constexpr(!CONV_K3_DWS)
    {
        const long long kc = static_cast<long long>(kg) * cg;
        float* out = a.partial + bz * a.partialElems + static_cast<long long>(g) * frows * S * kc;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            if(fr0 + f >= frows)
                continue;
#pragma unroll
            for(int y = 0; y < FN; ++y)
            {
                const int nn = y * DIM + lane % DIM;
                const int c = c0 + nn / S, s = nn % S;
                if(nn / S >= BC || c >= cg)
                    continue;
#pragma unroll
                for(int x = 0; x < FK; ++x)
#pragma unroll
                    for(int e = 0; e < Atom::ACC; ++e)
                    {
                        const int k = k0 + wk * WTK + x * DIM + Atom::dRow(lane, e);
                        if(k < kg)
                            out[static_cast<long long>((fr0 + f) * S + s) * kc
                                + static_cast<long long>(k) * cg + c]
                                = acc[f][x][y][e];
                    }
            }
        }
        return;
    }
    const K3Out out = k3OutOf<T>(a, g, frows * S, bz);
    k3WithOut<T>(a, out, [&](auto* base) {
        typedef typename K3PointeeOf<decltype(base)>::type E;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            if(fr0 + f >= frows)
                continue;
#pragma unroll
            for(int y = 0; y < FN; ++y)
            {
                const int nn = y * DIM + lane % DIM;
                const int c = c0 + nn / S, s = nn % S;
                if(nn / S >= BC || c >= cg)
                    continue;
                E* o = base + static_cast<long long>((fr0 + f) * S + s) * out.ts + c;
#pragma unroll
                for(int x = 0; x < FK; ++x)
#pragma unroll
                    for(int e = 0; e < Atom::ACC; ++e)
                    {
                        const int k = k0 + wk * WTK + x * DIM + Atom::dRow(lane, e);
                        if(k < kg)
                            o[k * out.ks] = K1Elem<E>::store(acc[f][x][y][e]);
                    }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Row-ring kernel (height stride 1). A block owns BK output channels, BC input channels, one
// depth tap t and all R x S taps of it. The reduction runs along tracks (image, output depth,
// height class rho mod dh, width chunk): step m of a track stages input row
// ih = rho - ph + m dh and output row oh = rho + m dh, and tap row r pairs the input row with
// output row oh - r dh, whose fragments sit in a register ring of the last R dy rows. A track
// has trackLen = ceil(ho / dh) + R - 1 steps; a split that starts inside a track first replays
// the R - 1 rows before it with its multiplies masked. Grid: (k tiles x c tiles, groups x ft,
// splits). Two LDS buffers, one barrier per step.
template <class T,
          class Atom,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int R,
          int S,
          int SW,
          int DW,
          int WK,
          int WC,
          int VEC,
          int XCD = 0,
          int ALIGN = 0>
__device__ K3_LOOP_INLINE void k3WrwRing(const K3Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WK * WC * WAVE;
    constexpr int DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int WTK = BK / WK, WTC = BC / WC;
    constexpr int FK = WTK / DIM, FC = WTC / DIM;
    constexpr int KK = BQ / KA;
    static_assert(FK * DIM * WK == BK && FC * DIM * WC == BC && BQ % KA == 0, "K3 tile");
    typedef K3Tiles<T, LAYOUT, BK, BC, BQ, 1, S, SW, DW, VEC, THREADS, 1, ALIGN> Tiles;
    static_assert(
        k3LdsBytes(1, LAYOUT, sizeof(T), BK, BC, BQ, R, S, SW, DW, WK, WC, 0, 0, VEC, ALIGN, 0)
            == 2 * (Tiles::A_ELEMS + Tiles::B_ELEMS) * sizeof(T),
        "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int LDA = Tiles::LDA, LDB = Tiles::LDB;
    constexpr int BUF = Tiles::A_ELEMS + Tiles::B_ELEMS;

    __shared__ __attribute__((aligned(16))) T lds[2 * BUF];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wk = wave % WK, wc = wave / WK;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    // Height stride sh > 1 (unit height dilation): grid rows (g, t, phase p). Phase p takes the
    // x rows ih = sh j + p and the filter rows r0, r0 + sh, ... (r0 = (p + ph) mod sh) as a
    // stride-1 ring: step m pairs x row sh (m - q0) + p with dy row m (ring position i holds
    // dy row m - i, filter row r0 + sh i), q0 = (p + ph - r0) / sh.
    const int sh = a.sh;
    const int phases = sh > 1 ? sh : 1;
    const int p = static_cast<int>(by % phases);
    const int g = static_cast<int>(by / phases) / a.ft;
    const int t = static_cast<int>(by / phases) % a.ft;
    const int r0 = sh > 1 ? (p + a.ph) % sh : 0;
    const int q0 = sh > 1 ? (p + a.ph - r0) / sh : 0;
    // Ring positions whose filter row exists (all, at height stride 1).
    unsigned phaseRows = 0;
#pragma unroll
    for(int i = 0; i < R; ++i)
        phaseRows |= (r0 + phases * i < a.fr) ? 1u << i : 0u;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;
    const int M = a.trackLen;

    typename Atom::acc acc[R * S][FK][FC];
#pragma unroll
    for(int i = 0; i < R * S; ++i)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FC; ++y)
                acc[i][x][y] = typename Atom::acc{};
    typename Atom::frag ring[R][KK][FK];
#pragma unroll
    for(int r = 0; r < R; ++r)
#pragma unroll
        for(int kk = 0; kk < KK; ++kk)
#pragma unroll
            for(int x = 0; x < FK; ++x)
                ring[r][kk][x] = typename Atom::frag{};

    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    // Cursor: (n, od, rho, chunk, m), m fastest. Replay up to R - 1 rows of the track before
    // `begin` (multiplies masked).
    long long it = begin;
    int n = 0, od = 0, rho = 0, chunk = 0, m = 0;
    if(begin < end)
    {
        const long long track0 = begin / M;
        const int m0 = static_cast<int>(begin - track0 * M);
        const int back = m0 < R - 1 ? m0 : R - 1;
        it = begin - back;
        m = m0 - back;
        chunk = static_cast<int>(track0 % a.chunks);
        const long long t2 = track0 / a.chunks;
        rho = static_cast<int>(t2 % a.dh);
        const long long t3 = t2 / a.dh;
        od = static_cast<int>(t3 % a.dout);
        n = static_cast<int>(t3 / a.dout);
    }
    Tiles tiles;
    tiles.init(a, tid, g, k0, c0, kg, cg);
    // Loads the cursor's step into register set P.
    // Loads the cursor's step into register set P; a step past the split loads nothing
    // (out-of-range offsets), so every step issues the same loads.
    auto issue = [&](auto pc, bool live, bool math) {
        constexpr int P = decltype(pc)::value;
        const int oh = rho + m * a.dh;
        const int id = od * a.sd + t * a.dd - a.pd;
        const int ih = sh > 1 ? sh * (m - q0) + p : rho - a.ph + m * a.dh;
        const bool xOk[1]
            = {live && math && id >= 0 && id < a.di && ih >= 0 && ih < a.hi && n < a.n};
        const int xRow[1] = {id * xs2 + ih * xs3};
        tiles.template load<P>(a,
                               n < a.n ? n : 0,
                               chunk * BQ,
                               live && oh < a.ho && n < a.n,
                               od * ys2 + oh * ys3,
                               xOk,
                               xRow);
    };
    auto advance = [&]() {
        if(++m == M)
        {
            m = 0;
            if(++chunk == a.chunks)
            {
                chunk = 0;
                if(++rho == a.dh)
                {
                    rho = 0;
                    if(++od == a.dout)
                    {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    };

    // Pipeline: LDS buffer j % 2 holds step j; the registers carry step j + 1, loaded while step j
    // multiplies.
    int mc = m; // the computed step's place in its track
    if(begin < end)
    {
        issue(K3Int<0>{}, true, it >= begin);
        tiles.template wait<0, 0>();
        tiles.template store<0>(a, lds, lds + Tiles::A_ELEMS, tid);
        __syncthreads();
    }
    const int kb = Atom::template kBase<KPL>(lane);
    constexpr int TRB = Tiles::TR ? 1 : 0;
    auto body = [&](auto pc) {
        constexpr int J = decltype(pc)::value; // step parity
        // A new track starts with an empty ring; else the ring moves down one row. The cursor
        // is still on the computed step: its tap rows with a dy row in the image.
        const bool trackStart = mc == 0;
        const unsigned rows
            = k3RowsOf(m, rho < a.ho ? (a.ho - rho + a.dh - 1) / a.dh : 0, R) & phaseRows;
        // A step whose x row lies in the padding only moves the ring: its window is zeros
        // (a third of the steps of a 3-row filter at dilation 36 on 128 rows).
        const int ihC = sh > 1 ? sh * (m - q0) + p : rho - a.ph + m * a.dh;
        const int idC = od * a.sd + t * a.dd - a.pd;
        const bool xIn = idC >= 0 && idC < a.di && ihC >= 0 && ihC < a.hi;
        mc = mc + 1 == M ? 0 : mc + 1;
        advance();
        issue(K3Int<0>{}, it + 1 < end, it + 1 >= begin);
        // The loads above stay ahead of the multiplies.
        __builtin_amdgcn_sched_barrier(0);
        const T* As = lds + J * BUF;
        const T* Bs = As + Tiles::A_ELEMS;
#pragma unroll
        for(int r = R - 1; r > 0; --r)
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    ring[r][kk][x] = trackStart ? typename Atom::frag{} : ring[r - 1][kk][x];
        if constexpr(Tiles::TR)
        {
            const T* aBase = As + wk * WTK + k3TrLane<Atom, LDA>(lane);
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    ring[0][kk][x] = k3FragTr<Atom, T, LDA>(aBase + kk * KA * LDA + x * DIM);
        }
        else
        {
            const T* aBase = As + (wk * WTK + lane % DIM) * LDA + kb;
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    ring[0][kk][x] = k3FragAt<Atom, T, 0>(aBase + x * DIM * LDA + kk * KA);
        }
        // Tap row r: the window against dy row ring[r]; acc index r S + s. A replayed step
        // (before the split's first) only fills the ring. The branch also keeps the
        // multiplies' register live ranges apart from the loads' and stores'.
        const T* bBase = Tiles::TR ? Bs + wc * WTC + k3TrLane<Atom, LDB>(lane)
                                   : Bs + (wc * WTC + lane % DIM) * LDB + kb;
        if(it >= begin && xIn)
            k3MultiplyRows<Atom, T, R, S, SW, DW, FK, FC, BC, LDB, KK, TRB, Tiles::PH>(
                acc, ring, bBase, rows);
        __builtin_amdgcn_sched_barrier(0);
        tiles.template wait<0, 0>();
        {
            T* nA = lds + (J ^ 1) * BUF;
            tiles.template store<0>(a, nA, nA + Tiles::A_ELEMS, tid);
        }
        __syncthreads();
        ++it;
    };
    while(it < end)
    {
        body(K3Int<0>{});
        if(it >= end)
            break;
        body(K3Int<1>{});
    }
    k3WaitLoads<0>();
    // Filter rows of this phase: t fr + r0 + phases i (the ring's rows of filter depth t).
    k3StorePartials<T, Atom, R, S, FK, FC, WTK, WTC>(
        a, acc, g, t * a.fr + r0, a.ft * a.fr, k0, c0, wk, wc, lane, bz, phases, (t + 1) * a.fr);
}

/// An FMA thread's run of TC (a multiple of 4) input channels' partials from channel c of the
/// row o (o[c]): 16-byte stores when the run is whole within cg and cg is a whole number of
/// quads (so the run is 16-byte aligned), else element by element up to cg. Lanes hold
/// consecutive runs, so a wave writes one contiguous stretch (single-float stores 16 bytes
/// apart made FCN-32s's conv kernel ~5x slower on gfx906).
template <int TC>
__device__ inline void k3StoreFmaRun(float* o, int c, int cg, const float (&v)[TC])
{
    typedef float f4 __attribute__((ext_vector_type(4)));
    static_assert(TC % 4 == 0, "K3 FMA runs are whole quads");
    if(cg % 4 == 0 && c + TC <= cg)
    {
#pragma unroll
        for(int y = 0; y < TC; y += 4)
            *reinterpret_cast<f4*>(o + c + y) = f4{v[y], v[y + 1], v[y + 2], v[y + 3]};
        return;
    }
#pragma unroll
    for(int y = 0; y < TC; ++y)
        if(c + y < cg)
            o[c + y] = v[y];
}

/// k3StoreFmaRun's run under CONV_K3_DWS, in E (fp32 partials or dw): the four-element stores
/// also need the run's start aligned to them (dw is the caller's buffer).
template <class E, int TC>
__device__ inline void k3StoreRun(E* o, int c, int cg, const float (&v)[TC])
{
    typedef E e4 __attribute__((ext_vector_type(4)));
    static_assert(TC % 4 == 0, "K3 FMA runs are whole quads");
    if(cg % 4 == 0 && c + TC <= cg && reinterpret_cast<unsigned long long>(o + c) % sizeof(e4) == 0)
    {
#pragma unroll
        for(int y = 0; y < TC; y += 4)
            *reinterpret_cast<e4*>(o + c + y) = e4{K1Elem<E>::store(v[y]),
                                                   K1Elem<E>::store(v[y + 1]),
                                                   K1Elem<E>::store(v[y + 2]),
                                                   K1Elem<E>::store(v[y + 3])};
        return;
    }
#pragma unroll
    for(int y = 0; y < TC; ++y)
        if(c + y < cg)
            o[c + y] = K1Elem<E>::store(v[y]);
}

// ---------------------------------------------------------------------------------------------
// FMA row-ring kernel (exact fp32 where no fp32 matrix core exists: RDNA). The row-ring loop of
// k3WrwRing with vector-ALU outer products: LDS tiles are [position][channel] rows, so a
// thread's TK output and TC input channels at one position are one vector read each; the
// last R dy rows stay in an LDS ring (R + 1 slots, one being filled), the x window in two
// buffers. A thread owns TK x TC channels of every R x S tap. Channels-first loads take VEC
// positions of four channels and store them transposed; channels-last loads store as they are.
// Global units start at multiples of VEC within the row (the launcher keeps rows a whole number
// of VEC), so a unit is wholly in or out of its row; x positions go to window rows
// position - iw0, those before the window dropped.
template <class T,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int R,
          int S,
          int DW,
          int TK,
          int TC,
          int VEC>
struct K3FmaTiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int THREADS = (BK / TK) * (BC / TC);
    static constexpr int WIN = BQ + (S - 1) * DW;
    static constexpr int NBV = (WIN + VEC - 1) / VEC + 1; // aligned units over the window
    static constexpr int PAD = 4;
    static constexpr int LDA = BK + PAD, LDB = BC + PAD;
    static constexpr int ROWS_B = NBV * VEC; // window rows (from iwA)
    static constexpr int A_ELEMS = BQ * LDA; // one dy row slot
    static constexpr int B_ELEMS = ROWS_B * LDB;
    static constexpr int CH = 4; // channels per channels-first unit
    // Units: channels-first (channel quad, position vector), channels-last (position, channel
    // vector).
    static constexpr int A_UNITS = LAYOUT == K3_NCHW ? (BK / CH) * (BQ / VEC) : BQ * (BK / VEC);
    static constexpr int B_UNITS = LAYOUT == K3_NCHW ? (BC / CH) * NBV : (NBV * VEC) * (BC / VEC);
    static constexpr int A_PASSES = (A_UNITS + THREADS - 1) / THREADS;
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static constexpr int PER = LAYOUT == K3_NCHW ? CH : 1; // loads per unit
    static_assert(BQ % VEC == 0 && BK % CH == 0 && BC % CH == 0 && BK % VEC == 0 && BC % VEC == 0,
                  "K3 FMA tiles");

    Raw ra[A_PASSES * PER], rb[B_PASSES * PER];
    int aOff[A_PASSES], aPos[A_PASSES], aCh[A_PASSES];
    int bOff[B_PASSES], bPos[B_PASSES], bCh[B_PASSES];
    unsigned aOk, bOk;
    int iw0s, iwAs, q0s;

    __device__ void init(const K3Args& a, int tid, int g, int k0, int c0, int kg, int cg)
    {
        aOk = bOk = 0;
        const int ys1 = static_cast<int>(a.ys[1]), ys4 = static_cast<int>(a.ys[4]);
        const int xs1 = static_cast<int>(a.xs[1]), xs4 = static_cast<int>(a.xs[4]);
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(LAYOUT == K3_NCHW)
            {
                const int qv = u % (BQ / VEC), kq = u / (BQ / VEC);
                aCh[p] = kq * CH;
                aPos[p] = qv * VEC;
                aOff[p] = (g * kg + k0 + aCh[p]) * ys1 + aPos[p];
            }
            else
            {
                const int kv = u % (BK / VEC), q = u / (BK / VEC);
                aCh[p] = kv * VEC;
                aPos[p] = q;
                aOff[p] = q * ys4 + g * kg + k0 + aCh[p];
            }
            if(u < A_UNITS && k0 + aCh[p] < kg)
                aOk |= 1u << p;
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(LAYOUT == K3_NCHW)
            {
                const int v = u % NBV, cq = u / NBV;
                bCh[p] = cq * CH;
                bPos[p] = v * VEC;
                bOff[p] = (g * cg + c0 + bCh[p]) * xs1 + bPos[p];
            }
            else
            {
                const int cv = u % (BC / VEC), r = u / (BC / VEC);
                bCh[p] = cv * VEC;
                bPos[p] = r;
                bOff[p] = r * xs4 + g * cg + c0 + bCh[p];
            }
            if(u < B_UNITS && c0 + bCh[p] < cg)
                bOk |= 1u << p;
        }
    }

    /// Issues one step's loads: dy row (yOk, yRow) chunk from q0, x row (xOk, xRow) window.
    __device__ void load(const K3Args& a,
                         int n,
                         int q0,
                         bool yOk,
                         int yRow,
                         bool xOk,
                         int xRow,
                         int kg,
                         int cg,
                         int k0,
                         int c0)
    {
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * a.yImage, a.yImage * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * a.xImage, a.xImage * sizeof(T));
        const int ys1 = static_cast<int>(a.ys[1]), ys4 = static_cast<int>(a.ys[4]);
        const int xs1 = static_cast<int>(a.xs[1]), xs4 = static_cast<int>(a.xs[4]);
        const int iw0 = q0 - a.pw;
        const int iwA = iw0 >= 0 ? iw0 / VEC * VEC : -((-iw0 + VEC - 1) / VEC) * VEC;
        iw0s = iw0;
        iwAs = iwA;
        q0s = q0;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int q = q0 + aPos[p];
            const bool ok = yOk && ((aOk >> p) & 1u) && q < a.wo;
#pragma unroll
            for(int i = 0; i < PER; ++i)
            {
                const int off = aOff[p] + yRow + (LAYOUT == K3_NCHW ? q0 + i * ys1 : q0 * ys4);
                const bool oki = ok && (LAYOUT != K3_NCHW || k0 + aCh[p] + i < kg);
                k3Load(ra[p * PER + i],
                       yr,
                       oki ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int iw = iwA + bPos[p];
            const bool ok = xOk && ((bOk >> p) & 1u) && iw >= 0 && iw < a.wi;
#pragma unroll
            for(int i = 0; i < PER; ++i)
            {
                const int off = bOff[p] + xRow + (LAYOUT == K3_NCHW ? iwA + i * xs1 : iwA * xs4);
                const bool oki = ok && (LAYOUT != K3_NCHW || c0 + bCh[p] + i < cg);
                k3Load(rb[p * PER + i],
                       xr,
                       oki ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
            }
        }
    }

    /// Stores the loaded step: dy into ring slot As, x window into Bs (row position - iw0).
    __device__ void store(T* As, T* Bs, int tid)
    {
        k3WaitLoads<0>();
        const int shift = iwAs - iw0s; // <= 0
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                continue;
            if(LAYOUT == K3_NCHW)
            {
                // CH channels x VEC positions -> VEC rows of CH channels.
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                {
                    T w[CH];
#pragma unroll
                    for(int i = 0; i < CH; ++i)
                        w[i] = reinterpret_cast<const T*>(&ra[p * PER + i])[e];
                    typedef typename K1Raw<CH * sizeof(T)>::type W;
                    *reinterpret_cast<W*>(As + (aPos[p] + e) * LDA + aCh[p])
                        = __builtin_bit_cast(W, w);
                }
            }
            else
            {
                *reinterpret_cast<Raw*>(As + aPos[p] * LDA + aCh[p]) = ra[p];
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            if(LAYOUT == K3_NCHW)
            {
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                {
                    const int row = bPos[p] + e + shift;
                    if(row < 0 || row >= ROWS_B)
                        continue;
                    T w[CH];
#pragma unroll
                    for(int i = 0; i < CH; ++i)
                        w[i] = reinterpret_cast<const T*>(&rb[p * PER + i])[e];
                    typedef typename K1Raw<CH * sizeof(T)>::type W;
                    *reinterpret_cast<W*>(Bs + row * LDB + bCh[p]) = __builtin_bit_cast(W, w);
                }
            }
            else
            {
                const int row = bPos[p] + shift;
                if(row >= 0 && row < ROWS_B)
                    *reinterpret_cast<Raw*>(Bs + row * LDB + bCh[p]) = rb[p];
            }
        }
    }
};

template <class T,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int R,
          int S,
          int DW,
          int TK,
          int TC,
          int VEC>
__device__ K3_LOOP_INLINE void k3WrwRingFma(const K3Args& a)
{
    typedef K3FmaTiles<T, LAYOUT, BK, BC, BQ, R, S, DW, TK, TC, VEC> Tiles;
    static_assert(k3LdsBytes(3, LAYOUT, sizeof(T), BK, BC, BQ, R, S, 1, DW, 0, 0, TK, TC, VEC, 0, 0)
                      == ((R + 1) * Tiles::A_ELEMS + 2 * Tiles::B_ELEMS) * sizeof(T),
                  "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int THREADS = Tiles::THREADS;
    constexpr int LDA = Tiles::LDA, LDB = Tiles::LDB;
    constexpr int SLOTS = R + 1;
    static_assert(TK % 4 == 0 && TC % 4 == 0, "K3 FMA thread tile");
    __shared__ __attribute__((aligned(16))) T ldsA[SLOTS * Tiles::A_ELEMS];
    __shared__ __attribute__((aligned(16))) T ldsB[2 * Tiles::B_ELEMS];

    const int tid = threadIdx.x;
    // Lanes of a wave share their output channels (broadcast dy reads), vary input channels.
    const int tc = tid % (BC / TC), tk = tid / (BC / TC);
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (blockIdx.x % kTiles) * BK;
    const int c0 = (blockIdx.x / kTiles) * BC;
    const int g = blockIdx.y / a.ft;
    const int t = blockIdx.y % a.ft;
    const long long begin = blockIdx.z * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;
    const int M = a.trackLen;

    float acc[R * S][TK][TC];
#pragma unroll
    for(int i = 0; i < R * S; ++i)
#pragma unroll
        for(int x = 0; x < TK; ++x)
#pragma unroll
            for(int y = 0; y < TC; ++y)
                acc[i][x][y] = 0.0f;

    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    long long it = begin;
    int n = 0, od = 0, rho = 0, chunk = 0, m = 0;
    if(begin < end)
    {
        const long long track0 = begin / M;
        const int m0 = static_cast<int>(begin - track0 * M);
        const int back = m0 < R - 1 ? m0 : R - 1;
        it = begin - back;
        m = m0 - back;
        chunk = static_cast<int>(track0 % a.chunks);
        const long long t2 = track0 / a.chunks;
        rho = static_cast<int>(t2 % a.dh);
        const long long t3 = t2 / a.dh;
        od = static_cast<int>(t3 % a.dout);
        n = static_cast<int>(t3 / a.dout);
    }
    Tiles tiles;
    tiles.init(a, tid, g, k0, c0, kg, cg);
    auto issue = [&](bool live, bool math) {
        const int oh = rho + m * a.dh;
        const int id = od * a.sd + t * a.dd - a.pd;
        const int ih = rho - a.ph + m * a.dh;
        const bool xOk = live && math && id >= 0 && id < a.di && ih >= 0 && ih < a.hi && n < a.n;
        tiles.load(a,
                   n < a.n ? n : 0,
                   chunk * BQ,
                   live && oh < a.ho && n < a.n,
                   od * ys2 + oh * ys3,
                   xOk,
                   id * xs2 + ih * xs3,
                   kg,
                   cg,
                   k0,
                   c0);
    };
    auto advance = [&]() {
        if(++m == M)
        {
            m = 0;
            if(++chunk == a.chunks)
            {
                chunk = 0;
                if(++rho == a.dh)
                {
                    rho = 0;
                    if(++od == a.dout)
                    {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    };
    // Step j: dy row into ring slot (j - first) % SLOTS, x window into buffer (j - first) % 2.
    int mc = m, slot = 0, xb = 0;
    for(int i = tid; i < SLOTS * Tiles::A_ELEMS; i += THREADS)
        ldsA[i] = T(0);
    __syncthreads();
    if(begin < end)
    {
        issue(true, it >= begin);
        tiles.store(ldsA, ldsB, tid);
        __syncthreads();
    }
    // Ring slots of the rows r = 0 .. R - 1 back (slot of the current row first).
    while(it < end)
    {
        const bool trackStart = mc == 0;
        // A step whose x row lies in the padding only moves the ring (its window is zeros).
        const int ihC = rho - a.ph + m * a.dh;
        const int idC = od * a.sd + t * a.dd - a.pd;
        const bool xIn = idC >= 0 && idC < a.di && ihC >= 0 && ihC < a.hi;
        mc = mc + 1 == M ? 0 : mc + 1;
        advance();
        issue(it + 1 < end, it + 1 >= begin);
        if(it >= begin && xIn)
        {
            const T* Bs = ldsB + xb * Tiles::B_ELEMS + tc * TC;
#pragma unroll 2
            for(int q = 0; q < BQ; ++q)
            {
                float av[R][TK], bv[S][TC];
#pragma unroll
                for(int r = 0; r < R; ++r)
                {
                    // Row m - r: slot - r (mod SLOTS); rows before the track start are zero.
                    int sl = slot - r;
                    sl = sl < 0 ? sl + SLOTS : sl;
                    const T* ap = ldsA + sl * Tiles::A_ELEMS + q * LDA + tk * TK;
#pragma unroll
                    for(int x = 0; x < TK; ++x)
                        av[r][x] = K1Elem<T>::load(ap[x]);
                }
#pragma unroll
                for(int s = 0; s < S; ++s)
                {
                    const T* bp = Bs + (q + s * DW) * LDB;
#pragma unroll
                    for(int y = 0; y < TC; ++y)
                        bv[s][y] = K1Elem<T>::load(bp[y]);
                }
#pragma unroll
                for(int r = 0; r < R; ++r)
#pragma unroll
                    for(int s = 0; s < S; ++s)
#pragma unroll
                        for(int x = 0; x < TK; ++x)
#pragma unroll
                            for(int y = 0; y < TC; ++y)
                                acc[r * S + s][x][y]
                                    = __builtin_fmaf(av[r][x], bv[s][y], acc[r * S + s][x][y]);
            }
        }
        // Next step's dy row replaces the oldest slot; a new track first clears the ring.
        const int ns = slot + 1 == SLOTS ? 0 : slot + 1;
        __syncthreads();
        if(mc == 0 && it + 1 < end)
        {
            for(int i = tid; i < SLOTS * Tiles::A_ELEMS; i += THREADS)
                ldsA[i] = T(0);
            __syncthreads();
        }
        tiles.store(ldsA + ns * Tiles::A_ELEMS, ldsB + (xb ^ 1) * Tiles::B_ELEMS, tid);
        __syncthreads();
        (void)trackStart;
        slot = ns;
        xb ^= 1;
        ++it;
    }
    k3WaitLoads<0>();
    // Partials [split][g][tap][k][c], or dw (k3OutOf).
    if constexpr(!CONV_K3_DWS)
    {
        const long long kc = static_cast<long long>(kg) * cg;
        const int taps = a.ft * R * S;
        float* out
            = a.partial + blockIdx.z * a.partialElems + static_cast<long long>(g) * taps * kc;
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < TK; ++x)
                {
                    const int k = k0 + tk * TK + x;
                    float* o = out + static_cast<long long>((t * R + r) * S + s) * kc
                               + static_cast<long long>(k) * cg;
                    if(k < kg)
                        k3StoreFmaRun<TC>(o, c0 + tc * TC, cg, acc[r * S + s][x]);
                }
        return;
    }
    const K3Out out = k3OutOf<T>(a, g, a.ft * R * S, blockIdx.z);
    k3WithOut<T>(a, out, [&](auto* base) {
        typedef typename K3PointeeOf<decltype(base)>::type E;
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < TK; ++x)
                {
                    const int k = k0 + tk * TK + x;
                    if(k < kg)
                        k3StoreRun<E, TC>(base + ((t * R + r) * S + s) * out.ts + k * out.ks,
                                          c0 + tc * TC,
                                          cg,
                                          acc[r * S + s][x]);
                }
    });
}

// ---------------------------------------------------------------------------------------------
// FMA output-row kernel (exact fp32 without an fp32 matrix core: RDNA), for what the FMA ring
// does not take: any stride, rows not a whole number of vectors (vec 1). A step is one output
// row chunk: the dy chunk and the FR filter rows' input windows are staged as [position]
// [channel] rows (window row q SW + s DW holds tap s of position q, so a stride is a row step,
// no phase split). A thread owns TK x TC channels of every FR x S tap.
template <class T,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int FR,
          int S,
          int SW,
          int DW,
          int TK,
          int TC,
          int VEC>
struct K3FmaRowTiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int THREADS = (BK / TK) * (BC / TC);
    static constexpr int WIN = (BQ - 1) * SW + (S - 1) * DW + 1;
    static constexpr int NBV = (WIN + VEC - 1) / VEC + (VEC > 1 ? 1 : 0); // units from iwA
    static constexpr int ROWS_B = NBV * VEC;
    static constexpr int LDA = BK + 4, LDB = BC + 4;
    static constexpr int A_ELEMS = BQ * LDA;
    static constexpr int B_ELEMS = FR * ROWS_B * LDB;
    static constexpr int CH = 4;
    static constexpr int A_UNITS = LAYOUT == K3_NCHW ? (BK / CH) * (BQ / VEC) : BQ * (BK / VEC);
    static constexpr int B_UNITS
        = LAYOUT == K3_NCHW ? FR * (BC / CH) * NBV : FR * ROWS_B * (BC / VEC);
    static constexpr int A_PASSES = (A_UNITS + THREADS - 1) / THREADS;
    static constexpr int B_PASSES = (B_UNITS + THREADS - 1) / THREADS;
    static constexpr int PER = LAYOUT == K3_NCHW ? CH : 1;
    static_assert(BQ % VEC == 0 && BK % CH == 0 && BC % CH == 0 && BK % VEC == 0 && BC % VEC == 0,
                  "K3 FMA row tiles");

    Raw ra[A_PASSES * PER], rb[B_PASSES * PER];
    int aOff[A_PASSES], aPos[A_PASSES], aCh[A_PASSES];
    int bOff[B_PASSES], bPos[B_PASSES], bCh[B_PASSES], bF[B_PASSES];
    unsigned aOk, bOk;
    int shift;

    __device__ void init(const K3Args& a, int tid, int g, int k0, int c0, int kg, int cg)
    {
        aOk = bOk = 0;
        const int ys1 = static_cast<int>(a.ys[1]), ys4 = static_cast<int>(a.ys[4]);
        const int xs1 = static_cast<int>(a.xs[1]), xs4 = static_cast<int>(a.xs[4]);
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(LAYOUT == K3_NCHW)
            {
                aCh[p] = (u / (BQ / VEC)) * CH;
                aPos[p] = (u % (BQ / VEC)) * VEC;
                aOff[p] = (g * kg + k0 + aCh[p]) * ys1 + aPos[p];
            }
            else
            {
                aCh[p] = (u % (BK / VEC)) * VEC;
                aPos[p] = u / (BK / VEC);
                aOff[p] = aPos[p] * ys4 + g * kg + k0 + aCh[p];
            }
            if(u < A_UNITS && k0 + aCh[p] < kg)
                aOk |= 1u << p;
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(LAYOUT == K3_NCHW)
            {
                const int v = u % NBV, rest = u / NBV;
                bCh[p] = (rest % (BC / CH)) * CH;
                bF[p] = rest / (BC / CH);
                bPos[p] = v * VEC;
                bOff[p] = (g * cg + c0 + bCh[p]) * xs1 + bPos[p];
            }
            else
            {
                const int cv = u % (BC / VEC), rest = u / (BC / VEC);
                bCh[p] = cv * VEC;
                bPos[p] = rest % ROWS_B;
                bF[p] = rest / ROWS_B;
                bOff[p] = bPos[p] * xs4 + g * cg + c0 + bCh[p];
            }
            if(u < B_UNITS && c0 + bCh[p] < cg)
                bOk |= 1u << p;
        }
    }

    /// One step's loads: the dy chunk (yOk, yRow) from q0 and filter row f's window (xOk[f],
    /// xRow[f]); channels-first units start at multiples of VEC within the row.
    __device__ void load(const K3Args& a,
                         int n,
                         int q0,
                         bool yOk,
                         int yRow,
                         const bool (&xOk)[FR],
                         const int (&xRow)[FR],
                         int kg,
                         int cg,
                         int k0,
                         int c0)
    {
        const k3i4 yr = k3Rsrc4(static_cast<const T*>(a.dy) + n * a.yImage, a.yImage * sizeof(T));
        const k3i4 xr = k3Rsrc4(static_cast<const T*>(a.x) + n * a.xImage, a.xImage * sizeof(T));
        const int ys1 = static_cast<int>(a.ys[1]), ys4 = static_cast<int>(a.ys[4]);
        const int xs1 = static_cast<int>(a.xs[1]), xs4 = static_cast<int>(a.xs[4]);
        const int iw0 = q0 * SW - a.pw;
        const int iwA = LAYOUT == K3_NCHW
                            ? (iw0 >= 0 ? iw0 / VEC * VEC : -((-iw0 + VEC - 1) / VEC) * VEC)
                            : iw0;
        shift = iwA - iw0;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int q = q0 + aPos[p];
            const bool ok = yOk && ((aOk >> p) & 1u) && q < a.wo;
#pragma unroll
            for(int i = 0; i < PER; ++i)
            {
                const int off = aOff[p] + yRow + (LAYOUT == K3_NCHW ? q0 + i * ys1 : q0 * ys4);
                const bool oki = ok && (LAYOUT != K3_NCHW || k0 + aCh[p] + i < kg);
                k3Load(ra[p * PER + i],
                       yr,
                       oki ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            bool rowOk = false;
            int row = 0;
#pragma unroll
            for(int f = 0; f < FR; ++f)
                if(bF[p] == f)
                {
                    rowOk = xOk[f];
                    row = xRow[f];
                }
            const int iw = iwA + bPos[p];
            const bool ok = rowOk && ((bOk >> p) & 1u) && iw >= 0 && iw < a.wi;
#pragma unroll
            for(int i = 0; i < PER; ++i)
            {
                const int off = bOff[p] + row + (LAYOUT == K3_NCHW ? iwA + i * xs1 : iwA * xs4);
                const bool oki = ok && (LAYOUT != K3_NCHW || c0 + bCh[p] + i < cg);
                k3Load(rb[p * PER + i],
                       xr,
                       oki ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE);
            }
        }
    }

    /// Stores the step: dy rows As[q][k], windows Bs[f][row][c] with row = position - iw0.
    __device__ void store(T* As, T* Bs, int tid) const
    {
        typedef typename K1Raw<CH * sizeof(T)>::type W;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(A_UNITS % THREADS != 0 && u >= A_UNITS)
                continue;
            if(LAYOUT == K3_NCHW)
            {
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                {
                    T w[CH];
#pragma unroll
                    for(int i = 0; i < CH; ++i)
                        w[i] = k3Elem<T, VEC>(ra[p * PER + i], e);
                    *reinterpret_cast<W*>(As + (aPos[p] + e) * LDA + aCh[p])
                        = __builtin_bit_cast(W, w);
                }
            }
            else
            {
                *reinterpret_cast<Raw*>(As + aPos[p] * LDA + aCh[p]) = ra[p];
            }
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            if(B_UNITS % THREADS != 0 && u >= B_UNITS)
                continue;
            T* base = Bs + bF[p] * ROWS_B * LDB;
            if(LAYOUT == K3_NCHW)
            {
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                {
                    const int row = bPos[p] + e + shift;
                    if(row < 0 || row >= ROWS_B)
                        continue;
                    T w[CH];
#pragma unroll
                    for(int i = 0; i < CH; ++i)
                        w[i] = k3Elem<T, VEC>(rb[p * PER + i], e);
                    *reinterpret_cast<W*>(base + row * LDB + bCh[p]) = __builtin_bit_cast(W, w);
                }
            }
            else
            {
                *reinterpret_cast<Raw*>(base + bPos[p] * LDB + bCh[p]) = rb[p];
            }
        }
    }
};

template <class T,
          int LAYOUT,
          int BK,
          int BC,
          int BQ,
          int FR,
          int S,
          int SW,
          int DW,
          int TK,
          int TC,
          int VEC>
__device__ K3_LOOP_INLINE void k3WrwRowsFma(const K3Args& a)
{
    typedef K3FmaRowTiles<T, LAYOUT, BK, BC, BQ, FR, S, SW, DW, TK, TC, VEC> Tiles;
    static_assert(
        k3LdsBytes(4, LAYOUT, sizeof(T), BK, BC, BQ, FR, S, SW, DW, 0, 0, TK, TC, VEC, 0, 0)
            == 2 * (Tiles::A_ELEMS + Tiles::B_ELEMS) * sizeof(T),
        "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int THREADS = Tiles::THREADS;
    constexpr int LDA = Tiles::LDA, LDB = Tiles::LDB, BUF = Tiles::A_ELEMS + Tiles::B_ELEMS;
    __shared__ __attribute__((aligned(16))) T lds[2 * BUF];

    const int tid = threadIdx.x;
    const int tc = tid % (BC / TC), tk = tid / (BC / TC);
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (blockIdx.x % kTiles) * BK;
    const int c0 = (blockIdx.x / kTiles) * BC;
    const int frows = a.ft * a.fr;
    const int frGroups = (frows + FR - 1) / FR;
    const int g = blockIdx.y / frGroups;
    const int fr0 = (blockIdx.y % frGroups) * FR;
    const long long begin = blockIdx.z * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;

    float acc[FR * S][TK][TC];
#pragma unroll
    for(int i = 0; i < FR * S; ++i)
#pragma unroll
        for(int x = 0; x < TK; ++x)
#pragma unroll
            for(int y = 0; y < TC; ++y)
                acc[i][x][y] = 0.0f;
    int ft[FR], frr[FR];
#pragma unroll
    for(int f = 0; f < FR; ++f)
    {
        ft[f] = (fr0 + f) / a.fr;
        frr[f] = (fr0 + f) - ft[f] * a.fr;
    }
    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    int n = 0, od = 0, oh = 0, chunk = 0;
    if(begin < end)
    {
        const long long row = begin / a.chunks;
        chunk = static_cast<int>(begin - row * a.chunks);
        oh = static_cast<int>(row % a.ho);
        const long long r2 = row / a.ho;
        od = static_cast<int>(r2 % a.dout);
        n = static_cast<int>(r2 / a.dout);
    }
    Tiles tiles;
    tiles.init(a, tid, g, k0, c0, kg, cg);
    auto issue = [&](bool live) {
        live = live && n < a.n;
        bool xOk[FR];
        int xRow[FR];
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            const int id = od * a.sd + ft[f] * a.dd - a.pd;
            const int ih = oh * a.sh + frr[f] * a.dh - a.ph;
            xOk[f] = live && fr0 + f < frows && id >= 0 && id < a.di && ih >= 0 && ih < a.hi;
            xRow[f] = id * xs2 + ih * xs3;
        }
        tiles.load(
            a, live ? n : 0, chunk * BQ, live, od * ys2 + oh * ys3, xOk, xRow, kg, cg, k0, c0);
    };
    auto advance = [&]() {
        if(++chunk == a.chunks)
        {
            chunk = 0;
            if(++oh == a.ho)
            {
                oh = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++n;
                }
            }
        }
    };
    if(begin < end)
    {
        issue(true);
        k3WaitLoads<0>();
        tiles.store(lds, lds + Tiles::A_ELEMS, tid);
        __syncthreads();
    }
    int buf = 0;
    for(long long step = begin; step < end; ++step)
    {
        advance();
        issue(step + 1 < end);
        const T* As = lds + buf * BUF + tk * TK;
        const T* Bs = lds + buf * BUF + Tiles::A_ELEMS + tc * TC;
#pragma unroll 2
        for(int q = 0; q < BQ; ++q)
        {
            float av[TK];
#pragma unroll
            for(int x = 0; x < TK; ++x)
                av[x] = K1Elem<T>::load(As[q * LDA + x]);
#pragma unroll
            for(int f = 0; f < FR; ++f)
#pragma unroll
                for(int s = 0; s < S; ++s)
                {
                    const T* bp = Bs + (f * Tiles::ROWS_B + q * SW + s * DW) * LDB;
                    float bv[TC];
#pragma unroll
                    for(int y = 0; y < TC; ++y)
                        bv[y] = K1Elem<T>::load(bp[y]);
#pragma unroll
                    for(int x = 0; x < TK; ++x)
#pragma unroll
                        for(int y = 0; y < TC; ++y)
                            acc[f * S + s][x][y]
                                = __builtin_fmaf(av[x], bv[y], acc[f * S + s][x][y]);
                }
        }
        k3WaitLoads<0>();
        tiles.store(lds + (buf ^ 1) * BUF, lds + (buf ^ 1) * BUF + Tiles::A_ELEMS, tid);
        __syncthreads();
        buf ^= 1;
    }
    k3WaitLoads<0>();
    if constexpr(!CONV_K3_DWS)
    {
        const long long kc = static_cast<long long>(kg) * cg;
        const int taps = frows * S;
        float* out
            = a.partial + blockIdx.z * a.partialElems + static_cast<long long>(g) * taps * kc;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            if(fr0 + f >= frows)
                continue;
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < TK; ++x)
                {
                    const int k = k0 + tk * TK + x;
                    float* o = out + static_cast<long long>((fr0 + f) * S + s) * kc
                               + static_cast<long long>(k) * cg;
                    if(k < kg)
                        k3StoreFmaRun<TC>(o, c0 + tc * TC, cg, acc[f * S + s][x]);
                }
        }
        return;
    }
    const K3Out out = k3OutOf<T>(a, g, frows * S, blockIdx.z);
    k3WithOut<T>(a, out, [&](auto* base) {
        typedef typename K3PointeeOf<decltype(base)>::type E;
#pragma unroll
        for(int f = 0; f < FR; ++f)
        {
            if(fr0 + f >= frows)
                continue;
#pragma unroll
            for(int s = 0; s < S; ++s)
#pragma unroll
                for(int x = 0; x < TK; ++x)
                {
                    const int k = k0 + tk * TK + x;
                    if(k < kg)
                        k3StoreRun<E, TC>(base + ((fr0 + f) * S + s) * out.ts + k * out.ks,
                                          c0 + tc * TC,
                                          cg,
                                          acc[f * S + s][x]);
                }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// gfx950 LDS-DMA row-ring kernel (channels-first, width stride 1). Same loop and ring as
// k3WrwRing, but the global loads write LDS directly (buffer_load ... lds, 16 bytes per lane),
// so no load registers are held and NS stages of LDS are in flight: step j is staged into slot
// j % NS while slots for steps j + 1 .. j + NS - 1 are loading. A DMA instruction writes 64
// lane-contiguous 16-byte units, so the LDS rows are whole units: dy rows of BQ / VEC + 1 units
// (one pad unit, for banks) and x windows of NBV units (odd, for banks). Every wave issues the
// same number of DMA instructions per step (surplus units land in the slot's tail), so
// one wait immediate fits all waves. A step whose window or chunk leaves its rows zeros the
// out-of-row elements once its data has landed; a unit past either end of the image is
// reloaded element by element there.
#if defined(__gfx950__)
#define K3_HAS_DMA 1
#else
#define K3_HAS_DMA 0
#endif

/// A barrier that waits only for this wave's LDS operations (__syncthreads' fence would also
/// drain the DMA in flight).
#define K3_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

template <class T, int BK, int BC, int BQ, int S, int DW, int WAVES>
struct K3DmaGeom
{
    static constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    static constexpr int NQA = BQ / VEC + 1;
    static constexpr int LDA = NQA * VEC;
    static constexpr int WIN = BQ + (S - 1) * DW;
    static constexpr int NBV0 = (WIN + 2 + VEC - 1) / VEC;
    static constexpr int NBV = NBV0 | 1;
    static constexpr int LDB = NBV * VEC;
    static constexpr int A_UNITS = BK * NQA, B_UNITS = BC * NBV;
    static constexpr int A_GROUPS = (A_UNITS + 63) / 64, B_GROUPS = (B_UNITS + 63) / 64;
    static constexpr int A_PASSES = (A_GROUPS + WAVES - 1) / WAVES;
    static constexpr int B_PASSES = (B_GROUPS + WAVES - 1) / WAVES;
    static constexpr int LOADS = A_PASSES + B_PASSES; // DMA instructions per wave per step
    // Every wave's passes land in the slot (surplus units in its tail), so a DMA's
    // destination is always the slot's own object and the compiler's waits stay exact.
    static constexpr int A_ELEMS = A_PASSES * WAVES * 64 * VEC;
    static constexpr int SLOT = (A_PASSES + B_PASSES) * WAVES * 64 * VEC; // elements
    static_assert(BQ % VEC == 0, "K3 DMA chunk");
};

template <class T,
          class Atom,
          int BK,
          int BC,
          int BQ,
          int R,
          int S,
          int DW,
          int WK,
          int WC,
          int NS,
          int XCD = 0>
__device__ K3_LOOP_INLINE void k3WrwRingDma(const K3Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int WAVES = WK * WC;
    constexpr int DIM = Atom::DIM, KA = Atom::KA, KPL = Atom::KPL;
    constexpr int WTK = BK / WK, WTC = BC / WC;
    constexpr int FK = WTK / DIM, FC = WTC / DIM;
    constexpr int KK = BQ / KA;
    static_assert(FK * DIM * WK == BK && FC * DIM * WC == BC && BQ % KA == 0, "K3 tile");
    static_assert(NS >= 2 && NS <= 4, "K3 DMA stages");
    typedef K3DmaGeom<T, BK, BC, BQ, S, DW, WAVES> G;
    static_assert(k3LdsBytes(2, 0, sizeof(T), BK, BC, BQ, R, S, 1, DW, WK, WC, 0, 0, 0, 0, NS)
                      == NS * G::SLOT * sizeof(T),
                  "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int VEC = G::VEC, LDA = G::LDA, LDB = G::LDB, NBV = G::NBV, NQA = G::NQA;

    // One __shared__ object per slot: the compiler's wait insertion tracks DMA per object, so
    // reading one slot does not wait for DMA into another.
    __shared__ __attribute__((aligned(16))) T slot0[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot1[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot2[NS > 2 ? G::SLOT : 1];
    __shared__ __attribute__((aligned(16))) T slot3[NS > 3 ? G::SLOT : 1];
    auto slotPtr = [&](auto sc) -> T* {
        constexpr int I = decltype(sc)::value;
        if constexpr(I == 0)
            return slot0;
        else if constexpr(I == 1)
            return slot1;
        else if constexpr(I == 2)
            return slot2;
        else
            return slot3;
    };

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wk = wave % WK, wc = wave / WK;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    const int g = by / a.ft;
    const int t = by % a.ft;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;
    const int M = a.trackLen;

    typename Atom::acc acc[R * S][FK][FC];
#pragma unroll
    for(int i = 0; i < R * S; ++i)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FC; ++y)
                acc[i][x][y] = typename Atom::acc{};
    typename Atom::frag ring[R][KK][FK];
#pragma unroll
    for(int r = 0; r < R; ++r)
#pragma unroll
        for(int kk = 0; kk < KK; ++kk)
#pragma unroll
            for(int x = 0; x < FK; ++x)
                ring[r][kk][x] = typename Atom::frag{};

    // Per pass: the unit's global offset part and validity (channel in range, a data unit).
    int aOff[G::A_PASSES], bOff[G::B_PASSES], bPos[G::B_PASSES];
    unsigned aOk = 0, bOk = 0;
#pragma unroll
    for(int p = 0; p < G::A_PASSES; ++p)
    {
        const int u = (p * WAVES + wave) * 64 + lane;
        const int row = u / NQA, qv = u % NQA;
        aOff[p] = (g * kg + k0 + row) * static_cast<int>(a.ys[1]) + qv * VEC;
        if(u < G::A_UNITS && qv < BQ / VEC && k0 + row < kg)
            aOk |= 1u << p;
    }
#pragma unroll
    for(int p = 0; p < G::B_PASSES; ++p)
    {
        const int u = (p * WAVES + wave) * 64 + lane;
        const int c = u / NBV, v = u % NBV;
        bPos[p] = v * VEC;
        bOff[p] = (g * cg + c0 + c) * static_cast<int>(a.xs[1]) + v * VEC;
        if(u < G::B_UNITS && c0 + c < cg)
            bOk |= 1u << p;
    }

    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    long long it = begin;
    int n = 0, od = 0, rho = 0, chunk = 0, m = 0;
    if(begin < end)
    {
        const long long track0 = begin / M;
        const int m0 = static_cast<int>(begin - track0 * M);
        const int back = m0 < R - 1 ? m0 : R - 1;
        it = begin - back;
        m = m0 - back;
        chunk = static_cast<int>(track0 % a.chunks);
        const long long t2 = track0 / a.chunks;
        rho = static_cast<int>(t2 % a.dh);
        const long long t3 = t2 / a.dh;
        od = static_cast<int>(t3 % a.dout);
        n = static_cast<int>(t3 / a.dout);
    }
    auto advance = [&]() {
        if(++m == M)
        {
            m = 0;
            if(++chunk == a.chunks)
            {
                chunk = 0;
                if(++rho == a.dh)
                {
                    rho = 0;
                    if(++od == a.dout)
                    {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    };
    // Per slot: the staged step's fix-up inputs (uniform).
    struct Staged
    {
        int n, q0, iw0, yRow, xRow;
        unsigned rows;
        bool yOk, xOk, edge;
    };
    Staged st[NS];

    // Issues the cursor's step into slot I (every wave the same DMA count).
    auto issue = [&](auto sc, bool live, bool math) {
        constexpr int I = decltype(sc)::value;
        T* base = slotPtr(sc);
        const bool ok = live && n < a.n;
        const int nn = ok ? n : 0;
        const int oh = rho + m * a.dh;
        const int id = od * a.sd + t * a.dd - a.pd;
        const int ih = rho - a.ph + m * a.dh;
        Staged s;
        s.n = nn;
        s.q0 = chunk * BQ;
        s.iw0 = s.q0 - a.pw;
        s.yOk = ok && oh < a.ho;
        s.xOk = ok && math && id >= 0 && id < a.di && ih >= 0 && ih < a.hi;
        s.yRow = od * ys2 + oh * ys3;
        s.xRow = id * xs2 + ih * xs3;
        s.rows = k3RowsOf(m, rho < a.ho ? (a.ho - rho + a.dh - 1) / a.dh : 0, R);
        // Edge: a run leaves its row, or (16-bit) a unit may pass the image's end.
        s.edge = (s.yOk && (s.q0 + BQ > a.wo || (sizeof(T) == 2 && oh == a.ho - 1)))
                 || (s.xOk && (s.iw0 < 0 || s.iw0 + NBV * VEC > a.wi || ih == 0 || ih == a.hi - 1));
        st[I] = s;
        const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.dy)) + nn * a.yImage,
            static_cast<short>(0),
            static_cast<int>(a.yImage * sizeof(T)),
            K1_RSRC_FLAGS);
        const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.x)) + nn * a.xImage,
            static_cast<short>(0),
            static_cast<int>(a.xImage * sizeof(T)),
            K1_RSRC_FLAGS);
        const int elemsX = static_cast<int>(a.xImage), elemsY = static_cast<int>(a.yImage);
#if K3_HAS_DMA
#pragma unroll
        for(int p = 0; p < G::A_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            const bool real = gi < G::A_GROUPS;
            const int off = aOff[p] + s.yRow + s.q0;
            const bool u = s.yOk && ((aOk >> p) & 1u) && off + VEC <= elemsY;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                yr,
                (__attribute__((address_space(3))) void*)(base + gi * 64 * VEC),
                16,
                u ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
#pragma unroll
        for(int p = 0; p < G::B_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            const bool real = gi < G::B_GROUPS;
            const int off = bOff[p] + s.xRow + s.iw0;
            // A unit past either end of the image loads nothing here (fixed after it lands).
            const bool u = s.xOk && ((bOk >> p) & 1u) && off >= 0 && off + VEC <= elemsX;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                xr,
                (__attribute__((address_space(3))) void*)(base + G::A_ELEMS + gi * 64 * VEC),
                16,
                u ? static_cast<unsigned>(off) * sizeof(T) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
#endif
    };
    // Fixes slot I's edge units once landed: out-of-row elements to zero, units past the
    // image's ends reloaded element by element.
    auto fixup = [&](auto sc) {
        constexpr int I = decltype(sc)::value;
        const Staged& s = st[I];
        if(!s.edge)
            return;
        T* base = slotPtr(sc);
        typedef typename K1Raw<sizeof(T)>::type One;
        One* lds1 = reinterpret_cast<One*>(base);
        const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.dy)) + s.n * a.yImage,
            static_cast<short>(0),
            static_cast<int>(a.yImage * sizeof(T)),
            K1_RSRC_FLAGS);
        const int elemsY = static_cast<int>(a.yImage);
#pragma unroll
        for(int p = 0; p < G::A_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            if(gi >= G::A_GROUPS || !s.yOk || !((aOk >> p) & 1u))
                continue;
            const int u = gi * 64 + lane;
            const int pos = s.q0 + (u % NQA) * VEC;
            const int off = aOff[p] + s.yRow + s.q0;
            const bool reload = off + VEC > elemsY;
            if(pos + VEC <= a.wo && !reload)
                continue;
#pragma unroll
            for(int i = 0; i < VEC; ++i)
            {
                const bool in = pos + i < a.wo;
                One v = One(0);
                if(in && reload)
                    v = k1BufferLoad<One>(yr, static_cast<unsigned>(off + i) * sizeof(T));
                if(!in || reload)
                    lds1[u * VEC + i] = v;
            }
        }
        const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.x)) + s.n * a.xImage,
            static_cast<short>(0),
            static_cast<int>(a.xImage * sizeof(T)),
            K1_RSRC_FLAGS);
        const int elemsX = static_cast<int>(a.xImage);
#pragma unroll
        for(int p = 0; p < G::B_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            if(gi >= G::B_GROUPS || !s.xOk || !((bOk >> p) & 1u))
                continue;
            const int u = gi * 64 + lane;
            const int iw = s.iw0 + bPos[p];
            const int off = bOff[p] + s.xRow + s.iw0;
            const bool reload = off < 0 || off + VEC > elemsX;
            if(iw >= 0 && iw + VEC <= a.wi && !reload)
                continue;
#pragma unroll
            for(int i = 0; i < VEC; ++i)
            {
                const bool in = iw + i >= 0 && iw + i < a.wi;
                One v = One(0);
                if(in && reload)
                    v = k1BufferLoad<One>(xr, static_cast<unsigned>(off + i) * sizeof(T));
                if(!in || reload)
                    lds1[G::A_ELEMS + u * VEC + i] = v;
            }
        }
    };

    const int kb = Atom::template kBase<KPL>(lane);
    // Prologue: steps it .. it + NS - 2 into slots 0 .. NS - 2.
    long long issued = it; // the step the cursor is on
    k3ForEach<NS - 1>([&](auto sc) {
        if(decltype(sc)::value > 0)
            advance();
        issue(sc, issued + decltype(sc)::value < end, issued + decltype(sc)::value >= begin);
    });
    // Step it sits in slot (it - first step) % NS; mc is its place in its track.
    int mc = static_cast<int>(it % M);
    while(it < end)
    {
        k3ForEach<NS>([&](auto sc) {
            constexpr int J = decltype(sc)::value;
            if(it >= end)
                return;
            // Wait for this step's slot (NS - 2 newer steps may still load), fix it, publish.
            k3WaitLoads<(NS - 2) * G::LOADS>();
            fixup(sc);
            __syncthreads();
            // Issue the step NS - 1 ahead into the slot freed last step.
            constexpr int NEXT = (J + NS - 1) % NS;
            advance();
            issue(K3Int<NEXT>{}, it + NS - 1 < end, it + NS - 1 >= begin);
            const bool trackStart = mc == 0;
            mc = mc + 1 == M ? 0 : mc + 1;
            const T* As = slotPtr(sc);
            const T* Bs = As + G::A_ELEMS;
#pragma unroll
            for(int r = R - 1; r > 0; --r)
#pragma unroll
                for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                    for(int x = 0; x < FK; ++x)
                        ring[r][kk][x] = trackStart ? typename Atom::frag{} : ring[r - 1][kk][x];
            const T* aBase = As + (wk * WTK + lane % DIM) * LDA + kb;
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    ring[0][kk][x] = k3FragAt<Atom, T, 0>(aBase + x * DIM * LDA + kk * KA);
            if(it >= begin && st[J].xOk)
                k3MultiplyRows<Atom, T, R, S, 1, DW, FK, FC, BC, LDB, KK>(
                    acc, ring, Bs + (wc * WTC + lane % DIM) * LDB + kb, st[J].rows);
            // All waves are done with this slot before it is refilled (NS - 1 steps later the
            // barrier above orders it).
            ++it;
        });
    }
    k3WaitLoads<0>();
    k3StorePartials<T, Atom, R, S, FK, FC, WTK, WTC>(
        a, acc, g, t * R, a.ft * R, k0, c0, wk, wc, lane, bz);
}

// ---------------------------------------------------------------------------------------------
// gfx950 LDS-DMA row-ring kernel for channels-last 16-bit data: k3WrwRingDma's pipeline with the
// tiles kept channels-last ([position][channel] rows, one pad unit per row against bank
// conflicts) and the fragments read by ds_read_b64_tr_b16. A unit is one position's 8 channels,
// so a position outside its row is one out-of-range unit (zeros): no fix-ups.
template <int BK, int BC, int BQ, int S, int DW, int WAVES>
struct K3DmaTrGeom
{
    static constexpr int VEC = 8;
    static constexpr int NA = k3TrPitch(BK) / VEC, NB = k3TrPitch(BC) / VEC;
    static constexpr int LDA = NA * VEC, LDB = NB * VEC;
    static constexpr int WIN = BQ + (S - 1) * DW;
    static constexpr int A_UNITS = BQ * NA, B_UNITS = WIN * NB;
    static constexpr int A_GROUPS = (A_UNITS + 63) / 64, B_GROUPS = (B_UNITS + 63) / 64;
    static constexpr int A_PASSES = (A_GROUPS + WAVES - 1) / WAVES;
    static constexpr int B_PASSES = (B_GROUPS + WAVES - 1) / WAVES;
    static constexpr int LOADS = A_PASSES + B_PASSES;
    // Every wave's passes land in the slot (surplus units in its tail, read as nothing), so a
    // DMA's destination is always the slot's own object and the compiler's waits stay exact.
    static constexpr int A_ELEMS = A_PASSES * WAVES * 64 * VEC;
    static constexpr int SLOT = (A_PASSES + B_PASSES) * WAVES * 64 * VEC;
    static_assert(BK % VEC == 0 && BC % VEC == 0, "K3 DMA TR channels");
};

template <class T,
          class Atom,
          int BK,
          int BC,
          int BQ,
          int R,
          int S,
          int DW,
          int WK,
          int WC,
          int NS,
          int XCD = 0>
__device__ K3_LOOP_INLINE void k3WrwRingDmaTr(const K3Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int WAVES = WK * WC;
    constexpr int DIM = Atom::DIM, KA = Atom::KA;
    constexpr int WTK = BK / WK, WTC = BC / WC;
    constexpr int FK = WTK / DIM, FC = WTC / DIM;
    constexpr int KK = BQ / KA;
    static_assert(FK * DIM * WK == BK && FC * DIM * WC == BC && BQ % KA == 0, "K3 tile");
    static_assert(NS >= 2 && NS <= 4 && sizeof(T) == 2, "K3 DMA TR");
    typedef K3DmaTrGeom<BK, BC, BQ, S, DW, WAVES> G;
    static_assert(k3LdsBytes(2, 2, sizeof(T), BK, BC, BQ, R, S, 1, DW, WK, WC, 0, 0, 0, 0, NS)
                      == NS * G::SLOT * sizeof(T),
                  "K3 LDS arithmetic (ConvWrwGeom.h)");
    constexpr int VEC = G::VEC, LDA = G::LDA, LDB = G::LDB, NA = G::NA, NB = G::NB;

    __shared__ __attribute__((aligned(16))) T slot0[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot1[G::SLOT];
    __shared__ __attribute__((aligned(16))) T slot2[NS > 2 ? G::SLOT : 1];
    __shared__ __attribute__((aligned(16))) T slot3[NS > 3 ? G::SLOT : 1];
    auto slotPtr = [&](auto sc) -> T* {
        constexpr int I = decltype(sc)::value;
        if constexpr(I == 0)
            return slot0;
        else if constexpr(I == 1)
            return slot1;
        else if constexpr(I == 2)
            return slot2;
        else
            return slot3;
    };

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int wk = wave % WK, wc = wave / WK;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    unsigned bx, by, bz;
    k3BlockIds<XCD>(bx, by, bz);
    const int kTiles = (kg + BK - 1) / BK;
    const int k0 = (bx % kTiles) * BK;
    const int c0 = (bx / kTiles) * BC;
    const int g = by / a.ft;
    const int t = by % a.ft;
    const long long begin = bz * a.stepsPerSplit;
    const long long end = begin + a.stepsPerSplit < a.steps ? begin + a.stepsPerSplit : a.steps;
    const int M = a.trackLen;

    typename Atom::acc acc[R * S][FK][FC];
#pragma unroll
    for(int i = 0; i < R * S; ++i)
#pragma unroll
        for(int x = 0; x < FK; ++x)
#pragma unroll
            for(int y = 0; y < FC; ++y)
                acc[i][x][y] = typename Atom::acc{};
    typename Atom::frag ring[R][KK][FK];
#pragma unroll
    for(int r = 0; r < R; ++r)
#pragma unroll
        for(int kk = 0; kk < KK; ++kk)
#pragma unroll
            for(int x = 0; x < FK; ++x)
                ring[r][kk][x] = typename Atom::frag{};

    // Per pass: the unit's offset part (position and channels), its position, channel validity.
    const int ys4 = static_cast<int>(a.ys[4]), xs4 = static_cast<int>(a.xs[4]);
    int aOff[G::A_PASSES], aPos[G::A_PASSES], bOff[G::B_PASSES], bPos[G::B_PASSES];
    unsigned aOk = 0, bOk = 0;
#pragma unroll
    for(int p = 0; p < G::A_PASSES; ++p)
    {
        const int u = (p * WAVES + wave) * 64 + lane;
        const int q = u / NA, col = u % NA;
        aPos[p] = q;
        aOff[p] = q * ys4 + g * kg + k0 + col * VEC;
        if(u < G::A_UNITS && col < BK / VEC && k0 + col * VEC < kg)
            aOk |= 1u << p;
    }
#pragma unroll
    for(int p = 0; p < G::B_PASSES; ++p)
    {
        const int u = (p * WAVES + wave) * 64 + lane;
        const int j = u / NB, col = u % NB;
        bPos[p] = j;
        bOff[p] = j * xs4 + g * cg + c0 + col * VEC;
        if(u < G::B_UNITS && col < BC / VEC && c0 + col * VEC < cg)
            bOk |= 1u << p;
    }

    const int ys2 = static_cast<int>(a.ys[2]), ys3 = static_cast<int>(a.ys[3]);
    const int xs2 = static_cast<int>(a.xs[2]), xs3 = static_cast<int>(a.xs[3]);
    long long it = begin;
    int n = 0, od = 0, rho = 0, chunk = 0, m = 0;
    if(begin < end)
    {
        const long long track0 = begin / M;
        const int m0 = static_cast<int>(begin - track0 * M);
        const int back = m0 < R - 1 ? m0 : R - 1;
        it = begin - back;
        m = m0 - back;
        chunk = static_cast<int>(track0 % a.chunks);
        const long long t2 = track0 / a.chunks;
        rho = static_cast<int>(t2 % a.dh);
        const long long t3 = t2 / a.dh;
        od = static_cast<int>(t3 % a.dout);
        n = static_cast<int>(t3 / a.dout);
    }
    auto advance = [&]() {
        if(++m == M)
        {
            m = 0;
            if(++chunk == a.chunks)
            {
                chunk = 0;
                if(++rho == a.dh)
                {
                    rho = 0;
                    if(++od == a.dout)
                    {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    };
    unsigned rowsOf[NS];
    bool xInOf[NS]; // the step's x row is in the image (else its window is zeros)
    // Issues the cursor's step into slot I.
    auto issue = [&](auto sc, bool live, bool math) {
        constexpr int I = decltype(sc)::value;
        T* base = slotPtr(sc);
        const bool ok = live && n < a.n;
        const int nn = ok ? n : 0;
        const int oh = rho + m * a.dh;
        const int id = od * a.sd + t * a.dd - a.pd;
        const int ih = rho - a.ph + m * a.dh;
        const int q0 = chunk * BQ;
        const int iw0 = q0 - a.pw;
        const bool yOk = ok && oh < a.ho;
        const bool xOk = ok && math && id >= 0 && id < a.di && ih >= 0 && ih < a.hi;
        const int yRow = od * ys2 + oh * ys3 + q0 * ys4;
        const int xRow = id * xs2 + ih * xs3 + iw0 * xs4;
        rowsOf[I] = k3RowsOf(m, rho < a.ho ? (a.ho - rho + a.dh - 1) / a.dh : 0, R);
        xInOf[I] = xOk;
#if K3_HAS_DMA
        const __amdgpu_buffer_rsrc_t yr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.dy)) + nn * a.yImage,
            static_cast<short>(0),
            static_cast<int>(a.yImage * sizeof(T)),
            K1_RSRC_FLAGS);
        const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<T*>(static_cast<const T*>(a.x)) + nn * a.xImage,
            static_cast<short>(0),
            static_cast<int>(a.xImage * sizeof(T)),
            K1_RSRC_FLAGS);
#pragma unroll
        for(int p = 0; p < G::A_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            const bool u = yOk && ((aOk >> p) & 1u) && q0 + aPos[p] < a.wo;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                yr,
                (__attribute__((address_space(3))) void*)(base + gi * 64 * VEC),
                16,
                u ? static_cast<unsigned>(aOff[p] + yRow) * sizeof(T) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
#pragma unroll
        for(int p = 0; p < G::B_PASSES; ++p)
        {
            const int gi = p * WAVES + wave;
            const int iw = iw0 + bPos[p];
            const bool u = xOk && ((bOk >> p) & 1u) && iw >= 0 && iw < a.wi;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                xr,
                (__attribute__((address_space(3))) void*)(base + G::A_ELEMS + gi * 64 * VEC),
                16,
                u ? static_cast<unsigned>(bOff[p] + xRow) * sizeof(T) : K1_OUT_OF_RANGE,
                0,
                0,
                0);
        }
#endif
    };

    const int aLane = wk * WTK + k3TrLane<Atom, LDA>(lane);
    const int bLane = wc * WTC + k3TrLane<Atom, LDB>(lane);
    long long issued = it;
    k3ForEach<NS - 1>([&](auto sc) {
        if(decltype(sc)::value > 0)
            advance();
        issue(sc, issued + decltype(sc)::value < end, issued + decltype(sc)::value >= begin);
    });
    int mc = static_cast<int>(it % M);
    while(it < end)
    {
        k3ForEach<NS>([&](auto sc) {
            constexpr int J = decltype(sc)::value;
            if(it >= end)
                return;
            k3WaitLoads<(NS - 2) * G::LOADS>();
            K3_BARRIER();
            constexpr int NEXT = (J + NS - 1) % NS;
            advance();
            const unsigned rows = rowsOf[J];
            const bool xIn = xInOf[J];
            issue(K3Int<NEXT>{}, it + NS - 1 < end, it + NS - 1 >= begin);
            const bool trackStart = mc == 0;
            mc = mc + 1 == M ? 0 : mc + 1;
            const T* As = slotPtr(sc);
            const T* Bs = As + G::A_ELEMS;
#pragma unroll
            for(int r = R - 1; r > 0; --r)
#pragma unroll
                for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                    for(int x = 0; x < FK; ++x)
                        ring[r][kk][x] = trackStart ? typename Atom::frag{} : ring[r - 1][kk][x];
#pragma unroll
            for(int kk = 0; kk < KK; ++kk)
#pragma unroll
                for(int x = 0; x < FK; ++x)
                    ring[0][kk][x] = k3FragTr<Atom, T, LDA>(As + aLane + kk * KA * LDA + x * DIM);
            if(it >= begin && xIn)
                k3MultiplyRows<Atom, T, R, S, 1, DW, FK, FC, BC, LDB, KK, 1, G::WIN>(
                    acc, ring, Bs + bLane, rows);
            ++it;
        });
    }
    k3WaitLoads<0>();
    k3StorePartials<T, Atom, R, S, FK, FC, WTK, WTC>(
        a, acc, g, t * R, a.ft * R, k0, c0, wk, wc, lane, bz);
}

/// The row reduce (ConvWrwReduceRows; K C T R S filters of 2 to K3_REDUCE_ROW_MAX_TAPS taps and at
/// most 2 splits): a block is one output channel's run of K3_REDUCE_ROW_CHANNELS channels by all
/// taps, contiguous in dw. Each element sums its splits in split order (reads: consecutive
/// threads take consecutive channels of one tap), the sums are transposed in LDS, and the run is
/// written contiguously; the quad blocks write it a channel apart, taps elements between
/// neighbours. With more splits the quad blocks' split groups win: each element's splits here
/// run in one thread.
template <class T>
__device__ inline void k3ReduceRows(const K3ReduceArgs& a)
{
    constexpr int CB = K3_REDUCE_ROW_CHANNELS;
    __shared__ float tile[CB * K3_REDUCE_ROW_MAX_TAPS];
    const int cTiles = (a.cg + CB - 1) / CB;
    const long long gk = blockIdx.x / cTiles; // g kg + k
    const int c0 = static_cast<int>(blockIdx.x % cTiles) * CB;
    const int cn = a.cg - c0 < CB ? a.cg - c0 : CB;
    const int k = static_cast<int>(gk % a.kg);
    const long long g = gk / a.kg;
    const int taps = a.taps;
    for(int e = threadIdx.x; e < CB * taps; e += blockDim.x)
    {
        const int tap = e / CB, cl = e % CB;
        if(cl >= cn)
            continue;
        const float* src
            = a.partial + ((g * taps + tap) * a.kg + k) * static_cast<long long>(a.cg) + c0 + cl;
        float sum = 0.0f;
        for(int sp = 0; sp < a.splits; ++sp)
            sum += __builtin_nontemporal_load(src + sp * a.partialElems);
        tile[cl * taps + tap] = sum;
    }
    __syncthreads();
    T* o = static_cast<T*>(a.dw) + gk * a.wk + static_cast<long long>(c0) * taps;
    for(int e = threadIdx.x; e < cn * taps; e += blockDim.x)
    {
        const int cl = e / taps, tap = e - cl * taps;
        o[cl * taps + (a.flip ? taps - 1 - tap : tap)] = K1Elem<T>::store(tile[e]);
    }
}

/// Adds the splits' partials into dw (its element type and layout) in a fixed order: a block's
/// 256 threads are 64 four-channel quads of dw by 4 split groups; group j adds splits j, j + 4,
/// j + 8, ... in order, and the four group sums are added in group order through LDS. The
/// association depends only on the split count, so the result is bitwise reproducible. (A
/// separate entry from the row reduce: sharing one kernel would give this one the row tile's
/// LDS and cut its occupancy.)
template <class T>
__device__ inline void k3Reduce(const K3ReduceArgs& a)
{
    typedef float f4 __attribute__((ext_vector_type(4)));
    __shared__ f4 part[3][64];
    T* dw = static_cast<T*>(a.dw);
    const int cq = (a.cg + 3) / 4;
    const long long quads = a.elems / a.cg * cq;
    const bool vec = (a.cg % 4) == 0;
    const int lane = threadIdx.x % 64, grp = threadIdx.x / 64;
    const long long i = static_cast<long long>(blockIdx.x) * 64 + lane;
    const bool live = i < quads;
    // i over [g][tap][k][c / 4] (the partials' order), c fastest.
    const int c = live ? static_cast<int>(i % cq) * 4 : 0;
    const long long r1 = live ? i / cq : 0;
    const float* src = a.partial + r1 * a.cg + c;
    f4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
    if(live)
    {
        if(vec)
        {
            int sp = grp;
            for(; sp + 12 < a.splits; sp += 16)
            {
                f4 v[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    v[j] = __builtin_nontemporal_load(
                        reinterpret_cast<const f4*>(src + (sp + 4 * j) * a.partialElems));
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    sum += v[j];
            }
            for(; sp < a.splits; sp += 4)
                sum += __builtin_nontemporal_load(
                    reinterpret_cast<const f4*>(src + sp * a.partialElems));
        }
        else
        {
            for(int sp = grp; sp < a.splits; sp += 4)
                for(int j = 0; j < 4 && c + j < a.cg; ++j)
                    sum[j] += src[sp * a.partialElems + j];
        }
    }
    if(grp > 0)
        part[grp - 1][lane] = sum;
    __syncthreads();
    if(grp != 0 || !live)
        return;
    sum += part[0][lane];
    sum += part[1][lane];
    sum += part[2][lane];
    const int k = static_cast<int>(r1 % a.kg);
    const long long r2 = r1 / a.kg;
    const int tap = static_cast<int>(r2 % a.taps);
    const int g = static_cast<int>(r2 / a.taps);
    const int wtap = a.flip ? a.taps - 1 - tap : tap;
    T* o = dw + (static_cast<long long>(g) * a.kg + k) * a.wk + wtap * a.wtap;
    for(int j = 0; j < 4 && c + j < a.cg; ++j)
        o[(c + j) * a.wc] = K1Elem<T>::store(sum[j]);
}

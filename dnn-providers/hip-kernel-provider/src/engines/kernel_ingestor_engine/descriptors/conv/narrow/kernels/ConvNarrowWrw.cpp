// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The narrow backward-weights kernels (CONV_NARROW_WRW): dw of a layer with at most 8 channels on
// one side -- an output layer (K <= 8 of dy against C of x) or a stem (C <= 8 of x against K of
// dy). The wide operand (`in`: x for an output layer, dy for a stem) is the traffic; dw is tiny.
//
// The taps go on the GEMM's N axis: per wide channel m and column (t, j, r, s) -- a narrow channel
// j and a tap --
//     D[m][t, j, r, s] = sum_q in[q][m] nar[map(q, t, r, s)][j]
// over the wide operand's pixels q, where map pairs q with the narrow pixel the tap reaches
// (CW_MAPB 0, a stem: q is a dy pixel, nar = x at q * stride - pad + tap * dilation; CW_MAPB 1, an
// output layer: q is an x pixel, nar = dy at (q + pad - tap * dilation) / stride where exact).
// M = the wide channels, N = KO x taps (81 for an RGB head's 3x3x3, at 84% use of six 16-wide
// tiles), the reduction the pixels: the wide operand is read once.
//
// A block owns a block of wide channels and a contiguous run of the reduction's steps (image, in
// plane, band of RB rows, strip of WR columns; strip fastest). Per step it stages the narrow
// window the step's taps reach (KO x FD x FR x RB rows of CW_XW narrow pixels, loaded into
// registers a step ahead) in LDS, expands it there into the B operand -- every column's narrow
// values at the step's KR = RB WR pixels, zero where the tap reaches no narrow pixel or the pixel
// lies outside in -- while the lanes' A fragments (KPL consecutive pixels of a wide channel) come
// straight from memory a step ahead (double-buffered), and multiplies on K1's matrix-core atoms
// (ConvIgemmCore.h), or on the vector ALU (CW_VALU: a lane a wide channel, the B operand broadcast
// from LDS). The block's sums go to fp32 partials
// [split][wide channel][column]; ConvNarrowWrwReduce adds the splits in order and writes dw in its
// element type, in the graph's taps (convolution mode reflected): deterministic, no atomics.
//
// Infinities and NaN follow an explicitly padded GEMM (im2col with x's padding as zeros, as
// PyTorch, the BLAS engines and CONV_WRW compute it): a stem's (CW_MAPB 0) wide dy pixel meets x's
// padding as inf x 0 (NaN). An output layer's (CW_MAPB 1) wide x pixel meets only the taps that
// reach a dy pixel -- the B operand's zeros where a tap reaches none (beyond dy, or a stride phase
// between dy pixels) are no products, not padding. The matrix cores multiply every column, so a
// step whose A fragments hold a non-finite value (a wave-wide vote on the loaded fragments) runs
// them with those values zeroed and adds the non-finite values' products on the taps that reach
// a dy pixel, lane by lane; the vector ALU skips those taps for a non-finite x value.
//
// One HIPRTC compile per config; every extent, stride, padding, dilation and channel count is a
// runtime argument (ConvNarrowArgs.h). Defines:
//   CW_T       element type: float, _Float16, or unsigned short (bf16 bits)
//   CW_VALU    1: vector-ALU products (lanes a wave CW_WAVE); 0: matrix-core atom CW_ATOM
//   CW_KO      narrow channels the config serves (ko <= KO)
//   CW_FD, CW_FR, CW_FS   filter depth, rows, columns
//   CW_MAPB    1: in = x, nar = dy (an output layer); 0: in = dy, nar = x (a stem)
//   CW_NCHW    1: channels-first in and nar; 0: channels-last
//   CW_AV      1: channels-first in with whole KPL-pixel runs (vector loads along a row)
//   CW_WR      in columns a step;  CW_RB  in rows a step;  CW_NW  waves a block
//   CW_XW      narrow pixels a window row (the host's: the step's columns, every filter column)
//   CW_MPW     row tiles (DIM wide channels) a wave; CW_VALU: column groups (a block's threads
//              split into MPW groups of whole waves, each a column group of every wide channel)

#include "ConvIgemmCore.h"
#include "ConvNarrowArgs.h"

#ifndef CW_T
#error "CW_T must name the element type"
#endif
#ifndef CW_VALU
#define CW_VALU 0
#endif
#ifndef CW_NCHW
#define CW_NCHW 0
#endif
#ifndef CW_AV
#define CW_AV 0
#endif
#ifndef CW_MAPB
#define CW_MAPB 0
#endif

namespace
{

typedef CW_T T;
constexpr int ES = static_cast<int>(sizeof(T));
constexpr int KO = CW_KO;
constexpr int FD = CW_FD;
constexpr int FR = CW_FR;
constexpr int FS = CW_FS;
constexpr int WR = CW_WR;
constexpr int RB = CW_RB;
constexpr int NW = CW_NW;
constexpr int MPW = CW_MPW;
constexpr bool MAPB = CW_MAPB != 0;
constexpr bool AV = CW_AV != 0;
constexpr int TAPS = FD * FR * FS;
constexpr int NCOL = KO * TAPS;
constexpr int KR = RB * WR; // reduction pixels a step

#if CW_VALU
constexpr int WAVE = CW_WAVE;
constexpr int NT = NW * WAVE;
constexpr int CG = MPW; // column groups
static_assert(NW % CG == 0, "whole waves a column group");
constexpr int CPB = NT / CG; // wide channels a block (a thread a channel of its group)
constexpr int CPT = (NCOL + CG - 1) / CG; // columns a thread
constexpr int NCP = CG * CPT;
typedef float BT; // the B operand in LDS
#else
typedef CW_ATOM Atom;
constexpr int WAVE = Atom::WAVE;
constexpr int NT = NW * WAVE;
constexpr int DIM = Atom::DIM;
constexpr int KA = Atom::KA;
constexpr int KPL = Atom::KPL;
static_assert(KR % KA == 0 && WR % KPL == 0, "whole k-steps a step, whole lane runs a row");
constexpr int KSR = KR / KA; // k-steps a step
constexpr int NTL = (NCOL + DIM - 1) / DIM;
constexpr int NCP = NTL * DIM;
constexpr int CPB = MPW * NW * DIM; // wide channels a block
typedef typename Atom::frag Frag;
typedef typename Atom::acc Acc;
typedef T BT;
#endif
#ifdef CW_NCP
static_assert(CW_NCP == NCP, "the host's columns");
#endif
constexpr int BPITCH = KR + 16 / static_cast<int>(sizeof(BT)); // a column's pixels in LDS
constexpr int XW = CW_XW;
constexpr int WROWS = KO * FD * FR * RB; // window rows: (j, t, r, jr)
constexpr int WE = WROWS * XW; // window elements
constexpr int WPT = (WE + NT - 1) / NT; // window elements a thread
constexpr int EPT = (NCP * KR + NT - 1) / NT; // B elements a thread

constexpr unsigned OOB = K1_OUT_OF_RANGE;

/// The exponent bits of an element type (all set: infinite or NaN).
template <class X>
struct ExpBits;
template <>
struct ExpBits<float>
{
    typedef unsigned raw;
    static constexpr unsigned MASK = 0x7f800000u;
};
template <>
struct ExpBits<k1_half>
{
    typedef unsigned short raw;
    static constexpr unsigned MASK = 0x7c00u;
};
template <>
struct ExpBits<k1_bf16>
{
    typedef unsigned short raw;
    static constexpr unsigned MASK = 0x7f80u;
};

/// Whether @p v is infinite or NaN.
__device__ inline bool nonFinite(T v)
{
    const unsigned bits = __builtin_bit_cast(typename ExpBits<T>::raw, v);
    return (bits & ExpBits<T>::MASK) == ExpBits<T>::MASK;
}

__device__ inline int floorDiv(int a, int b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

/// One element at byte offset @p byteOffset (out of range: zero).
__device__ inline T loadElem(__amdgpu_buffer_rsrc_t r, unsigned byteOffset)
{
    return __builtin_bit_cast(T, k1BufferLoad<typename K1Raw<ES>::type>(r, byteOffset));
}

/// NB bytes from @p byteOffset as raw dwords (NB a multiple of 4, at most 32).
template <int NB>
__device__ inline void
    loadBytes(__amdgpu_buffer_rsrc_t r, unsigned byteOffset, unsigned (&d)[NB / 4])
{
    static_assert(NB % 4 == 0 && NB <= 32, "dword runs of at most 32 bytes");
    if constexpr(NB >= 16)
    {
#pragma unroll
        for(int h = 0; h < NB / 16; ++h)
        {
            typedef unsigned U4 __attribute__((ext_vector_type(4)));
            const U4 v = __builtin_bit_cast(
                U4, __builtin_amdgcn_raw_buffer_load_b128(r, byteOffset + 16u * h, 0, 0));
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                d[4 * h + i] = v[i];
            }
        }
    }
    else if constexpr(NB == 8)
    {
        typedef unsigned U2 __attribute__((ext_vector_type(2)));
        const U2 v
            = __builtin_bit_cast(U2, __builtin_amdgcn_raw_buffer_load_b64(r, byteOffset, 0, 0));
        d[0] = v[0];
        d[1] = v[1];
    }
    else
    {
        d[0] = __builtin_amdgcn_raw_buffer_load_b32(r, byteOffset, 0, 0);
    }
}

#if !CW_VALU
/// A lane's A fragment: KPL pixels of a wide channel from element offset @p base (pixel stride
/// @p ps), the first @p avail of them in the row (zero past it), or none when @p ok is false.
template <int L = KPL>
__device__ inline Frag loadRun(__amdgpu_buffer_rsrc_t r, int base, int ps, int avail, bool ok)
{
    typedef typename K1If<(L > 0), Frag, Frag>::type FragL; // dependent on L
    if constexpr(AV && L * ES >= 4)
    {
        // Channels-first rows of whole runs: KPL pixels, all in or all past the row.
        unsigned d[L * ES / 4];
        loadBytes<L * ES>(r, ok && avail > 0 ? static_cast<unsigned>(base * ES) : OOB, d);
        return __builtin_bit_cast(FragL, d);
    }
    else
    {
        FragL f;
#pragma unroll
        for(int e = 0; e < L; ++e)
        {
            const bool in = ok && e < avail;
            const T v = loadElem(r, in ? static_cast<unsigned>((base + e * ps) * ES) : OOB);
            if constexpr(L == 1)
            {
                f = __builtin_bit_cast(FragL, v);
            }
            else
            {
                f[e] = __builtin_bit_cast(typename Atom::elem, v);
            }
        }
        return f;
    }
}

/// A fragment's KPL elements.
template <int L = KPL>
struct FragElems
{
    T e[L];
};

/// Whether any element of @p f is infinite or NaN.
__device__ inline bool fragNonFinite(const Frag& f)
{
    const auto x = __builtin_bit_cast(FragElems<>, f);
    bool any = false;
#pragma unroll
    for(int e = 0; e < KPL; ++e)
    {
        any = any || nonFinite(x.e[e]);
    }
    return any;
}

/// @p f with its infinite and NaN elements zeroed.
__device__ inline Frag finitePart(const Frag& f)
{
    auto x = __builtin_bit_cast(FragElems<>, f);
#pragma unroll
    for(int e = 0; e < KPL; ++e)
    {
        x.e[e] = nonFinite(x.e[e]) ? K1Elem<T>::store(0.0f) : x.e[e];
    }
    return __builtin_bit_cast(Frag, x);
}

/// Whether any lane of the wave has @p v.
template <int WV = WAVE>
__device__ inline bool anyLane(bool v)
{
    if constexpr(WV == 64)
    {
        return __builtin_amdgcn_ballot_w64(v) != 0;
    }
    else
    {
        return __builtin_amdgcn_ballot_w32(v) != 0;
    }
}

/// The B fragment at LDS @p p: KPL consecutive elements, aligned to their size.
template <int NB = KPL * ES>
__device__ inline Frag loadFrag(const T* p)
{
    typedef typename K1If<(NB > 0), Frag, Frag>::type FragB; // dependent on NB
    if constexpr(NB <= 16)
    {
        typedef typename K1Raw<NB>::type Raw;
        return __builtin_bit_cast(FragB, *reinterpret_cast<const Raw*>(p));
    }
    else
    {
        typedef unsigned U4 __attribute__((ext_vector_type(4)));
        typedef unsigned U8 __attribute__((ext_vector_type(8)));
        const U4 lo = reinterpret_cast<const U4*>(p)[0];
        const U4 hi = reinterpret_cast<const U4*>(p)[1];
        const U8 v{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
        return __builtin_bit_cast(FragB, v);
    }
}
#endif

/// The narrow coordinate tap index @p k along one axis reaches from wide coordinate @p q (-1:
/// none): CW_MAPB 0, q * s - p + k * d; 1, (q + p - k * d) / s where exact.
__device__ inline int mapAxis(int q, int k, int s, int p, int d, int extent)
{
    int v;
    if constexpr(MAPB)
    {
        const int num = q + p - k * d;
        v = floorDiv(num, s);
        if(v * s != num)
        {
            return -1;
        }
    }
    else
    {
        v = q * s - p + k * d;
    }
    return v >= 0 && v < extent ? v : -1;
}

/// A block: its wide channels, steps and the reduction's step geometry.
struct Wrw
{
    const ConvNarrowWrwArgs& a;
    BT* sB;
    T* sWin;
    T wreg[WPT];
    int tid;
    int c0; ///< first wide channel of the block
    int s0; ///< the block's steps [s0, s1)
    int s1;
#if !CW_VALU
    int lane;
    int wave;
    int col;
    int kb;
    Acc d[MPW][NTL];
    Frag abuf[2][MPW][KSR];
#else
    int group;
    float acc[CPT];
#endif

    __device__ Wrw(const ConvNarrowWrwArgs& args, BT* b, T* win)
        : a(args)
        , sB(b)
        , sWin(win)
    {
    }

    /// Step @p st's image, in plane, first row and first column.
    __device__ void stepAt(int st, int& img, int& z, int& q0, int& iw0) const
    {
        const int strip = st % a.strips;
        int rest = st / a.strips;
        const int band = rest % a.bands;
        rest /= a.bands;
        z = rest % a.di;
        img = rest / a.di;
        q0 = band * RB;
        iw0 = strip * WR;
    }

    /// CW_MAPB 1: whether column @p n (t, j, r, s) at the step's pixel @p k reaches a dy pixel
    /// (the B operand holds a dy value there, not a zero for a missing product).
    __device__ bool reaches(int n, int k, int z, int q0, int iw0) const
    {
        const int s = n % FS;
        const int r = (n / FS) % FR;
        const int j = (n / (FS * FR)) % KO;
        const int t = n / (FS * FR * KO);
        const int jr = k / WR;
        const int q = q0 + jr;
        const int w = iw0 + k - jr * WR;
        return n < NCOL && j < a.ko && q < a.hi && w < a.wi
               && mapAxis(z, t, a.sd, a.pd, a.dd, a.dn) >= 0
               && mapAxis(q, r, a.sh, a.ph, a.dh, a.hn) >= 0
               && mapAxis(w, s, a.sw, a.pw, a.dw2, a.wn) >= 0;
    }

    /// The narrow window step @p st's taps reach into registers: element e = (((j FD + t) FR +
    /// r) RB + jr) XW + x, narrow channel j, depth tap t, filter row r, step row jr, window column
    /// x; zero where no narrow pixel is.
    __device__ void loadWindow(int st)
    {
        int img, z, q0, iw0;
        stepAt(st, img, z, q0, iw0);
        const T* nar = static_cast<const T*>(a.nar) + img * a.ns[0];
        const int colBase
            = MAPB ? floorDiv(iw0 + a.pw - (FS - 1) * a.dw2, a.sw) : iw0 * a.sw - a.pw;
#pragma unroll
        for(int i = 0; i < WPT; ++i)
        {
            const int e = tid + i * NT;
            const int x = e % XW;
            const int row = e / XW;
            const int jr = row % RB;
            const int r = (row / RB) % FR;
            const int t = (row / (RB * FR)) % FD;
            const int j = row / (RB * FR * FD);
            T v = K1Elem<T>::store(0.0f);
            if(e < WE && j < a.ko)
            {
                const int nz = mapAxis(z, t, a.sd, a.pd, a.dd, a.dn);
                const int nh = mapAxis(q0 + jr, r, a.sh, a.ph, a.dh, a.hn);
                const int nw = colBase + x;
                if(nz >= 0 && nh >= 0 && nw >= 0 && nw < a.wn)
                {
                    v = nar[j * a.ns[1] + nz * a.ns[2] + nh * a.ns[3] + nw * a.ns[4]];
                }
            }
            wreg[i] = v;
        }
    }

    /// The registers' window into LDS.
    __device__ void storeWindow()
    {
#pragma unroll
        for(int i = 0; i < WPT; ++i)
        {
            const int e = tid + i * NT;
            if(e < WE)
            {
                sWin[e] = wreg[i];
            }
        }
    }

    /// The window expanded into the B operand: column n = (t, j, r, s), pixel k = (jr, u) ->
    /// sB[n][k].
    __device__ void expand(int st)
    {
        int img, z, q0, iw0;
        stepAt(st, img, z, q0, iw0);
        const int colBase
            = MAPB ? floorDiv(iw0 + a.pw - (FS - 1) * a.dw2, a.sw) : iw0 * a.sw - a.pw;
#pragma unroll 4
        for(int i = 0; i < EPT; ++i)
        {
            const int e = tid + i * NT;
            const int n = e / KR;
            const int k = e - n * KR;
            const int s = n % FS;
            const int r = (n / FS) % FR;
            const int j = (n / (FS * FR)) % KO;
            const int t = n / (FS * FR * KO);
            const int jr = k / WR;
            const int u = k - jr * WR;
            float v = 0.0f;
            if(n < NCOL && q0 + jr < a.hi && iw0 + u < a.wi)
            {
                int x;
                bool ok = true;
                if constexpr(MAPB)
                {
                    const int num = iw0 + u + a.pw - s * a.dw2;
                    const int d = floorDiv(num, a.sw);
                    ok = d * a.sw == num;
                    x = d - colBase;
                }
                else
                {
                    x = u * a.sw + s * a.dw2;
                }
                if(ok)
                {
                    v = K1Elem<T>::load(sWin[(((j * FD + t) * FR + r) * RB + jr) * XW + x]);
                }
            }
            if(e < NCP * KR)
            {
#if CW_VALU
                sB[n * BPITCH + k] = v;
#else
                sB[n * BPITCH + k] = K1Elem<T>::store(v); // exact: v came from a T
#endif
            }
        }
    }

#if !CW_VALU
    /// The A fragments of step @p st into buffer B: wide channel c0 + row, KPL pixels a lane.
    template <int B>
    __device__ void loadStep(int st)
    {
        int img, z, q0, iw0;
        stepAt(st, img, z, q0, iw0);
        const char* inImage = static_cast<const char*>(a.in) + img * a.is[0] * ES;
        const __amdgpu_buffer_rsrc_t r = k1Rsrc(inImage, a.inImageBytes);
        const int cs = static_cast<int>(a.is[1]);
        const int ps = static_cast<int>(a.is[4]);
#pragma unroll
        for(int mt = 0; mt < MPW; ++mt)
        {
            const int m = c0 + (mt * NW + wave) * DIM + col;
            const bool mOk = m < a.ci;
#pragma unroll
            for(int ks = 0; ks < KSR; ++ks)
            {
                const int k0 = ks * KA + kb;
                const int jr = k0 / WR;
                const int u = k0 - jr * WR;
                const int q = q0 + jr;
                const bool rowOk = mOk && q < a.hi;
                const int base = m * cs + z * static_cast<int>(a.is[2])
                                 + q * static_cast<int>(a.is[3]) + (iw0 + u) * ps;
                abuf[B][mt][ks] = loadRun(r, base, ps, a.wi - (iw0 + u), rowOk);
            }
        }
    }

    /// The B fragment of column tile @p nt, k-step @p ks from LDS.
    __device__ Frag loadB(int nt, int ks) const
    {
        return loadFrag(sB + (nt * DIM + col) * BPITCH + ks * KA + kb);
    }

    /// CW_MAPB 1, a step with non-finite A values (run with them zeroed): their products on the
    /// columns that reach a dy pixel, into each lane's accumulators.
    __device__ void addNonFinite(int st)
    {
        int img, z, q0, iw0;
        stepAt(st, img, z, q0, iw0);
        const char* inImage = static_cast<const char*>(a.in) + img * a.is[0] * ES;
        const __amdgpu_buffer_rsrc_t r = k1Rsrc(inImage, a.inImageBytes);
        const int cs = static_cast<int>(a.is[1]);
        const int ps = static_cast<int>(a.is[4]);
        for(int mt = 0; mt < MPW; ++mt)
        {
            for(int i = 0; i < Atom::ACC; ++i)
            {
                const int m = c0 + (mt * NW + wave) * DIM + Atom::dRow(lane, i);
                if(m >= a.ci)
                {
                    continue;
                }
                for(int k = 0; k < KR; ++k)
                {
                    const int jr = k / WR;
                    const int q = q0 + jr;
                    const int w = iw0 + k - jr * WR;
                    if(q >= a.hi || w >= a.wi)
                    {
                        continue;
                    }
                    const T xr
                        = loadElem(r,
                                   static_cast<unsigned>((m * cs + z * static_cast<int>(a.is[2])
                                                          + q * static_cast<int>(a.is[3]) + w * ps)
                                                         * ES));
                    if(!nonFinite(xr))
                    {
                        continue;
                    }
                    const float xv = K1Elem<T>::load(xr);
                    for(int nt = 0; nt < NTL; ++nt)
                    {
                        const int n = nt * DIM + col;
                        if(reaches(n, k, z, q0, iw0))
                        {
                            d[mt][nt][i] += xv * K1Elem<T>::load(sB[n * BPITCH + k]);
                        }
                    }
                }
            }
        }
    }
#endif

    /// One step on buffer B: its window (in registers) into LDS, the next step's window and A
    /// loads issued, the B operand expanded, the products.
    template <int B>
    __device__ void step(int st)
    {
        __syncthreads(); // the previous step's window and B reads are done
        storeWindow();
        if(st + 1 < s1)
        {
            loadWindow(st + 1);
#if !CW_VALU
            loadStep<1 - B>(st + 1);
#endif
        }
        __syncthreads();
        expand(st);
        __syncthreads();
#if CW_VALU
        int img, z, q0, iw0;
        stepAt(st, img, z, q0, iw0);
        const char* inImage = static_cast<const char*>(a.in) + img * a.is[0] * ES;
        const __amdgpu_buffer_rsrc_t r = k1Rsrc(inImage, a.inImageBytes);
        const int cs = static_cast<int>(a.is[1]);
        const int ps = static_cast<int>(a.is[4]);
        const int m = c0 + tid % CPB;
        const bool mOk = m < a.ci;
        const BT* bg = sB + group * CPT * BPITCH; // the group's columns: wave-uniform reads
        for(int jr = 0; jr < RB; ++jr)
        {
            const int q = q0 + jr;
            const int rowOff
                = m * cs + z * static_cast<int>(a.is[2]) + q * static_cast<int>(a.is[3]) + iw0 * ps;
            const bool rowOk = mOk && q < a.hi;
            for(int u = 0; u < WR; ++u)
            {
                const bool ok = rowOk && iw0 + u < a.wi;
                const T xr = loadElem(r, ok ? static_cast<unsigned>((rowOff + u * ps) * ES) : OOB);
                const float xv = K1Elem<T>::load(xr);
                const int k = jr * WR + u;
                if constexpr(MAPB)
                {
                    if(nonFinite(xr))
                    {
                        // Only the taps that reach a dy pixel (see the file comment).
                        for(int n = 0; n < CPT; ++n)
                        {
                            if(reaches(group * CPT + n, k, z, q0, iw0))
                            {
                                acc[n] = __builtin_fmaf(xv, bg[n * BPITCH + k], acc[n]);
                            }
                        }
                        continue;
                    }
                }
#pragma unroll
                for(int n = 0; n < CPT; ++n)
                {
                    acc[n] = __builtin_fmaf(xv, bg[n * BPITCH + k], acc[n]);
                }
            }
        }
#else
        bool special = false;
        if constexpr(MAPB)
        {
            bool nf = false;
#pragma unroll
            for(int mt = 0; mt < MPW; ++mt)
            {
#pragma unroll
                for(int ks = 0; ks < KSR; ++ks)
                {
                    nf = nf || fragNonFinite(abuf[B][mt][ks]);
                }
            }
            special = anyLane(nf); // wave-uniform
            if(special)
            {
#pragma unroll
                for(int mt = 0; mt < MPW; ++mt)
                {
#pragma unroll
                    for(int ks = 0; ks < KSR; ++ks)
                    {
                        abuf[B][mt][ks] = finitePart(abuf[B][mt][ks]);
                    }
                }
            }
        }
#pragma unroll
        for(int ks = 0; ks < KSR; ++ks)
        {
#pragma unroll
            for(int nt = 0; nt < NTL; ++nt)
            {
                const Frag bf = loadB(nt, ks);
#pragma unroll
                for(int mt = 0; mt < MPW; ++mt)
                {
                    d[mt][nt] = Atom::mma(abuf[B][mt][ks], bf, d[mt][nt]);
                }
            }
        }
        if constexpr(MAPB)
        {
            if(special)
            {
                addNonFinite(st);
            }
        }
#endif
    }
};

} // namespace

/// Grid: a.splits x a.cblocks blocks of NT threads. See the file comment.
extern "C" __global__ void __launch_bounds__(NT) ConvNarrowWrw(ConvNarrowWrwArgs a)
{
    __shared__ __attribute__((aligned(16))) BT sB[NCP * BPITCH];
    __shared__ T sWin[WE];
    Wrw w(a, sB, sWin);
    w.tid = static_cast<int>(threadIdx.x);
    const int split = static_cast<int>(blockIdx.x) % a.splits;
    const int cblock = static_cast<int>(blockIdx.x) / a.splits;
    w.c0 = cblock * CPB;
    w.s0 = static_cast<int>(static_cast<long long>(a.steps) * split / a.splits);
    w.s1 = static_cast<int>(static_cast<long long>(a.steps) * (split + 1) / a.splits);
#if CW_VALU
    w.group = __builtin_amdgcn_readfirstlane(w.tid / CPB);
#pragma unroll
    for(int n = 0; n < CPT; ++n)
    {
        w.acc[n] = 0.0f;
    }
#else
    w.lane = w.tid % WAVE;
    w.wave = __builtin_amdgcn_readfirstlane(w.tid / WAVE);
    w.col = w.lane % DIM;
    w.kb = Atom::template kBase<KPL>(w.lane);
#pragma unroll
    for(int mt = 0; mt < MPW; ++mt)
    {
#pragma unroll
        for(int nt = 0; nt < NTL; ++nt)
        {
#pragma unroll
            for(int i = 0; i < Atom::ACC; ++i)
            {
                w.d[mt][nt][i] = 0.0f;
            }
        }
    }
    if(w.s0 < w.s1)
    {
        w.loadStep<0>(w.s0);
    }
#endif
    if(w.s0 < w.s1)
    {
        w.loadWindow(w.s0);
    }
    for(int st = w.s0; st < w.s1; st += 2)
    {
        w.step<0>(st);
        if(st + 1 < w.s1)
        {
            w.step<1>(st + 1);
        }
    }

    // ---- the block's partial sums: part[split][wide channel][column]
#if CW_VALU
    {
        const int m = w.c0 + w.tid % CPB;
        if(m < a.ci)
        {
            float* p = a.part + (static_cast<long long>(split) * a.ci + m) * NCP + w.group * CPT;
#pragma unroll
            for(int n = 0; n < CPT; ++n)
            {
                p[n] = w.acc[n];
            }
        }
    }
#else
#pragma unroll
    for(int mt = 0; mt < MPW; ++mt)
    {
#pragma unroll
        for(int i = 0; i < Atom::ACC; ++i)
        {
            const int m = w.c0 + (mt * NW + w.wave) * DIM + Atom::dRow(w.lane, i);
            if(m < a.ci)
            {
                float* p = a.part + (static_cast<long long>(split) * a.ci + m) * NCP;
#pragma unroll
                for(int nt = 0; nt < NTL; ++nt)
                {
                    p[nt * DIM + w.col] = w.d[mt][nt][i];
                }
            }
        }
    }
#endif
}

/// dw: each element the sum, in split order, of its partials; one thread an element of dw.
extern "C" __global__ void __launch_bounds__(256) ConvNarrowWrwReduce(ConvNarrowWrwArgs a)
{
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if(i >= a.dwElems)
    {
        return;
    }
    // dw element (graph roles) k, c, tap (tg, rg, sg), packed K C T R S order of i.
    const int sg = i % FS;
    const int rg = (i / FS) % FR;
    const int tg = (i / (FS * FR)) % FD;
    const int kc = i / TAPS;
    const int cGraph = a.wideIsC ? a.ci : a.ko; // the graph filter's input channels
    const int c = kc % cGraph;
    const int k = kc / cGraph;
    const int m = a.wideIsC ? c : k; // wide channel
    const int j = a.wideIsC ? k : c; // narrow channel
    const bool reflect = a.flip != 0;
    const int t = reflect ? FD - 1 - tg : tg;
    const int r = reflect ? FR - 1 - rg : rg;
    const int s = reflect ? FS - 1 - sg : sg;
    const int n = ((t * KO + j) * FR + r) * FS + s;
    float sum = 0.0f;
    for(int split = 0; split < a.splits; ++split)
    {
        sum += a.part[(static_cast<long long>(split) * a.ci + m) * NCP + n];
    }
    T* dw = static_cast<T*>(a.dw);
    dw[k * a.ws[0] + c * a.ws[1] + tg * a.ws[2] + rg * a.ws[3] + sg * a.ws[4]]
        = K1Elem<T>::store(sum);
}

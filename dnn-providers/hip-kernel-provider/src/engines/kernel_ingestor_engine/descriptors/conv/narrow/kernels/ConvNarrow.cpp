// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The narrow-output convolution kernels (CONV_NARROW_FWD, CONV_NARROW_BWD): a wide operand
// reduced into at most 8 output channels -- a network's output layer (forward, C channels into
// K <= 8: a VAE's RGB head) or a stem's input gradient (backward data, K channels of dy into
// C <= 8 of dx). Both are memory-bound: the wide tensor is the traffic, and a GEMM tile over
// the narrow channels is mostly padding.
//
// The taps go on the GEMM's N axis instead. Per in pixel p (an x pixel forward, a dy pixel
// backward) the products of every narrow channel j and tap (t, r, s) are
//     P[p][j, t, r, s] = sum_c in[p][c] w'[c][j, t, r, s]
// one GEMM row of N = KO x taps columns (27 for an RGB head's 3x3: two 16-wide tiles at 84% use)
// over the reduction channels. Each output element then gathers its taps' products:
//     forward:        y[o][j] = sum_{t,r,s} P[o * stride - pad + (t, r, s) * dilation][j, t, r, s]
//     backward data: dx[i][j] = sum_{t,r,s} P[(i + pad - (t, r, s) * dilation) / stride][j, t, r, s]
// (the backward term only where the division is exact and the dy pixel exists). Every in pixel
// a block reads is read once from memory (plus the band's row and the strip's column halo), and
// the gather reads each product once.
//
// A block owns a segment of ds output planes, a band of rs output rows and a strip of tw output
// columns of one image. It streams the in rows its band reads, RB rows of WR pixels a step (the
// strip's columns and their halo), plane by plane:
//   - CN_DS 1 (unit depth stride and dilation): every in plane the segment reads, once, with the
//     products of every depth tap, added into a ring of FD output planes' sums; a plane is stored
//     once no later in plane reaches it.
//   - CN_DS 0 (ds 1): per depth tap, the in plane of the block's output plane, with the products
//     of that tap.
// A step's products go through LDS (M = RB WR rows, NCP columns, fp32), and each thread adds the
// products of its OP output pixels in registers. Products come from K1's matrix-core atoms
// (ConvIgemmCore.h: MFMA, WMMA) -- the lanes read their in pixels' channel runs straight from
// memory into registers a step ahead (double-buffered), the filter from LDS -- or from the vector
// ALU (CN_VALU: a lane a pixel, the prepared filter read as wave-uniform values). Every output
// element is summed by one thread in a fixed order and rounded once: deterministic. Products of
// pixels outside in (padding) are never gathered, so an infinite or NaN weight meets padding only
// as the reference does (not at all).
//
// The prepare kernel writes the filter as the products' B operand, per column group (CN_DS 1: one
// group, every depth tap; CN_DS 0: one group a depth tap): [group][NCP][cpad] in the element type
// for the matrix cores (cpad = ci rounded up to the atom's k-step), [group][ci][NCP] fp32 for
// CN_VALU; zero past ci and past the columns. Column (t, j, r, s) = ((t KO + j) FR + r) FS + s
// (t only with CN_DS 1); columns of channels j >= ko are zero. Convolution mode reflects the taps.
//
// One HIPRTC compile per config; every extent, stride, padding, dilation and channel count but
// the width stride is a runtime argument (ConvNarrowArgs.h). Defines:
//   CN_T       element type: float, _Float16, or unsigned short (bf16 bits)
//   CN_VALU    1: vector-ALU products (lanes a wave CN_WAVE); 0: matrix-core atom CN_ATOM
//   CN_KO      narrow channels the config serves (ko <= KO)
//   CN_FD, CN_FR, CN_FS   filter depth, rows, columns;  CN_SW  width stride
//   CN_DS      1: depth streaming (see above)
//   CN_BWD     1: backward data (in = dy, out = dx)
//   CN_NCHW    1: channels-first in and out; 0: channels-last
//   CN_AV      1: in's channels packed and ci whole runs (vector loads of a pixel's channels)
//   CN_WR      pixels a row of products;  CN_RB  in rows a step;  CN_NW  waves a block
//   CN_OP      output pixels a thread
//   CN_CM      reduction channels the LDS filter holds (matrix cores; cpad <= CM)
//   CN_PD      steps the matrix cores' A loads run ahead (CN_PD + 1 register buffers; default 1)
//   CN_BR      1 (matrix cores): the B fragments held in registers for the whole step walk (loaded
//              from the prepared filter once a column group); 0: read from the LDS filter each
//              step. The host's choice: registers where they fit.
//   CN_SF      1 (matrix cores, unit width stride): the filter columns folded into the reduction.
//              A product row is then an output column o, its reduction (s, c) reads in pixel
//              o * 1 - pad + s * dilation (backward data: o + pad - s * dilation), its columns
//              (t, j, r): the gather adds FD FR products an output, and a strip has no column halo.

#include "ConvIgemmCore.h"
#include "ConvNarrowArgs.h"

#ifndef CN_T
#error "CN_T must name the element type"
#endif
#ifndef CN_VALU
#define CN_VALU 0
#endif
#ifndef CN_BWD
#define CN_BWD 0
#endif
#ifndef CN_NCHW
#define CN_NCHW 0
#endif
#ifndef CN_AV
#define CN_AV 0
#endif
#ifndef CN_DS
#define CN_DS 1
#endif
#ifndef CN_PD
#define CN_PD 1
#endif
#ifndef CN_SF
#define CN_SF 0
#endif
#ifndef CN_BR
#define CN_BR 0
#endif

namespace
{

typedef CN_T T;
constexpr int ES = static_cast<int>(sizeof(T));
constexpr int KO = CN_KO;
constexpr int FD = CN_FD;
constexpr int FR = CN_FR;
constexpr int FS = CN_FS;
constexpr int SW = CN_SW;
constexpr int WR = CN_WR;
constexpr int RB = CN_RB;
constexpr int NW = CN_NW;
constexpr int OP = CN_OP;
constexpr bool BWD = CN_BWD != 0;
constexpr bool AV = CN_AV != 0;
constexpr bool DS = CN_DS != 0;
constexpr bool SF = CN_SF != 0;
constexpr int XS = SF ? FS : 1; // filter columns a product row's reduction holds
constexpr int CS = SF ? 1 : FS; // filter columns on the product columns
static_assert(!SF || SW == 1, "folded filter columns at unit width stride");
constexpr int TAPS2 = FR * CS; // products of one narrow channel and depth tap
constexpr int GT = DS ? FD : 1; // depth taps a column group holds
constexpr int NCOL = GT * KO * TAPS2;
constexpr int RING = DS ? FD : 1; // output planes summed at once
constexpr int M = RB * WR; // product rows a step

#if CN_VALU
static_assert(!SF, "the vector ALU products keep the filter columns");
constexpr int WAVE = CN_WAVE;
constexpr int NT = NW * WAVE;
static_assert(M % WAVE == 0 && NT % M == 0, "lanes own whole pixels, waves whole column groups");
constexpr int CG = NT / M; // column groups
constexpr int CPT = (NCOL + CG - 1) / CG; // columns a thread
constexpr int NCP = CG * CPT;
#else
typedef CN_ATOM Atom;
constexpr int WAVE = Atom::WAVE;
constexpr int NT = NW * WAVE;
constexpr int DIM = Atom::DIM;
constexpr int KA = Atom::KA;
constexpr int KPL = Atom::KPL;
static_assert(M % (DIM * NW) == 0, "whole row tiles a wave");
constexpr int MPW = M / DIM / NW; // row tiles a wave
constexpr int NTL = (NCOL + DIM - 1) / DIM; // column tiles
constexpr int NCP = NTL * DIM;
constexpr int CM = CN_CM;
static_assert(CM % KA == 0, "the LDS filter holds whole k-steps");
// A lane's runs of U consecutive k-steps are adjacent in memory: the order of k within an atom is
// free so long as A and B agree, so k-step s at lane run kb reads channels kOf(s) .. + KPL, and a
// channels-packed pixel gives U steps' fragments in one 16-byte load.
constexpr int U = KPL * ES < 16 ? 16 / (KPL * ES) : 1;
static_assert(CM % (KA * U) == 0, "the LDS filter holds whole runs of U k-steps");
constexpr int KSC = CM / KA; // k-steps a filter column
constexpr int KS = XS * KSC; // k-steps a step
constexpr int WPITCH = XS * CM + 16 / ES; // a column's run in LDS, 16 bytes past the channels
constexpr bool BR = CN_BR != 0;
constexpr int GROUPS = DS ? 1 : FD; // column groups of the filter
static_assert(!BR || GROUPS == 1, "registers hold one column group");
constexpr int WLDS = BR ? 1 : GROUPS * NCP * WPITCH; // the LDS filter (none with BR)
typedef typename Atom::frag Frag;
typedef typename Atom::acc Acc;
constexpr int PD = CN_PD;
constexpr int NB = PD + 1; // A register buffers
static_assert(PD >= 1 && PD <= 3, "one to three steps ahead");
#endif
#ifdef CN_NCP
static_assert(CN_NCP == NCP, "the host's product columns");
#endif
constexpr int PITCH = NCP + 1; // odd: the gather's column reads spread over the banks
constexpr int ZR = KO * TAPS2; // zeros ahead of the products, read by taps that reach none
constexpr int SP_SIZE = ZR + 2 * M * PITCH; // two product buffers: one barrier a step
#if CN_VALU
constexpr int NBUF = 1; // steps a round
#else
constexpr int NBUF = NB;
#endif

constexpr unsigned OOB = K1_OUT_OF_RANGE;

template <int V>
struct CnInt
{
    static constexpr int value = V;
};

__device__ inline int floorDiv(int a, int b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}
__device__ inline int ceilDiv(int a, int b)
{
    return -floorDiv(-a, b);
}

__device__ inline float toFloat(T v)
{
    return K1Elem<T>::load(v);
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

#if !CN_VALU
/// The A fragment of the lane's pixel (element offset @p pix within the image, or -1 outside
/// in): KPL reduction channels from @p c0, zero past ci.
template <int L = KPL>
__device__ inline Frag loadA(__amdgpu_buffer_rsrc_t r, int pix, int c0, int ci, int cs)
{
    typedef typename K1If<(L > 0), Frag, Frag>::type FragL; // dependent on L
    if constexpr(AV && L * ES >= 4)
    {
        // Channels packed, ci whole runs: one run of KPL channels (all in, or all past ci).
        const bool ok = pix >= 0 && c0 < ci;
        unsigned d[L * ES / 4];
        loadBytes<L * ES>(r, ok ? static_cast<unsigned>((pix + c0) * ES) : OOB, d);
        return __builtin_bit_cast(FragL, d);
    }
    else
    {
        FragL f;
#pragma unroll
        for(int e = 0; e < L; ++e)
        {
            const int c = c0 + e;
            const bool ok = pix >= 0 && c < ci;
            const T v = loadElem(r, ok ? static_cast<unsigned>((pix + c * cs) * ES) : OOB);
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

/// The B fragment at LDS @p p: KPL consecutive filter elements, aligned to their size.
template <int NB = KPL * ES>
__device__ inline Frag loadB(const T* p)
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

/// The element offset (within the image) of in pixel (z, q, iw), or -1 outside in.
__device__ inline int pixelOffset(int z, int q, int iw, bool rowOk, const ConvNarrowArgs& a)
{
    const bool ok = rowOk && iw >= 0 && iw < a.wi;
    return ok ? static_cast<int>(z * a.is[2] + q * a.is[3] + iw * a.is[4]) : -1;
}

/// A block's geometry and the output pixels its thread owns.
struct Block
{
    int img;
    int od0; ///< the segment's output planes [od0, odN)
    int odN;
    int oh0;
    int ow0;
    int rowsOut;
    int colsOut;
    int iw0; ///< the in column of product column 0
    int qlo; ///< the in rows the band reads
    int qhi;
    int steps; ///< steps a plane
    int orow[OP];
    int ocol[OP];
    bool oval[OP];
    int ucol[OP][FS]; ///< per filter column, the product column a pixel reads (-1: none)
};

/// Adds a step's products (rows q0.. of a plane) of depth tap TAP's columns into ring slot SLOT.
/// Branch-free per lane: a tap that reaches no product reads the zero run past the products.
template <int TAP, int SLOT>
__device__ inline void gatherTap(const float* sP,
                                 int zero,
                                 const Block& b,
                                 const ConvNarrowArgs& a,
                                 int q0,
                                 float (&acc)[RING][OP][KO])
{
#pragma unroll
    for(int jr = 0; jr < RB; ++jr)
    {
        const int q = q0 + jr;
        if(q > b.qhi)
        {
            break;
        }
#pragma unroll
        for(int o = 0; o < OP; ++o)
        {
            const int oh = b.oh0 + b.orow[o];
            int tr = BWD ? oh + a.ph - q * a.sh : q - oh * a.sh + a.ph;
            bool rowOk = b.oval[o] && tr >= 0;
            if(a.dh != 1)
            {
                const int rr = tr / a.dh;
                rowOk = rowOk && rr * a.dh == tr;
                tr = rr;
            }
            rowOk = rowOk && tr < FR;
            const int base = jr * WR * PITCH + (TAP * KO * FR + tr) * CS;
#pragma unroll
            for(int s = 0; s < CS; ++s)
            {
                const int u = SF ? b.ocol[o] : b.ucol[o][s];
                const int p = rowOk && u >= 0 ? base + u * PITCH + s : zero;
#pragma unroll
                for(int j = 0; j < KO; ++j)
                {
                    acc[SLOT][o][j] += sP[p + j * TAPS2];
                }
            }
        }
    }
}

/// Every depth tap's gather (CN_DS 1): tap t adds to ring slot FD - 1 - t (forward) or t
/// (backward data).
template <int TAP>
__device__ inline void gatherTaps(const float* sP,
                                  int zero,
                                  const Block& b,
                                  const ConvNarrowArgs& a,
                                  int q0,
                                  float (&acc)[RING][OP][KO])
{
    if constexpr(TAP < FD)
    {
        gatherTap<TAP, BWD ? TAP : FD - 1 - TAP>(sP, zero, b, a, q0, acc);
        gatherTaps<TAP + 1>(sP, zero, b, a, q0, acc);
    }
}

/// Stores @p sums as output plane @p od, if the block owns it.
__device__ inline void
    storePlane(const Block& b, const ConvNarrowArgs& a, int od, const float (&sums)[OP][KO])
{
    if(od < b.od0 || od >= b.odN)
    {
        return;
    }
    T* outImage = static_cast<T*>(a.out) + b.img * a.os[0];
#pragma unroll
    for(int o = 0; o < OP; ++o)
    {
        if(!b.oval[o])
        {
            continue;
        }
        T* p = outImage + od * a.os[2] + (b.oh0 + b.orow[o]) * a.os[3]
               + (b.ow0 + b.ocol[o]) * a.os[4];
#pragma unroll
        for(int j = 0; j < KO; ++j)
        {
            if(j < a.ko)
            {
                p[j * a.os[1]] = K1Elem<T>::store(sums[o][j]);
            }
        }
    }
}

/// The plane walk. CN_DS 1: plane index pz is in plane zfirst + pz of the segment's window
/// (output plane o reads in planes [o + A, o + A + FD - 1]: forward A = -pd, backward data
/// A = pd - FD + 1; at in plane z the ring holds output planes z - A - FD + 1 + slot). CN_DS 0:
/// plane index pz is depth tap pz of output plane od0.
struct Walk
{
    int A;
    int zfirst;
    int planes;

    __device__ int planeOf(int pz, const Block& b, const ConvNarrowArgs& a) const
    {
        if constexpr(DS)
        {
            return zfirst + pz;
        }
        else if constexpr(BWD)
        {
            const int num = b.od0 + a.pd - pz * a.dd;
            const int z = num >= 0 ? num / a.sd : -1;
            return num >= 0 && z * a.sd == num ? z : -1;
        }
        else
        {
            return b.od0 * a.sd - a.pd + pz * a.dd;
        }
    }

    /// The first plane index from @p pz whose plane has steps (planes when none).
    __device__ int seek(int pz, const Block& b, const ConvNarrowArgs& a) const
    {
        while(pz < planes)
        {
            const int z = planeOf(pz, b, a);
            if(b.steps > 0 && z >= 0 && z < a.di)
            {
                break;
            }
            ++pz;
        }
        return pz;
    }
};

} // namespace

/// Writes the prepared filter; one thread an element. See the file comment.
extern "C" __global__ void __launch_bounds__(256) ConvNarrowPrepare(ConvNarrowArgs a)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if(i >= a.prepared)
    {
        return;
    }
#if CN_VALU
    const int col = static_cast<int>(i % NCP);
    const long long rest = i / NCP;
    const int c = static_cast<int>(rest % a.cpad);
    const int group = static_cast<int>(rest / a.cpad);
#else
    // [group][NCP][XS][cpad]: SF puts the filter column on the reduction.
    const int kk = static_cast<int>(i % (XS * a.cpad));
    const int c = kk % a.cpad;
    const long long rest = i / (XS * a.cpad);
    const int col = static_cast<int>(rest % NCP);
    const int group = static_cast<int>(rest / NCP);
#endif
    float v = 0.0f;
    if(col < NCOL && c < a.ci)
    {
        const int t = DS ? col / (KO * TAPS2) : group;
        const int j = (col / TAPS2) % KO;
        const int r = (col % TAPS2) / CS;
#if CN_VALU
        const int s = col % FS;
#else
        const int s = SF ? kk / a.cpad : col % CS;
#endif
        if(j < a.ko)
        {
            const bool reflect = a.flip != 0;
            const int tt = reflect ? FD - 1 - t : t;
            const int rr = reflect ? FR - 1 - r : r;
            const int ss = reflect ? FS - 1 - s : s;
            // The graph filter's (output, input) channel: forward (j, c), backward data (c, j).
            const int gk = BWD ? c : j;
            const int gc = BWD ? j : c;
            const long long off
                = gk * a.ws[0] + gc * a.ws[1] + tt * a.ws[2] + rr * a.ws[3] + ss * a.ws[4];
            v = toFloat(static_cast<const T*>(a.w)[off]);
        }
    }
#if CN_VALU
    static_cast<float*>(a.wp)[i] = v;
#else
    static_cast<T*>(a.wp)[i] = K1Elem<T>::store(v); // exact: v came from a T
#endif
}

namespace
{

/// The kernel's state: the block, its sums, the plane walk and (matrix cores) the A buffers.
struct Kernel
{
    const ConvNarrowArgs& a;
    float* sP;
    int steps = 0; ///< steps run (the product buffer alternates)
    Block b;
    Walk walk;
    float acc[RING][OP][KO];
    int done = 0; ///< planes whose completion was handled (CN_DS 1)
    __amdgpu_buffer_rsrc_t rsrc;
    int tid;
#if CN_VALU
    int m;
    int group;
#else
    T* sW;
    Frag breg[NTL][KS]; ///< the B fragments (CN_BR; unused otherwise)
    int lane;
    int wave;
    int col;
    int kb;
    int cs;
    Frag abuf[NB][MPW][KS];
    int ip = 0; ///< the next step to load: plane index, first row
    int iq = 0;
#endif

    __device__ Kernel(const ConvNarrowArgs& args, float* products)
        : a(args)
        , sP(products)
    {
    }

    /// Stores the output planes complete before plane index @p p and rotates the ring.
    __device__ void completeThrough(int p)
    {
        if constexpr(DS)
        {
            for(; done < p; ++done)
            {
                storePlane(b, a, walk.zfirst + done - walk.A - FD + 1, acc[0]);
#pragma unroll
                for(int r = 0; r < RING - 1; ++r)
                {
#pragma unroll
                    for(int o = 0; o < OP; ++o)
                    {
#pragma unroll
                        for(int j = 0; j < KO; ++j)
                        {
                            acc[r][o][j] = acc[r + 1][o][j];
                        }
                    }
                }
#pragma unroll
                for(int o = 0; o < OP; ++o)
                {
#pragma unroll
                    for(int j = 0; j < KO; ++j)
                    {
                        acc[RING - 1][o][j] = 0.0f;
                    }
                }
            }
        }
    }

#if !CN_VALU
    /// The filter's B operand, every column group, once before the steps (no loads but the A
    /// fragments' inside the walk, so its waits count only those): into registers (CN_BR, one
    /// group), or into LDS.
    __device__ void stageFilter()
    {
        if constexpr(BR)
        {
            // Every lane's B fragments, from the prepared filter (zero past cpad).
            const T* src = static_cast<const T*>(a.wp);
#pragma unroll
            for(int nt = 0; nt < NTL; ++nt)
            {
#pragma unroll
                for(int k = 0; k < KS; ++k)
                {
                    const int c0 = kOf(k);
                    const T* run = src + ((nt * DIM + col) * XS + k / KSC) * a.cpad + c0;
                    breg[nt][k] = c0 < a.cpad ? loadB(run) : Frag{};
                }
            }
            return;
        }
        constexpr int VE = 16 / ES;
        const T* src = static_cast<const T*>(a.wp);
        // Every k-step of CM runs a filter column (zero past cpad), so the multiplies need no
        // runtime bound.
        const int runs = a.cpad / VE; // cpad: whole k-steps, so whole 16-byte runs
        constexpr int RUNS = CM / VE;
        for(int i = tid; i < GROUPS * NCP * XS * RUNS; i += NT)
        {
            const int n = i / (XS * RUNS); // group x NCP + column
            const int sv = i - n * (XS * RUNS);
            const int sc = sv / RUNS; // filter column (SF)
            const int v = sv - sc * RUNS;
            typedef unsigned U4 __attribute__((ext_vector_type(4)));
            *reinterpret_cast<U4*>(sW + n * WPITCH + sc * CM + v * VE)
                = v < runs ? *reinterpret_cast<const U4*>(src + (n * XS + sc) * a.cpad + v * VE)
                           : U4{0, 0, 0, 0};
        }
        __syncthreads();
    }

    /// The A fragments of step (plane index @p p, rows @p q0..) into buffer B; zeros for every
    /// k-step past the channels, and for every step when @p any is false. Unconditional loads, so
    /// the waits before the multiplies count only the older step's.
    template <int B>
    __device__ void loadStep(int p, int q0, bool any = true)
    {
        const int z = any ? walk.planeOf(p, b, a) : 0;
#pragma unroll
        for(int mt = 0; mt < MPW; ++mt)
        {
            const int mm = (wave * MPW + mt) * DIM + col;
            const int jr = mm / WR;
            const int u = mm - jr * WR; // product column (SF: output column of the strip)
            int pixs[XS];
#pragma unroll
            for(int sc = 0; sc < XS; ++sc)
            {
                const int shift = SF ? (BWD ? a.pw - sc * a.dw : sc * a.dw - a.pw) : 0;
                pixs[sc] = pixelOffset(z, q0 + jr, b.iw0 + u + shift, any && q0 + jr <= b.qhi, a);
            }
#pragma unroll
            for(int k = 0; k < KS; ++k)
            {
                const int pix = pixs[k / KSC];
                {
                    if constexpr(AV && U > 1)
                    {
                        // One 16-byte load: the run's U fragments.
                        if(k % U == 0)
                        {
                            const int c0 = kOf(k);
                            unsigned d[4];
                            loadBytes<16>(rsrc,
                                          pix >= 0 && c0 < a.ci
                                              ? static_cast<unsigned>((pix + c0) * ES)
                                              : OOB,
                                          d);
#pragma unroll
                            for(int v = 0; v < U; ++v)
                            {
                                if(k + v < KS)
                                {
                                    unsigned f[4 / U];
#pragma unroll
                                    for(int i = 0; i < 4 / U; ++i)
                                    {
                                        f[i] = d[v * (4 / U) + i];
                                    }
                                    typedef typename K1If<(B >= 0), Frag, Frag>::type FragD;
                                    abuf[B][mt][k + v] = __builtin_bit_cast(FragD, f);
                                }
                            }
                        }
                    }
                    else
                    {
                        abuf[B][mt][k] = loadA(rsrc, pix, kOf(k), a.ci, cs);
                    }
                }
            }
        }
    }

    /// The first channel of k-step @p k at the lane's run (see U), within its filter column.
    __device__ int kOf(int k) const
    {
        const int kc = k % KSC;
        return (kc / U) * (KA * U) + kb * U + (kc % U) * KPL;
    }

    /// Loads the next step to load into buffer B and moves past it.
    template <int B>
    __device__ void issue()
    {
        const bool any = ip < walk.planes;
        loadStep<B>(ip, iq, any);
        if(any)
        {
            next(ip, iq, ip, iq);
        }
    }
#endif

    /// One step: the products of rows q0.. of plane index p (matrix cores: held in buffer B, the
    /// step PD ahead loaded into buffer (B + PD) mod NB first), into LDS, then the gather.
    template <int B>
    __device__ void step(int p, int q0)
    {
#if CN_VALU
        const int z = walk.planeOf(p, b, a);
        float pv[CPT];
#pragma unroll
        for(int x = 0; x < CPT; ++x)
        {
            pv[x] = 0.0f;
        }
        {
            typedef const __attribute__((address_space(4))) float* ConstF;
            const int jr = m / WR;
            const int pix = pixelOffset(z, q0 + jr, b.iw0 + (m - jr * WR), q0 + jr <= b.qhi, a);
            const ConstF wf = __builtin_bit_cast(ConstF, static_cast<const float*>(a.wp))
                              + (static_cast<long long>(DS ? 0 : p) * a.cpad) * NCP + group * CPT;
            if constexpr(AV)
            {
                constexpr int VE = 16 / ES; // channels a 16-byte load
                for(int c0 = 0; c0 < a.ci; c0 += VE)
                {
                    unsigned d[4];
                    loadBytes<16>(rsrc, pix >= 0 ? static_cast<unsigned>((pix + c0) * ES) : OOB, d);
                    T xs[VE];
                    __builtin_memcpy(xs, d, 16);
#pragma unroll
                    for(int e = 0; e < VE; ++e)
                    {
                        const float xv = toFloat(xs[e]);
                        const ConstF wr = wf + (c0 + e) * NCP;
#pragma unroll
                        for(int x = 0; x < CPT; ++x)
                        {
                            pv[x] = __builtin_fmaf(xv, wr[x], pv[x]);
                        }
                    }
                }
            }
            else
            {
                const int cs = static_cast<int>(a.is[1]);
                for(int c = 0; c < a.ci; ++c)
                {
                    const float xv = toFloat(loadElem(
                        rsrc, pix >= 0 ? static_cast<unsigned>((pix + c * cs) * ES) : OOB));
                    const ConstF wr = wf + c * NCP;
#pragma unroll
                    for(int x = 0; x < CPT; ++x)
                    {
                        pv[x] = __builtin_fmaf(xv, wr[x], pv[x]);
                    }
                }
            }
        }
#else
        issue<(B + PD) % NB>();
        Acc d[MPW][NTL];
#pragma unroll
        for(int mt = 0; mt < MPW; ++mt)
        {
#pragma unroll
            for(int nt = 0; nt < NTL; ++nt)
            {
#pragma unroll
                for(int i = 0; i < Atom::ACC; ++i)
                {
                    d[mt][nt][i] = 0.0f;
                }
            }
        }
#pragma unroll
        for(int k = 0; k < KS; ++k)
        {
#pragma unroll
            for(int nt = 0; nt < NTL; ++nt)
            {
                const Frag bf = BR ? breg[nt][k]
                                   : loadB(sW + ((DS ? 0 : p) * NCP + nt * DIM + col) * WPITCH
                                           + (k / KSC) * CM + kOf(k));
#pragma unroll
                for(int mt = 0; mt < MPW; ++mt)
                {
                    d[mt][nt] = Atom::mma(abuf[B][mt][k], bf, d[mt][nt]);
                }
            }
        }
#endif
        // ---- through LDS: the step's own buffer (the other one's gather may still run; the
        // barrier below orders it before the next write to this one)
        const int base = ZR + (steps++ & 1) * M * PITCH;
        float* pb = sP + base;
#if CN_VALU
#pragma unroll
        for(int x = 0; x < CPT; ++x)
        {
            pb[m * PITCH + group * CPT + x] = pv[x];
        }
#else
#pragma unroll
        for(int mt = 0; mt < MPW; ++mt)
        {
#pragma unroll
            for(int nt = 0; nt < NTL; ++nt)
            {
#pragma unroll
                for(int i = 0; i < Atom::ACC; ++i)
                {
                    const int mm = (wave * MPW + mt) * DIM + Atom::dRow(lane, i);
                    pb[mm * PITCH + nt * DIM + col] = d[mt][nt][i];
                }
            }
        }
#endif
        __syncthreads();
        // ---- the gather
        if constexpr(DS)
        {
            gatherTaps<0>(pb, -base, b, a, q0, acc);
        }
        else
        {
            gatherTap<0, 0>(pb, -base, b, a, q0, acc);
        }
    }

    /// The step after (p, q0): (np, nq).
    __device__ void next(int p, int q0, int& np, int& nq) const
    {
        np = p;
        nq = q0 + RB;
        if(nq > b.qhi)
        {
            np = walk.seek(p + 1, b, a);
            nq = b.qlo;
        }
    }

    /// Runs step (p, q0) on buffer B (the filter group and the ring first) and moves past it;
    /// false when no step is left.
    template <int B>
    __device__ bool turn(int& p, int& q0)
    {
        if(p >= walk.planes)
        {
            return false;
        }
        completeThrough(p);
        step<B>(p, q0);
        next(p, q0, p, q0);
        return true;
    }

    /// The steps, NB a round so the A buffers are named at compile time.
    template <int B>
    __device__ bool round(int& p, int& q0)
    {
        if constexpr(B < NBUF)
        {
            return turn<B>(p, q0) && round<B + 1>(p, q0);
        }
        else
        {
            return true;
        }
    }

#if !CN_VALU
    /// Loads the first PD steps into buffers 0 .. PD - 1.
    template <int B>
    __device__ void prime()
    {
        if constexpr(B < PD)
        {
            issue<B>();
            prime<B + 1>();
        }
    }
#endif
};

} // namespace

/// Grid: a.blocks blocks of NT threads. See the file comment.
extern "C" __global__ void __launch_bounds__(NT) ConvNarrow(ConvNarrowArgs a)
{
    __shared__ float sP[SP_SIZE];
#if !CN_VALU
    __shared__ __attribute__((aligned(16))) T sW[WLDS];
#endif
    Kernel k(a, sP);
    Block& b = k.b;
    {
        int blk = static_cast<int>(blockIdx.x);
        const int seg = blk % a.segs;
        blk /= a.segs;
        const int strip = blk % a.strips;
        blk /= a.strips;
        const int band = blk % a.bands;
        b.img = blk / a.bands;
        b.od0 = seg * a.ds;
        b.odN = min(b.od0 + a.ds, a.dOut);
        b.ow0 = strip * a.tw;
        b.oh0 = band * a.rs;
    }
    b.rowsOut = min(a.rs, a.ho - b.oh0);
    b.colsOut = min(a.tw, a.wo - b.ow0);
    if constexpr(BWD)
    {
        b.iw0 = ceilDiv(b.ow0 + a.pw - (FS - 1) * a.dw, SW);
        b.qlo = ceilDiv(b.oh0 + a.ph - (FR - 1) * a.dh, a.sh);
        b.qhi = floorDiv(b.oh0 + b.rowsOut - 1 + a.ph, a.sh);
    }
    else
    {
        b.iw0 = b.ow0 * SW - a.pw;
        b.qlo = b.oh0 * a.sh - a.ph;
        b.qhi = (b.oh0 + b.rowsOut - 1) * a.sh - a.ph + (FR - 1) * a.dh;
    }
    if constexpr(SF)
    {
        b.iw0 = b.ow0; // product columns are output columns; the loads carry the shifts
    }
    b.qlo = max(b.qlo, 0);
    b.qhi = min(b.qhi, a.hi - 1);
    b.steps = b.qhi >= b.qlo ? (b.qhi - b.qlo) / RB + 1 : 0;

    k.tid = static_cast<int>(threadIdx.x);
    for(int i = k.tid; i < ZR; i += NT)
    {
        sP[i] = 0.0f; // read by taps that reach no product
    }
#pragma unroll
    for(int o = 0; o < OP; ++o)
    {
        const int idx = k.tid + o * NT;
        const int r = idx / a.tw;
        int c = idx - r * a.tw;
        if constexpr(BWD && SW > 1)
        {
            // Columns phase-major (tw is whole strides): a wave's lanes share a stride phase, so
            // they agree on which filter columns reach dy, and read consecutive product columns.
            const int per = a.tw / SW;
            c = (c % per) * SW + c / per;
        }
        b.orow[o] = r;
        b.ocol[o] = c;
        b.oval[o] = r < b.rowsOut && c < b.colsOut;
        const int ow = b.ow0 + c;
#pragma unroll
        for(int s = 0; s < FS; ++s)
        {
            int u;
            if constexpr(BWD)
            {
                const int t = ow + a.pw - s * a.dw;
                const int iw = t / SW;
                u = t >= 0 && t - iw * SW == 0 && iw < a.wi ? iw - b.iw0 : -1;
            }
            else
            {
                const int iw = ow * SW - a.pw + s * a.dw;
                u = iw >= 0 && iw < a.wi ? iw - b.iw0 : -1;
            }
            b.ucol[o][s] = u;
        }
#pragma unroll
        for(int r2 = 0; r2 < RING; ++r2)
        {
#pragma unroll
            for(int j = 0; j < KO; ++j)
            {
                k.acc[r2][o][j] = 0.0f;
            }
        }
    }
    k.rsrc = k1Rsrc(static_cast<const char*>(a.in) + b.img * a.is[0] * ES, a.inImageBytes);
    k.walk.A = BWD ? a.pd - FD + 1 : -a.pd;
    k.walk.zfirst = DS ? b.od0 + k.walk.A : 0;
    k.walk.planes = DS ? (b.odN - b.od0) + FD - 1 : FD;
#if CN_VALU
    k.m = k.tid % M;
    k.group = __builtin_amdgcn_readfirstlane(k.tid / M);
#else
    k.sW = sW;
    k.lane = k.tid % WAVE;
    k.wave = __builtin_amdgcn_readfirstlane(k.tid / WAVE);
    k.col = k.lane % DIM;
    k.kb = Atom::template kBase<KPL>(k.lane);
    k.cs = static_cast<int>(a.is[1]);
#endif

    // The steps in order, the matrix cores' A loads PD steps ahead.
    int p = k.walk.seek(0, b, a);
    int q0 = b.qlo;
#if !CN_VALU
    k.stageFilter();
    k.ip = p;
    k.iq = q0;
    k.prime<0>();
#endif
    while(k.round<0>(p, q0))
    {
    }
    // ---- the remaining stores
    if constexpr(DS)
    {
        k.completeThrough(k.walk.planes);
    }
    else
    {
        storePlane(b, a, b.od0, k.acc[0]);
    }
}

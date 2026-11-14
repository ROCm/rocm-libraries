// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The VALU grouped backward-weights kernels: a main kernel and a fold kernel.
//
// The design of MIOpen's hand-written ConvAsmBwdWrW3x3 (conv3x3wrw.s), reimplemented in HIP and
// generalized: a wave owns a tile of CB input by KB output channels of one group and keeps every
// tap of the tile's CB x KB filters in registers, fp32. Its lanes lie along the output row: a
// lane owns P consecutive output columns and the P x SW input columns under them, read with one
// vector load a row; the window the filter's columns reach past them (the sliding window) comes
// from the neighbouring lanes through DPP (or a lane permute), so each element is read from memory
// once a row, not once a tap. Narrow rows put several rows side by side in a wave (lanes / row a
// power of two). The waves of a workgroup take turns along the reduction rows (images x output
// rows) of the workgroup's run, the run being one of `splits`; at the end each wave reduces its
// lanes' sums (a halving butterfly through lane permutes), the workgroup adds its waves' sums in
// LDS in wave order, and either writes dw (one split) or one fp32 sum per filter element to the
// workspace, which the fold kernel adds in split order. No atomics: the result is the same on
// every run.
//
// 3-D: a tile is also one depth tap t (the grid's tiles are fd x the plane's), its reduction rows
// images x output depth x output rows; a row reads the x depth slice od sd - pd + t dd, and a
// row whose slice is depth padding adds nothing. The depth is a runtime argument: the 2-D configs
// serve every depth filter (GVW_3D compiles the depth in; 2-D problems run without it). (The row
// ring stays 2-D.)
//
// One HIPRTC compile per config and alignment class; every shape is a runtime argument
// (ConvGroupValuWrwArgs.h).
//   GVW_DT      element type of x, dy and dw: 0 fp32, 1 fp16, 2 bf16 (16-bit read as raw bits)
//   GVW_LAYOUT  1 = channels-first (x, dy NCHW, the width packed), 0 = channels-last (NHWC, the
//               channels packed)
//   GVW_CB      input channels a tile;   GVW_KB  output channels a tile
//   GVW_P       output columns a lane
//   GVW_FR      filter rows;   GVW_FS  filter columns
//   GVW_SW      stride along the width;   GVW_DW  dilation along the width
//   GVW_PW      pre-padding along the width
//   GVW_NW      waves a workgroup;   GVW_WAVE  lanes a wave (64 on gfx9, 32 on gfx10+)
//   GVW_VBX     bytes a load of x (16, 8, 4 or the element size);  GVW_VBY  the same for dy
//   GVW_RAGGED  1 when a lane's window can reach past the end of a row (the width not a multiple
//               of the lane's columns): its elements past the row read zero
//   GVW_COLMASK 1 when a lane's output columns can reach past the output row (wo not a multiple of
//               P): the products of those columns are dropped (their x operand selected to 0), so
//               an inf / NaN in x never meets an output column the convolution does not have
//   GVW_3D      1 for a 3-D problem: tiles per depth tap, rows over the output depth (see above)
//   GVW_RING    channels-first, unit height stride and dilation: a lane group walks consecutive
//               output rows of one image with the filter's input rows in a register ring, so
//               each input row is read once (not once a filter row), and the next row's loads
//               are issued before the current row's multiplies (one row in flight)
//   GVW_DOT2    fp16 channels-first at unit width stride, P even, whole-dword loads: pairs of
//               adjacent output columns multiply on v_dot2_f32_f16 (two exact fp16 products
//               added to the fp32 sum), the window kept as dwords; the pairs a tap reads at an
//               odd element come from one half-dword shifted copy of the window

#include "ConvArchFamily.h"
#include "ConvGroupValuWrwArgs.h"

#ifndef GVW_DT
#error "GVW_DT must name the element type"
#endif
#ifndef GVW_LAYOUT
#define GVW_LAYOUT 1
#endif
#ifndef GVW_CB
#define GVW_CB 4
#endif
#ifndef GVW_KB
#define GVW_KB 4
#endif
#ifndef GVW_P
#define GVW_P 4
#endif
#ifndef GVW_FR
#define GVW_FR 3
#endif
#ifndef GVW_FS
#define GVW_FS 3
#endif
#ifndef GVW_SW
#define GVW_SW 1
#endif
#ifndef GVW_DW
#define GVW_DW 1
#endif
#ifndef GVW_PW
#define GVW_PW 1
#endif
#ifndef GVW_NW
#define GVW_NW 4
#endif
#ifndef GVW_WAVE
#define GVW_WAVE 64
#endif
#ifndef GVW_RAGGED
#define GVW_RAGGED 0
#endif
#ifndef GVW_DOT2
#define GVW_DOT2 0
#endif
#ifndef GVW_RING
#define GVW_RING 0
#endif
#ifndef GVW_3D
#define GVW_3D 0
#endif
#ifndef GVW_COLMASK
#define GVW_COLMASK 0
#endif

namespace
{

constexpr int CB = GVW_CB;
constexpr int KB = GVW_KB;
constexpr int P = GVW_P;
constexpr int FR = GVW_FR;
constexpr int FS = GVW_FS;
constexpr int SW = GVW_SW;
constexpr int DWD = GVW_DW;
constexpr int PW = GVW_PW;
constexpr int NW = GVW_NW;
constexpr int WAVE = GVW_WAVE;
constexpr bool RAGGED = GVW_RAGGED != 0;
constexpr bool DOT2 = GVW_DOT2 != 0;
constexpr bool RING = GVW_RING != 0;
constexpr bool D3 = GVW_3D != 0;
constexpr bool COLMASK = GVW_COLMASK != 0;
constexpr bool NCHW = GVW_LAYOUT == 1;

#if GVW_DT == 0
typedef unsigned Raw;
constexpr int ES = 4;
#else
typedef unsigned short Raw;
constexpr int ES = 2;
#endif

#ifndef GVW_VBX
#define GVW_VBX ES
#endif
#ifndef GVW_VBY
#define GVW_VBY ES
#endif
constexpr int VBX = GVW_VBX;
constexpr int VBY = GVW_VBY;

/// Input columns a lane's window holds: PW to its left (from the lane before), its own P x SW,
/// and RH to its right (from the lane after).
constexpr int OWN = P * SW;
constexpr int RH_ = (P - 1) * SW + (FS - 1) * DWD + 1 - PW - OWN;
constexpr int RH = RH_ > 0 ? RH_ : 0;
constexpr int SPAN = PW + OWN + RH;
static_assert(PW <= OWN && RH <= OWN, "the window reaches past the neighbouring lanes");
static_assert(VBX >= ES && VBY >= ES, "load granule");

/// The dot2 window as dwords: storage element t = u + O (window element u), so the lane's own
/// elements start on a dword; LW dwords of left halo, OWNW own, RW of right halo.
constexpr int O = PW & 1;
constexpr int OWNW = OWN / 2;
constexpr int LW = (PW + O) / 2;
constexpr int RW = (RH + 1) / 2;
constexpr int NWD = LW + OWNW + RW;
static_assert(!RING || NCHW, "row-ring configs: channels-first");
static_assert(!(DOT2 && COLMASK), "dot2 configs: whole output columns");
static_assert(!DOT2
                  || (GVW_DT == 1 && NCHW && SW == 1 && P % 2 == 0 && !RAGGED && VBX >= 4
                      && VBY >= 4),
              "dot2 configs: fp16 channels-first, unit width stride, even P, dword loads");

/// Filter elements a tile: (c, k, r, s), s fastest.
constexpr int NA = CB * KB * FR * FS;
/// NA padded to whole lanes: the butterfly leaves Q of them in every lane.
constexpr int NAP = (NA + WAVE - 1) / WAVE * WAVE;
constexpr int Q = NAP / WAVE;

#if HKP_ARCH_GFX11 || HKP_ARCH_GFX12
#define GVW_RSRC_FLAGS 0x31004000
#elif defined(__gfx1010__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
#define GVW_RSRC_FLAGS 0x31014000
#else
#define GVW_RSRC_FLAGS 0x00020000
#endif
constexpr unsigned OOB = 0x80000000u;

typedef unsigned GvwU2 __attribute__((ext_vector_type(2)));
typedef unsigned GvwU4 __attribute__((ext_vector_type(4)));

__device__ inline float toFloat(Raw bits)
{
#if GVW_DT == 0
    return __uint_as_float(bits);
#elif GVW_DT == 1
    return static_cast<float>(__builtin_bit_cast(_Float16, bits));
#else
    return __uint_as_float(static_cast<unsigned>(bits) << 16);
#endif
}

__device__ inline Raw fromFloat(float value)
{
#if GVW_DT == 0
    return __float_as_uint(value);
#elif GVW_DT == 1
    return __builtin_bit_cast(Raw, static_cast<_Float16>(value));
#else
    // Round to nearest even; NaN stays a quiet NaN.
    const unsigned u = __float_as_uint(value);
    if((u & 0x7fffffffu) > 0x7f800000u)
    {
        return static_cast<Raw>((u >> 16) | 0x40u);
    }
    return static_cast<Raw>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
#endif
}

/// A buffer descriptor over [base, base + bytes) (at most 2^31 - 1 bytes). Every caller passes
/// wave-uniform values; readfirstlane says so, so no access is wrapped in a waterfall loop.
__device__ inline __amdgpu_buffer_rsrc_t gvwRsrc(const void* base, long long bytes)
{
    const int records = bytes < 0x7fffffffLL ? static_cast<int>(bytes) : 0x7fffffff;
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>((static_cast<unsigned long long>(hi) << 32) | lo),
        static_cast<short>(0),
        __builtin_amdgcn_readfirstlane(records),
        GVW_RSRC_FLAGS);
}

/// The dword @p d of a load into elements from @p at.
template <int NE>
__device__ inline void putDword(Raw (&w)[NE], int at, unsigned d)
{
#if GVW_DT == 0
    w[at] = d;
#else
    w[at] = static_cast<Raw>(d & 0xffffu);
    w[at + 1] = static_cast<Raw>(d >> 16);
#endif
}

/// NE consecutive elements from byte @p offset (OOB: zeros) in loads of VB bytes.
template <int NE, int VB>
__device__ inline void loadRaw(__amdgpu_buffer_rsrc_t r, unsigned offset, Raw (&w)[NE])
{
    constexpr int NB = NE * ES;
    static_assert(NB % VB == 0, "a window is whole loads");
    constexpr int PER = VB / ES; // elements a load
#pragma unroll
    for(int i = 0; i < NB / VB; ++i)
    {
        const unsigned o = offset + static_cast<unsigned>(i * VB);
        if constexpr(VB == 16)
        {
            const GvwU4 v
                = __builtin_bit_cast(GvwU4, __builtin_amdgcn_raw_buffer_load_b128(r, o, 0, 0));
#pragma unroll
            for(int d = 0; d < 4; ++d)
            {
                putDword(w, i * PER + d * (4 / ES), v[d]);
            }
        }
        else if constexpr(VB == 8)
        {
            const GvwU2 v
                = __builtin_bit_cast(GvwU2, __builtin_amdgcn_raw_buffer_load_b64(r, o, 0, 0));
#pragma unroll
            for(int d = 0; d < 2; ++d)
            {
                putDword(w, i * PER + d * (4 / ES), v[d]);
            }
        }
        else if constexpr(VB == 4)
        {
            putDword(w, i * PER, __builtin_amdgcn_raw_buffer_load_b32(r, o, 0, 0));
        }
        else
        {
            static_assert(VB == 2 && ES == 2, "load granule");
            w[i] = static_cast<Raw>(__builtin_amdgcn_raw_buffer_load_b16(r, o, 0, 0));
        }
    }
}

/// NW32 consecutive dwords from byte @p offset (OOB: zeros) in loads of VB bytes (VB >= 4).
template <int NW32, int VB>
__device__ inline void loadDwords(__amdgpu_buffer_rsrc_t r, unsigned offset, unsigned (&w)[NW32])
{
    static_assert(VB >= 4 && (NW32 * 4) % VB == 0, "dword loads");
#pragma unroll
    for(int i = 0; i < NW32 * 4 / VB; ++i)
    {
        const unsigned o = offset + static_cast<unsigned>(i * VB);
        if constexpr(VB == 16)
        {
            const GvwU4 v
                = __builtin_bit_cast(GvwU4, __builtin_amdgcn_raw_buffer_load_b128(r, o, 0, 0));
#pragma unroll
            for(int d = 0; d < 4; ++d)
            {
                w[i * 4 + d] = v[d];
            }
        }
        else if constexpr(VB == 8)
        {
            const GvwU2 v
                = __builtin_bit_cast(GvwU2, __builtin_amdgcn_raw_buffer_load_b64(r, o, 0, 0));
            w[i * 2] = v[0];
            w[i * 2 + 1] = v[1];
        }
        else
        {
            w[i] = __builtin_amdgcn_raw_buffer_load_b32(r, o, 0, 0);
        }
    }
}

/// The value of @p v in the lane before (lane - 1) and after (lane + 1). Lanes whose neighbour
/// is in another row (or outside the wave) are zeroed by the caller.
__device__ inline unsigned fromLanePrev(unsigned v, int lane)
{
#if GVW_WAVE == 64 && !defined(__GFX10__) && !defined(__GFX11__) && !defined(__GFX12__)
    (void)lane;
    return static_cast<unsigned>(__builtin_amdgcn_update_dpp(
        0, static_cast<int>(v), 0x138 /* wave_shr:1 */, 0xF, 0xF, false));
#else
    return static_cast<unsigned>(
        __builtin_amdgcn_ds_bpermute((lane - 1) << 2, static_cast<int>(v)));
#endif
}
__device__ inline unsigned fromLaneNext(unsigned v, int lane)
{
#if GVW_WAVE == 64 && !defined(__GFX10__) && !defined(__GFX11__) && !defined(__GFX12__)
    (void)lane;
    return static_cast<unsigned>(__builtin_amdgcn_update_dpp(
        0, static_cast<int>(v), 0x130 /* wave_shl:1 */, 0xF, 0xF, false));
#else
    return static_cast<unsigned>(
        __builtin_amdgcn_ds_bpermute((lane + 1) << 2, static_cast<int>(v)));
#endif
}

/// A barrier over the workgroup's LDS only (no global memory fence: gfx12's full barrier
/// invalidates caches).
__device__ inline void ldsBarrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/// One halving step of the cross-lane reduction: lanes M apart exchange half of their CUR
/// values; the lower lane keeps the low half, the upper the high half, each adding its
/// partner's. After the last step lane l holds elements [l Q, l Q + Q).
template <int M, int CUR>
__device__ inline void halve(float (&v)[NAP], int lane)
{
    if constexpr(M >= 1)
    {
        constexpr int H = CUR / 2;
        const bool up = (lane & M) != 0;
#pragma unroll
        for(int i = 0; i < H; ++i)
        {
            const float send = up ? v[i] : v[i + H];
            const float keep = up ? v[i + H] : v[i];
            const float got = __int_as_float(
                __builtin_amdgcn_ds_bpermute((lane ^ M) << 2, __float_as_int(send)));
            v[i] = keep + got;
        }
        halve<M / 2, H>(v, lane);
    }
}

/// Writes the sum @p value of filter element @p e of tile @p tile to dw, in dw's element type.
__device__ inline void writeDw(const ConvGroupValuWrwArgs& a, int tile, int e, float value)
{
    const int t = D3 ? tile / a.tilesPlane : 0;
    const int planeTile = tile - t * a.tilesPlane;
    const int g = planeTile / (a.cBlocks * a.kBlocks);
    const int rem = planeTile - g * (a.cBlocks * a.kBlocks);
    const int cb = rem / a.kBlocks;
    const int kb = rem - cb * a.kBlocks;
    const int s = e % FS;
    const int r = (e / FS) % FR;
    const int k = (e / (FS * FR)) % KB;
    const int c = e / (FS * FR * KB);
    const long long kk = static_cast<long long>(g) * a.kpg + kb * KB + k;
    const long long cc = static_cast<long long>(cb) * CB + c;
    const long long rr = a.flip != 0 ? FR - 1 - r : r;
    const long long ss = a.flip != 0 ? FS - 1 - s : s;
    const long long tt = D3 ? (a.flip != 0 ? a.fd - 1 - t : t) : 0;
    Raw* dw = static_cast<Raw*>(a.dw);
    dw[kk * a.ws[0] + cc * a.ws[1] + tt * a.wsd + rr * a.ws[2] + ss * a.ws[3]] = fromFloat(value);
}

} // namespace

/// One pass of the dot2 path (GVW_DOT2): dy as KB rows of P / 2 dwords, x per filter row and
/// channel as the NWD-dword window, the odd-element pairs from its half-dword shifted copy.
__device__ inline void passDot2(float (&acc)[CB][KB][FR][FS],
                                __amdgpu_buffer_rsrc_t xr,
                                __amdgpu_buffer_rsrc_t yr,
                                const ConvGroupValuWrwArgs& a,
                                bool rowOk,
                                bool colX,
                                bool colY,
                                bool firstCol,
                                bool lastCol,
                                int lane,
                                int xImg, // element offset of the row's x image and depth slice
                                int yImg, // the same for dy
                                int oy,
                                int ox0,
                                int xsc,
                                int xsh,
                                int ysk,
                                int ysh)
{
    if constexpr(DOT2)
    {
        typedef _Float16 GvwH2 __attribute__((ext_vector_type(2)));
        unsigned dyd[KB][P / 2];
        {
            const bool ok = rowOk & colY;
#pragma unroll
            for(int k = 0; k < KB; ++k)
            {
                const int e = yImg + k * ysk + oy * ysh + ox0;
                loadDwords<P / 2, VBY>(yr, ok ? static_cast<unsigned>(e * ES) : OOB, dyd[k]);
            }
        }
        unsigned wd[FR][CB][NWD];
#pragma unroll
        for(int r = 0; r < FR; ++r)
        {
            const int iy = oy * a.sh - a.ph + r * a.dh;
            const bool ok
                = rowOk & colX & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
#pragma unroll
            for(int c = 0; c < CB; ++c)
            {
                unsigned own[OWNW];
                const int e = xImg + c * xsc + iy * xsh + ox0;
                loadDwords<OWNW, VBX>(xr, ok ? static_cast<unsigned>(e * ES) : OOB, own);
#pragma unroll
                for(int i = 0; i < LW; ++i)
                {
                    const unsigned v = fromLanePrev(own[OWNW - LW + i], lane);
                    wd[r][c][i] = firstCol ? 0u : v;
                }
#pragma unroll
                for(int i = 0; i < OWNW; ++i)
                {
                    wd[r][c][LW + i] = own[i];
                }
#pragma unroll
                for(int i = 0; i < RW; ++i)
                {
                    const unsigned v = fromLaneNext(own[i], lane);
                    wd[r][c][LW + OWNW + i] = lastCol ? 0u : v;
                }
            }
        }
        if(rowOk & colY)
        {
#pragma unroll
            for(int r = 0; r < FR; ++r)
            {
#pragma unroll
                for(int c = 0; c < CB; ++c)
                {
                    // Storage pairs (2i + 1, 2i + 2).
                    unsigned sh[NWD > 1 ? NWD - 1 : 1];
#pragma unroll
                    for(int i = 0; i + 1 < NWD; ++i)
                    {
                        sh[i] = __builtin_amdgcn_alignbit(wd[r][c][i + 1], wd[r][c][i], 16);
                    }
#pragma unroll
                    for(int s = 0; s < FS; ++s)
                    {
#pragma unroll
                        for(int jp = 0; jp < P / 2; ++jp)
                        {
                            // Columns 2 jp and 2 jp + 1 read window elements u and u + 1.
                            const int t = 2 * jp + s * DWD + O;
                            const unsigned pair = (t & 1) == 0 ? wd[r][c][t / 2] : sh[(t - 1) / 2];
#pragma unroll
                            for(int k = 0; k < KB; ++k)
                            {
                                acc[c][k][r][s]
                                    = __builtin_amdgcn_fdot2(__builtin_bit_cast(GvwH2, pair),
                                                             __builtin_bit_cast(GvwH2, dyd[k][jp]),
                                                             acc[c][k][r][s],
                                                             false);
                            }
                        }
                    }
                }
            }
        }
    }
}

#if GVW_RING
/// The row-ring path (GVW_RING). One input row of CB channels as loaded (dwords for dot2, else
/// elements), its window with the neighbours' halos, and the multiplies of one output row.
#if GVW_DOT2
typedef unsigned RingRaw[CB][OWNW];
typedef unsigned RingWin[CB][NWD];
typedef unsigned RingDy[KB][P / 2];
#else
typedef Raw RingRaw[CB][OWN];
typedef Raw RingWin[CB][SPAN];
typedef Raw RingDy[KB][P];
#endif

__device__ inline void ringLoadX(RingRaw& out,
                                 __amdgpu_buffer_rsrc_t xr,
                                 bool ok,
                                 int base, // element offset of (dn, c0, iy, ox0)
                                 int xsc)
{
#pragma unroll
    for(int c = 0; c < CB; ++c)
    {
        const unsigned off = ok ? static_cast<unsigned>((base + c * xsc) * ES) : OOB;
#if GVW_DOT2
        loadDwords<OWNW, VBX>(xr, off, out[c]);
#else
        loadRaw<OWN, VBX>(xr, off, out[c]);
#endif
    }
}

__device__ inline void
    ringLoadDy(RingDy& out, __amdgpu_buffer_rsrc_t yr, bool ok, int base, int ysk)
{
#pragma unroll
    for(int k = 0; k < KB; ++k)
    {
        const unsigned off = ok ? static_cast<unsigned>((base + k * ysk) * ES) : OOB;
#if GVW_DOT2
        loadDwords<P / 2, VBY>(yr, off, out[k]);
#else
        loadRaw<P, VBY>(yr, off, out[k]);
#endif
    }
}

__device__ inline void ringWindow(
    RingWin& win, const RingRaw& own, bool firstCol, bool lastCol, int lane, int ox0, int wi)
{
#pragma unroll
    for(int c = 0; c < CB; ++c)
    {
#if GVW_DOT2
        (void)ox0;
        (void)wi;
#pragma unroll
        for(int i = 0; i < LW; ++i)
        {
            const unsigned v = fromLanePrev(own[c][OWNW - LW + i], lane);
            win[c][i] = firstCol ? 0u : v;
        }
#pragma unroll
        for(int i = 0; i < OWNW; ++i)
        {
            win[c][LW + i] = own[c][i];
        }
#pragma unroll
        for(int i = 0; i < RW; ++i)
        {
            const unsigned v = fromLaneNext(own[c][i], lane);
            win[c][LW + OWNW + i] = lastCol ? 0u : v;
        }
#else
        Raw o[OWN];
#pragma unroll
        for(int u = 0; u < OWN; ++u)
        {
            o[u] = own[c][u];
            if constexpr(RAGGED)
            {
                o[u] = ox0 * SW + u < wi ? o[u] : Raw(0);
            }
        }
#pragma unroll
        for(int t = 0; t < PW; ++t)
        {
            const unsigned v = fromLanePrev(o[OWN - PW + t], lane);
            win[c][t] = firstCol ? Raw(0) : static_cast<Raw>(v);
        }
#pragma unroll
        for(int u = 0; u < OWN; ++u)
        {
            win[c][PW + u] = o[u];
        }
#pragma unroll
        for(int t = 0; t < RH; ++t)
        {
            const unsigned v = fromLaneNext(o[t], lane);
            win[c][PW + OWN + t] = lastCol ? Raw(0) : static_cast<Raw>(v);
        }
#endif
    }
}

/// The multiplies of one output row: filter row r reads ring slot (U + r) % FR.
template <int U>
__device__ inline void ringRow(
    float (&acc)[CB][KB][FR][FS], const RingWin (&ring)[FR], const RingDy& dy, int ox0, int wo)
{
#if GVW_DOT2
    (void)ox0;
    (void)wo;
    typedef _Float16 GvwH2 __attribute__((ext_vector_type(2)));
#pragma unroll
    for(int r = 0; r < FR; ++r)
    {
#pragma unroll
        for(int c = 0; c < CB; ++c)
        {
            const unsigned* wd = ring[(U + r) % FR][c];
            unsigned sh[NWD > 1 ? NWD - 1 : 1];
#pragma unroll
            for(int i = 0; i + 1 < NWD; ++i)
            {
                sh[i] = __builtin_amdgcn_alignbit(wd[i + 1], wd[i], 16);
            }
#pragma unroll
            for(int s = 0; s < FS; ++s)
            {
#pragma unroll
                for(int jp = 0; jp < P / 2; ++jp)
                {
                    const int t = 2 * jp + s * DWD + O;
                    const unsigned pair = (t & 1) == 0 ? wd[t / 2] : sh[(t - 1) / 2];
#pragma unroll
                    for(int k = 0; k < KB; ++k)
                    {
                        acc[c][k][r][s]
                            = __builtin_amdgcn_fdot2(__builtin_bit_cast(GvwH2, pair),
                                                     __builtin_bit_cast(GvwH2, dy[k][jp]),
                                                     acc[c][k][r][s],
                                                     false);
                    }
                }
            }
        }
    }
#else
    float dyf[KB][P];
#pragma unroll
    for(int k = 0; k < KB; ++k)
    {
#pragma unroll
        for(int j = 0; j < P; ++j)
        {
            Raw v = dy[k][j];
            if constexpr(RAGGED)
            {
                v = ox0 + j < wo ? v : Raw(0);
            }
            dyf[k][j] = toFloat(v);
        }
    }
#pragma unroll
    for(int r = 0; r < FR; ++r)
    {
#pragma unroll
        for(int c = 0; c < CB; ++c)
        {
#pragma unroll
            for(int s = 0; s < FS; ++s)
            {
#pragma unroll
                for(int j = 0; j < P; ++j)
                {
                    float xv = toFloat(ring[(U + r) % FR][c][j * SW + s * DWD]);
                    if constexpr(COLMASK)
                    {
                        xv = ox0 + j < wo ? xv : 0.0f;
                    }
#pragma unroll
                    for(int k = 0; k < KB; ++k)
                    {
                        acc[c][k][r][s] = __builtin_fmaf(xv, dyf[k][j], acc[c][k][r][s]);
                    }
                }
            }
        }
    }
#endif
}

/// One ring step U of a group of FR: the staged row becomes ring slot (U + FR - 1) % FR, the next
/// row's loads go out, then the multiplies.
template <int U>
__device__ inline void ringStep(float (&acc)[CB][KB][FR][FS],
                                RingWin (&ring)[FR],
                                RingRaw& sx,
                                RingDy& sy,
                                __amdgpu_buffer_rsrc_t xr,
                                __amdgpu_buffer_rsrc_t yr,
                                const ConvGroupValuWrwArgs& a,
                                bool itemOk,
                                int oy,
                                int oyEnd,
                                int xRow0, // element offset of (dn, c0, 0, ox0)
                                int yRow0, // element offset of (dn, k0, 0, ox0)
                                int xsc,
                                int xsh,
                                int ysk,
                                int ysh,
                                bool firstCol,
                                bool lastCol,
                                bool colX,
                                bool colY,
                                int lane,
                                int ox0)
{
    RingDy dy;
#pragma unroll
    for(int k = 0; k < KB; ++k)
    {
#if GVW_DOT2
#pragma unroll
        for(int i = 0; i < P / 2; ++i)
            dy[k][i] = sy[k][i];
#else
#pragma unroll
        for(int i = 0; i < P; ++i)
            dy[k][i] = sy[k][i];
#endif
    }
    ringWindow(ring[(U + FR - 1) % FR], sx, firstCol, lastCol, lane, ox0, a.wi);
    // Next row's input row (oy + 1 - ph + FR - 1) and dy row (oy + 1).
    {
        const int iy = oy + 1 - a.ph + FR - 1;
        const bool ok = itemOk & colX & (oy + 1 < oyEnd)
                        & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
        ringLoadX(sx, xr, ok, xRow0 + iy * xsh, xsc);
        ringLoadDy(sy, yr, itemOk & colY & (oy + 1 < oyEnd), yRow0 + (oy + 1) * ysh, ysk);
    }
    if(itemOk & colY & (oy < oyEnd))
    {
        ringRow<U>(acc, ring, dy, ox0, a.wo);
    }
}

template <int U>
__device__ inline void ringSteps(float (&acc)[CB][KB][FR][FS],
                                 RingWin (&ring)[FR],
                                 RingRaw& sx,
                                 RingDy& sy,
                                 __amdgpu_buffer_rsrc_t xr,
                                 __amdgpu_buffer_rsrc_t yr,
                                 const ConvGroupValuWrwArgs& a,
                                 bool itemOk,
                                 int i,
                                 int oyBase,
                                 int oyEnd,
                                 int xRow0,
                                 int yRow0,
                                 int xsc,
                                 int xsh,
                                 int ysk,
                                 int ysh,
                                 bool firstCol,
                                 bool lastCol,
                                 bool colX,
                                 bool colY,
                                 int lane,
                                 int ox0)
{
    if constexpr(U < FR)
    {
        if(i + U < a.chunk)
        {
            ringStep<U>(acc,
                        ring,
                        sx,
                        sy,
                        xr,
                        yr,
                        a,
                        itemOk,
                        oyBase + i + U,
                        oyEnd,
                        xRow0,
                        yRow0,
                        xsc,
                        xsh,
                        ysk,
                        ysh,
                        firstCol,
                        lastCol,
                        colX,
                        colY,
                        lane,
                        ox0);
            ringSteps<U + 1>(acc,
                             ring,
                             sx,
                             sy,
                             xr,
                             yr,
                             a,
                             itemOk,
                             i,
                             oyBase,
                             oyEnd,
                             xRow0,
                             yRow0,
                             xsc,
                             xsh,
                             ysk,
                             ysh,
                             firstCol,
                             lastCol,
                             colX,
                             colY,
                             lane,
                             ox0);
        }
    }
}
#endif // GVW_RING

/// Grid (tiles, splits), NW waves a workgroup. See the file comment.
extern "C" __global__ void __launch_bounds__(GVW_NW* GVW_WAVE)
    ConvGroupValuWrw(ConvGroupValuWrwArgs a)
{
    const int tile = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int tap = D3 ? tile / a.tilesPlane : 0; // the depth tap
    const int planeTile = tile - tap * a.tilesPlane;
    const int g = planeTile / (a.cBlocks * a.kBlocks);
    const int rem = planeTile - g * (a.cBlocks * a.kBlocks);
    const int cb = rem / a.kBlocks;
    const int kb = rem - cb * a.kBlocks;
    const int c0 = g * a.cpg + cb * CB; // x channel of the tile's first input channel
    const int k0 = g * a.kpg + kb * KB; // dy channel of its first output channel

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane = tid % WAVE;
    const int lanes = a.lanes;
    const int slot = lane / lanes; // the row of the pass this lane works on
    const int col = lane - slot * lanes;
    const int rpw = WAVE / lanes; // rows a pass
    const int ox0 = col * P;
    const bool firstCol = col == 0;
    const bool lastCol = col == lanes - 1;
    const bool colX = ox0 * SW < a.wi; // the lane's own x columns start inside the row
    const bool colY = ox0 < a.wo; // the lane owns output columns

    const int rowBegin = split * a.rowsPerSplit;
    const int rowEnd = min(a.rows, rowBegin + a.rowsPerSplit);

    // 32-bit element strides: every offset within the images a pass spans fits (host gate).
    const int xsn = static_cast<int>(a.xs[0]);
    const int xsc = static_cast<int>(a.xs[1]);
    const int xsh = static_cast<int>(a.xs[2]);
    const int xsw = static_cast<int>(a.xs[3]);
    const int ysn = static_cast<int>(a.ys[0]);
    const int ysk = static_cast<int>(a.ys[1]);
    const int ysh = static_cast<int>(a.ys[2]);
    const int ysw = static_cast<int>(a.ys[3]);
    const int xsd = static_cast<int>(a.xsd);
    const int ysd = static_cast<int>(a.ysd);

    float acc[CB][KB][FR][FS];
#pragma unroll
    for(int c = 0; c < CB; ++c)
#pragma unroll
        for(int k = 0; k < KB; ++k)
#pragma unroll
            for(int r = 0; r < FR; ++r)
#pragma unroll
                for(int s = 0; s < FS; ++s)
                    acc[c][k][r][s] = 0.0f;

#if GVW_RING
    if constexpr(RING)
    {
        const int itemBegin = split * a.itemsPerSplit;
        const int itemEnd = min(a.items, itemBegin + a.itemsPerSplit);
        for(int ib = itemBegin + wave * rpw; ib < itemEnd; ib += NW * rpw)
        {
            const int n0 = ib / a.itemsPerImage;
            const long long xBase
                = (static_cast<long long>(n0) * a.xs[0] + static_cast<long long>(c0) * a.xs[1])
                  * ES;
            const long long yBase
                = (static_cast<long long>(n0) * a.ys[0] + static_cast<long long>(k0) * a.ys[1])
                  * ES;
            const __amdgpu_buffer_rsrc_t xr
                = gvwRsrc(static_cast<const char*>(a.x) + xBase, a.xBytes - xBase);
            const __amdgpu_buffer_rsrc_t yr
                = gvwRsrc(static_cast<const char*>(a.dy) + yBase, a.yBytes - yBase);
            const int item = ib + slot;
            const bool itemOk = item < itemEnd;
            const int n = item / a.itemsPerImage;
            const int oyBase = (item - n * a.itemsPerImage) * a.chunk;
            const int oyEnd = min(a.ho, oyBase + a.chunk);
            const int dn = n - n0;
            const int xRow0 = dn * xsn + ox0 * SW;
            const int yRow0 = dn * ysn + ox0;
            RingWin ring[FR];
            RingRaw sx;
            RingDy sy;
            // Prologue: the first FR - 1 input rows into slots 0 .. FR - 2.
#pragma unroll
            for(int r = 0; r + 1 < FR; ++r)
            {
                const int iy = oyBase - a.ph + r;
                const bool ok
                    = itemOk & colX & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
                ringLoadX(sx, xr, ok, xRow0 + iy * xsh, xsc);
                ringWindow(ring[r], sx, firstCol, lastCol, lane, ox0, a.wi);
            }
            {
                const int iy = oyBase - a.ph + FR - 1;
                const bool ok
                    = itemOk & colX & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
                ringLoadX(sx, xr, ok, xRow0 + iy * xsh, xsc);
                ringLoadDy(sy, yr, itemOk & colY, yRow0 + oyBase * ysh, ysk);
            }
            for(int i = 0; i < a.chunk; i += FR)
            {
                ringSteps<0>(acc,
                             ring,
                             sx,
                             sy,
                             xr,
                             yr,
                             a,
                             itemOk,
                             i,
                             oyBase,
                             oyEnd,
                             xRow0,
                             yRow0,
                             xsc,
                             xsh,
                             ysk,
                             ysh,
                             firstCol,
                             lastCol,
                             colX,
                             colY,
                             lane,
                             ox0);
            }
        }
    }
    else
#endif
        for(int pr = rowBegin + wave * rpw; pr < rowEnd; pr += NW * rpw)
        {
            // The pass's first image: a 64-bit base the lanes' 32-bit offsets start from.
            const int n0 = pr / a.rowsAnImage;
            const long long xBase
                = (static_cast<long long>(n0) * a.xs[0] + static_cast<long long>(c0) * a.xs[1])
                  * ES;
            const long long yBase
                = (static_cast<long long>(n0) * a.ys[0] + static_cast<long long>(k0) * a.ys[1])
                  * ES;
            const __amdgpu_buffer_rsrc_t xr
                = gvwRsrc(static_cast<const char*>(a.x) + xBase, a.xBytes - xBase);
            const __amdgpu_buffer_rsrc_t yr
                = gvwRsrc(static_cast<const char*>(a.dy) + yBase, a.yBytes - yBase);

            const int row = pr + slot;
            const int n = row / a.rowsAnImage;
            const int plane = row - n * a.rowsAnImage;
            int od = 0;
            int oy = plane;
            if constexpr(D3)
            {
                od = plane / a.ho;
                oy = plane - od * a.ho;
            }
            const int dn = n - n0;
            // The x depth slice this row's tap reads; outside x (depth padding) the row adds
            // nothing.
            const int id = od * a.sd - a.pd + tap * a.dd;
            const bool rowOk
                = (row < rowEnd) & (!D3 || static_cast<unsigned>(id) < static_cast<unsigned>(a.di));
            const int xImg = D3 ? dn * xsn + id * xsd : dn * xsn;
            const int yImg = D3 ? dn * ysn + od * ysd : dn * ysn;

            if constexpr(DOT2)
            {
                passDot2(acc,
                         xr,
                         yr,
                         a,
                         rowOk,
                         colX,
                         colY,
                         firstCol,
                         lastCol,
                         lane,
                         xImg,
                         yImg,
                         oy,
                         ox0,
                         xsc,
                         xsh,
                         ysk,
                         ysh);
            }
            else
            {
                // dy: KB output channels x P columns.
                float dyf[KB][P];
                {
                    Raw dyw[KB][P];
                    if constexpr(NCHW)
                    {
                        const bool ok = rowOk & colY;
#pragma unroll
                        for(int k = 0; k < KB; ++k)
                        {
                            const int e = yImg + k * ysk + oy * ysh + ox0;
                            loadRaw<P, VBY>(yr, ok ? static_cast<unsigned>(e * ES) : OOB, dyw[k]);
                        }
                    }
                    else
                    {
#pragma unroll
                        for(int j = 0; j < P; ++j)
                        {
                            const bool ok = rowOk & (ox0 + j < a.wo);
                            const int e = yImg + oy * ysh + (ox0 + j) * ysw;
                            Raw v[KB];
                            loadRaw<KB, VBY>(yr, ok ? static_cast<unsigned>(e * ES) : OOB, v);
#pragma unroll
                            for(int k = 0; k < KB; ++k)
                            {
                                dyw[k][j] = v[k];
                            }
                        }
                    }
#pragma unroll
                    for(int k = 0; k < KB; ++k)
                    {
#pragma unroll
                        for(int j = 0; j < P; ++j)
                        {
                            Raw v = dyw[k][j];
                            if constexpr(RAGGED && NCHW)
                            {
                                v = ox0 + j < a.wo ? v : Raw(0);
                            }
                            dyf[k][j] = toFloat(v);
                        }
                    }
                }

                // x: per filter row, CB channels x the lane's window.
                Raw xw[FR][CB][SPAN];
#pragma unroll
                for(int r = 0; r < FR; ++r)
                {
                    const int iy = oy * a.sh - a.ph + r * a.dh;
                    const bool rowIn
                        = rowOk & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
                    Raw own[CB][OWN];
                    if constexpr(NCHW)
                    {
                        const bool ok = rowIn & colX;
#pragma unroll
                        for(int c = 0; c < CB; ++c)
                        {
                            const int e = xImg + c * xsc + iy * xsh + ox0 * SW;
                            loadRaw<OWN, VBX>(xr, ok ? static_cast<unsigned>(e * ES) : OOB, own[c]);
                            if constexpr(RAGGED)
                            {
#pragma unroll
                                for(int u = 0; u < OWN; ++u)
                                {
                                    own[c][u] = ox0 * SW + u < a.wi ? own[c][u] : Raw(0);
                                }
                            }
                        }
                    }
                    else
                    {
#pragma unroll
                        for(int u = 0; u < OWN; ++u)
                        {
                            const int ix = ox0 * SW + u;
                            const bool ok = rowIn & (ix < a.wi);
                            const int e = xImg + iy * xsh + ix * xsw;
                            Raw v[CB];
                            loadRaw<CB, VBX>(xr, ok ? static_cast<unsigned>(e * ES) : OOB, v);
#pragma unroll
                            for(int c = 0; c < CB; ++c)
                            {
                                own[c][u] = v[c];
                            }
                        }
                    }
#pragma unroll
                    for(int c = 0; c < CB; ++c)
                    {
#pragma unroll
                        for(int t = 0; t < PW; ++t)
                        {
                            const unsigned v = fromLanePrev(own[c][OWN - PW + t], lane);
                            xw[r][c][t] = firstCol ? Raw(0) : static_cast<Raw>(v);
                        }
#pragma unroll
                        for(int u = 0; u < OWN; ++u)
                        {
                            xw[r][c][PW + u] = own[c][u];
                        }
#pragma unroll
                        for(int t = 0; t < RH; ++t)
                        {
                            const unsigned v = fromLaneNext(own[c][t], lane);
                            xw[r][c][PW + OWN + t] = lastCol ? Raw(0) : static_cast<Raw>(v);
                        }
                    }
                }

                if(rowOk & colY)
                {
#pragma unroll
                    for(int r = 0; r < FR; ++r)
                    {
#pragma unroll
                        for(int c = 0; c < CB; ++c)
                        {
#pragma unroll
                            for(int s = 0; s < FS; ++s)
                            {
#pragma unroll
                                for(int j = 0; j < P; ++j)
                                {
                                    // Output column ox0 + j reads input column (ox0 + j) SW - PW + s DW,
                                    // window element j SW + s DW.
                                    float xv = toFloat(xw[r][c][j * SW + s * DWD]);
                                    if constexpr(COLMASK)
                                    {
                                        // A column past the output row: no product (its dy
                                        // is 0, but x there can be inf).
                                        xv = ox0 + j < a.wo ? xv : 0.0f;
                                    }
#pragma unroll
                                    for(int k = 0; k < KB; ++k)
                                    {
                                        acc[c][k][r][s]
                                            = __builtin_fmaf(xv, dyf[k][j], acc[c][k][r][s]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    // Every lane holds sums of the same NA filter elements: halve them across the wave.
    float v[NAP];
#pragma unroll
    for(int c = 0; c < CB; ++c)
#pragma unroll
        for(int k = 0; k < KB; ++k)
#pragma unroll
            for(int r = 0; r < FR; ++r)
#pragma unroll
                for(int s = 0; s < FS; ++s)
                    v[((c * KB + k) * FR + r) * FS + s] = acc[c][k][r][s];
#pragma unroll
    for(int i = NA; i < NAP; ++i)
    {
        v[i] = 0.0f;
    }
    halve<WAVE / 2, NAP>(v, lane);

    // The workgroup's waves, added in wave order.
    __shared__ float red[NW * NAP];
#pragma unroll
    for(int i = 0; i < Q; ++i)
    {
        red[wave * NAP + lane * Q + i] = v[i];
    }
    ldsBarrier();
    for(int e = tid; e < NA; e += NW * WAVE)
    {
        float sum = red[e];
#pragma unroll
        for(int w = 1; w < NW; ++w)
        {
            sum += red[w * NAP + e];
        }
        if(a.splits == 1)
        {
            writeDw(a, tile, e, sum);
        }
        else
        {
            a.partial[(static_cast<long long>(split) * a.tiles + tile) * NA + e] = sum;
        }
    }
}

/// Adds the splits' sums of every (tile, filter element) in split order and writes dw. Grid:
/// tiles x NA / 256 blocks of 256 threads.
extern "C" __global__ void __launch_bounds__(256) ConvGroupValuWrwFold(ConvGroupValuWrwArgs a)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    const long long total = static_cast<long long>(a.tiles) * NA;
    if(i >= total)
    {
        return;
    }
    const int tile = static_cast<int>(i / NA);
    const int e = static_cast<int>(i - static_cast<long long>(tile) * NA);
    const long long stride = static_cast<long long>(a.tiles) * NA;
    float sum = 0.0f;
    for(int s = 0; s < a.splits; ++s)
    {
        sum += a.partial[s * stride + i];
    }
    writeDw(a, tile, e, sum);
}

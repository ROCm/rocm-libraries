// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The VALU grouped forward / backward-data kernels: a prepare kernel and a main kernel.
//
// The forward counterpart of CONV_GROUP_VALU_WRW (ConvGroupValuWrw.cpp): a wave owns KB output
// channels of one group; its lanes lie along the output row, each owning P consecutive output
// columns (narrow rows put several rows side by side, lanes a row a power of two). Per input
// channel and filter row it reads the input window under its columns once (one vector load a
// row), the window's halo from the neighbouring lanes (DPP on gfx9, a lane permute elsewhere), and
// multiplies every filter column and output channel from registers; the filter comes from the
// prepared copy as wave-uniform values (scalar loads). Every output element belongs to exactly
// one lane, so there is no reduction across lanes or workgroups: sums are fp32 in a fixed order,
// one rounding at the store.
//
// The prepare kernel writes the filter as [group][ko block][ci][t][r][s][KB] fp32: forward the
// graph's filter (reflected in convolution mode), backward data the filter with its roles swapped
// and reflected (dx = dy correlated with it, padding span - 1 - pad, dy upsampled by the graph's
// stride). An upsampled tap reads dy only where its position is a multiple of the stride: along
// the width that is known at compile time (GV_US divides P), so the taps that read zeros are not
// emitted; along the height and depth it is a runtime check per row, and the rows are ordered by
// phase (rows of the same position modulo the stride together) so that a pass's rows mostly agree
// on which filter rows read dy and the others are skipped by a wave-uniform branch.
//
// One HIPRTC compile per config and alignment class; every shape is a runtime argument
// (ConvGroupValuArgs.h).
//   GV_DT      element type: 0 fp32, 1 fp16, 2 bf16
//   GV_LAYOUT  1 = channels-first (the width packed), 0 = channels-last (the channels packed)
//   GV_KB      output channels a wave;  GV_CB  input channels a step (an unroll of the reduction)
//   GV_P       output columns a lane
//   GV_FD, GV_FR, GV_FS  filter depth, rows, columns
//   GV_SW      stride along the width (forward);  GV_US  upsampling of in along the width
//              (backward data: the graph's width stride);  GV_DW  dilation along the width;
//              GV_PW  the kernel's width padding
//   GV_UPROWS  1 when in is upsampled along the height or depth (backward data at a height or
//              depth stride above 1): the per-row stride checks and the phase order
//   GV_NW      waves a workgroup;  GV_WAVE  lanes a wave
//   GV_VBX     bytes a load of in;  GV_VBY  bytes a store of out
//   GV_RAGGED  1 when a lane's window can reach past the end of an input row (element loads, the
//              elements past the row zeroed)
//   GV_COLMASK 1 when a lane's output columns can reach past the output row (those are not stored)

#include "ConvArchFamily.h"
#include "ConvGroupValuArgs.h"

#ifndef GV_DT
#error "GV_DT must name the element type"
#endif
#ifndef GV_LAYOUT
#define GV_LAYOUT 1
#endif
#ifndef GV_KB
#define GV_KB 4
#endif
#ifndef GV_CB
#define GV_CB 2
#endif
#ifndef GV_P
#define GV_P 4
#endif
#ifndef GV_FD
#define GV_FD 1
#endif
#ifndef GV_FR
#define GV_FR 3
#endif
#ifndef GV_FS
#define GV_FS 3
#endif
#ifndef GV_SW
#define GV_SW 1
#endif
#ifndef GV_US
#define GV_US 1
#endif
#ifndef GV_DW
#define GV_DW 1
#endif
#ifndef GV_UPROWS
#define GV_UPROWS 0
#endif
#ifndef GV_PW
#define GV_PW 1
#endif
#ifndef GV_NW
#define GV_NW 4
#endif
#ifndef GV_WAVE
#define GV_WAVE 64
#endif
#ifndef GV_RAGGED
#define GV_RAGGED 0
#endif
#ifndef GV_COLMASK
#define GV_COLMASK 0
#endif

namespace
{

constexpr int KB = GV_KB;
constexpr int CB = GV_CB;
constexpr int P = GV_P;
constexpr int FD = GV_FD;
constexpr int FR = GV_FR;
constexpr int FS = GV_FS;
constexpr int TAPS = FD * FR * FS;
constexpr int SW = GV_SW;
constexpr int US = GV_US;
constexpr bool UPROWS = GV_UPROWS != 0;
constexpr int DWD = GV_DW;
constexpr int PW = GV_PW;
constexpr int NW = GV_NW;
constexpr int WAVE = GV_WAVE;
constexpr bool RAGGED = GV_RAGGED != 0;
constexpr bool COLMASK = GV_COLMASK != 0;
constexpr bool NCHW = GV_LAYOUT == 1;

#if GV_DT == 0
typedef unsigned Raw;
constexpr int ES = 4;
#else
typedef unsigned short Raw;
constexpr int ES = 2;
#endif

#ifndef GV_VBX
#define GV_VBX ES
#endif
#ifndef GV_VBY
#define GV_VBY ES
#endif
constexpr int VBX = GV_VBX;
constexpr int VBY = GV_VBY;

constexpr int floorDiv(int a, int b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

/// Output column j of a lane, filter column s: the (upsampled) input position j SW + s DW - PW
/// past the lane's first; a tap exists where that is a multiple of US, reading the lane's input
/// column floorDiv(., US).
constexpr int tapPos(int j, int s)
{
    return j * SW + s * DWD - PW;
}
constexpr bool tapReads(int j, int s)
{
    return tapPos(j, s) % US == 0;
}

/// The input columns a lane's window holds: PL to its left (from the lane before), its own
/// P x SW / US, and RH to its right (from the lane after).
constexpr int OWN = P * SW / US;
constexpr int PL = -floorDiv(-PW, US);
constexpr int RH_ = floorDiv((P - 1) * SW + (FS - 1) * DWD - PW, US) + 1 - OWN;
constexpr int RH = RH_ > 0 ? RH_ : 0;
constexpr int SPAN = PL + OWN + RH;
static_assert((P * SW) % US == 0, "a lane's columns are whole input columns");
static_assert(US == 1 || SW == 1, "strided and upsampled");
static_assert(PL <= OWN && RH <= OWN, "the window reaches past the neighbouring lanes");

/// The window element output column j, filter column s reads.
constexpr int tapAt(int j, int s)
{
    return floorDiv(tapPos(j, s), US) + PL;
}
static_assert(VBX >= ES && VBY >= ES, "load granule");

#if HKP_ARCH_GFX11 || HKP_ARCH_GFX12
#define GV_RSRC_FLAGS 0x31004000
#elif defined(__gfx1010__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
#define GV_RSRC_FLAGS 0x31014000
#else
#define GV_RSRC_FLAGS 0x00020000
#endif
constexpr unsigned OOB = 0x80000000u;

typedef unsigned GvU2 __attribute__((ext_vector_type(2)));
typedef unsigned GvU4 __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(4))) float* GvConstF;

__device__ inline float toFloat(Raw bits)
{
#if GV_DT == 0
    return __uint_as_float(bits);
#elif GV_DT == 1
    return static_cast<float>(__builtin_bit_cast(_Float16, bits));
#else
    return __uint_as_float(static_cast<unsigned>(bits) << 16);
#endif
}

__device__ inline Raw fromFloat(float value)
{
#if GV_DT == 0
    return __float_as_uint(value);
#elif GV_DT == 1
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

/// A buffer descriptor over [base, base + bytes) (at most 2^31 - 1 bytes) from wave-uniform
/// values (readfirstlane: no waterfall loop around the accesses).
__device__ inline __amdgpu_buffer_rsrc_t gvRsrc(const void* base, long long bytes)
{
    const int records = bytes < 0x7fffffffLL ? static_cast<int>(bytes) : 0x7fffffff;
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>((static_cast<unsigned long long>(hi) << 32) | lo),
        static_cast<short>(0),
        __builtin_amdgcn_readfirstlane(records),
        GV_RSRC_FLAGS);
}

template <int NE>
__device__ inline void putDword(Raw (&w)[NE], int at, unsigned d)
{
#if GV_DT == 0
    w[at] = d;
#else
    w[at] = static_cast<Raw>(d & 0xffffu);
    w[at + 1] = static_cast<Raw>(d >> 16);
#endif
}

template <int NE>
__device__ inline unsigned getDword(const Raw (&w)[NE], int at)
{
#if GV_DT == 0
    return w[at];
#else
    return static_cast<unsigned>(w[at]) | (static_cast<unsigned>(w[at + 1]) << 16);
#endif
}

/// NE consecutive elements from byte @p offset (out of range: zeros) in loads of VB bytes.
template <int NE, int VB>
__device__ inline void loadRaw(__amdgpu_buffer_rsrc_t r, unsigned offset, Raw (&w)[NE])
{
    constexpr int NB = NE * ES;
    static_assert(NB % VB == 0, "a window is whole loads");
    constexpr int PER = VB / ES;
#pragma unroll
    for(int i = 0; i < NB / VB; ++i)
    {
        const unsigned o = offset + static_cast<unsigned>(i * VB);
        if constexpr(VB == 16)
        {
            const GvU4 v
                = __builtin_bit_cast(GvU4, __builtin_amdgcn_raw_buffer_load_b128(r, o, 0, 0));
#pragma unroll
            for(int d = 0; d < 4; ++d)
            {
                putDword(w, i * PER + d * (4 / ES), v[d]);
            }
        }
        else if constexpr(VB == 8)
        {
            const GvU2 v
                = __builtin_bit_cast(GvU2, __builtin_amdgcn_raw_buffer_load_b64(r, o, 0, 0));
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

/// NE consecutive elements to byte @p offset (out of range: dropped) in stores of VB bytes.
template <int NE, int VB>
__device__ inline void storeRaw(__amdgpu_buffer_rsrc_t r, unsigned offset, const Raw (&w)[NE])
{
    constexpr int NB = NE * ES;
    static_assert(NB % VB == 0, "whole stores");
    constexpr int PER = VB / ES;
#pragma unroll
    for(int i = 0; i < NB / VB; ++i)
    {
        const unsigned o = offset + static_cast<unsigned>(i * VB);
        if constexpr(VB == 16)
        {
            GvU4 v;
#pragma unroll
            for(int d = 0; d < 4; ++d)
            {
                v[d] = getDword(w, i * PER + d * (4 / ES));
            }
            __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(GvU4, v), r, o, 0, 0);
        }
        else if constexpr(VB == 8)
        {
            GvU2 v;
#pragma unroll
            for(int d = 0; d < 2; ++d)
            {
                v[d] = getDword(w, i * PER + d * (4 / ES));
            }
            __builtin_amdgcn_raw_buffer_store_b64(__builtin_bit_cast(GvU2, v), r, o, 0, 0);
        }
        else if constexpr(VB == 4)
        {
            __builtin_amdgcn_raw_buffer_store_b32(getDword(w, i * PER), r, o, 0, 0);
        }
        else
        {
            static_assert(VB == 2 && ES == 2, "store granule");
            __builtin_amdgcn_raw_buffer_store_b16(w[i], r, o, 0, 0);
        }
    }
}

__device__ inline unsigned fromLanePrev(unsigned v, int lane)
{
#if GV_WAVE == 64 && !defined(__GFX10__) && !defined(__GFX11__) && !defined(__GFX12__)
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
#if GV_WAVE == 64 && !defined(__GFX10__) && !defined(__GFX11__) && !defined(__GFX12__)
    (void)lane;
    return static_cast<unsigned>(__builtin_amdgcn_update_dpp(
        0, static_cast<int>(v), 0x130 /* wave_shl:1 */, 0xF, 0xF, false));
#else
    return static_cast<unsigned>(
        __builtin_amdgcn_ds_bpermute((lane + 1) << 2, static_cast<int>(v)));
#endif
}

/// The products of output columns J.. of a lane with filter column S of one filter row: output
/// column J, (upsampled) input position J SW - PW + S DW past the lane's, window element
/// tapAt(J, S); a position between upsampled inputs has no product.
template <int J, int S>
__device__ inline void tapColumns(float (&acc)[KB][P], const Raw (&xw)[SPAN], const float (&w)[KB])
{
    if constexpr(J < P)
    {
        if constexpr(tapReads(J, S))
        {
            const float xv = toFloat(xw[tapAt(J, S)]);
#pragma unroll
            for(int k = 0; k < KB; ++k)
            {
                acc[k][J] = __builtin_fmaf(xv, w[k], acc[k][J]);
            }
        }
        tapColumns<J + 1, S>(acc, xw, w);
    }
}

/// Filter columns S.. of one filter row and input channel: @p wv its KB x FS weights (s-major).
template <int S>
__device__ inline void filterColumns(float (&acc)[KB][P], const Raw (&xw)[SPAN], GvConstF wv)
{
    if constexpr(S < FS)
    {
        float w[KB];
#pragma unroll
        for(int k = 0; k < KB; ++k)
        {
            w[k] = wv[S * KB + k];
        }
        tapColumns<0, S>(acc, xw, w);
        filterColumns<S + 1>(acc, xw, wv);
    }
}

/// Position @p q of @p count rows, ordered by phase modulo @p u: the rows 0, u, 2u, ... first, then
/// 1, u + 1, ..., and so on (u = 1: the identity).
__device__ inline int phased(int q, int count, int u)
{
    int p = 0;
    for(; p < u - 1; ++p)
    {
        const int phaseRows = (count - p + u - 1) / u;
        if(q < phaseRows)
        {
            break;
        }
        q -= phaseRows;
    }
    return p + q * u;
}

/// Whether any lane of the wave has @p v.
__device__ inline bool anyLane(bool v)
{
#if GV_WAVE == 64
    return __builtin_amdgcn_ballot_w64(v) != 0;
#else
    return __builtin_amdgcn_ballot_w32(v) != 0;
#endif
}

} // namespace

/// Writes the prepared filter: one thread an element of [group][ko block][ci][t][r][s][KB].
extern "C" __global__ void __launch_bounds__(256) ConvGroupValuPrepare(ConvGroupValuArgs a)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if(i >= a.prepared)
    {
        return;
    }
    const int kk = static_cast<int>(i % KB);
    long long rest = i / KB;
    const int tap = static_cast<int>(rest % TAPS);
    rest /= TAPS;
    const int c = static_cast<int>(rest % a.ci);
    rest /= a.ci;
    const int kb = static_cast<int>(rest % a.kBlocks);
    const int g = static_cast<int>(rest / a.kBlocks);
    const int k = kb * KB + kk; // kernel output channel within the group
    const int s = tap % FS;
    const int r = (tap / FS) % FR;
    const int t = tap / (FS * FR);
    // Backward data reflects the filter; convolution mode reflects it (again).
    const bool reflect = (a.backward != 0) != (a.flip != 0);
    const int ts = reflect ? FD - 1 - t : t;
    const int rs = reflect ? FR - 1 - r : r;
    const int ss = reflect ? FS - 1 - s : s;
    // The graph filter's (output, input) channel within the group.
    const int gk = a.backward != 0 ? c : k;
    const int gc = a.backward != 0 ? k : c;
    const long long off = (static_cast<long long>(g) * a.wGraphK + gk) * a.ws[0] + gc * a.ws[1]
                          + ts * a.ws[2] + rs * a.ws[3] + ss * a.ws[4];
    a.wt[i] = toFloat(static_cast<const Raw*>(a.w)[off]);
}

/// Grid (rowBlocks, groups x kBlocks), NW waves a workgroup. See the file comment.
extern "C" __global__ void __launch_bounds__(GV_NW* GV_WAVE) ConvGroupValu(ConvGroupValuArgs a)
{
    const int rowBlock = static_cast<int>(blockIdx.x);
    const int gk = static_cast<int>(blockIdx.y); // group x kBlocks + k block
    const int g = gk / a.kBlocks;
    const int kb = gk - g * a.kBlocks;
    const int c0 = g * a.ci; // in channel of the group's first input channel
    const int k0 = g * a.ko + kb * KB; // out channel of the wave's first output channel

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane = tid % WAVE;
    const int lanes = a.lanes;
    const int slot = lane / lanes;
    const int col = lane - slot * lanes;
    const int rpw = WAVE / lanes;
    const int ox0 = col * P;
    const bool firstCol = col == 0;
    const bool lastCol = col == lanes - 1;
    const bool colY = ox0 < a.wo;

    const int rowBegin = rowBlock * a.rowsPerBlock;
    const int rowEnd = min(a.rows, rowBegin + a.rowsPerBlock);

    // 32-bit element strides: offsets within the images a pass spans fit (host gate).
    const int isn = static_cast<int>(a.is[0]);
    const int isc = static_cast<int>(a.is[1]);
    const int isd = static_cast<int>(a.is[2]);
    const int ish = static_cast<int>(a.is[3]);
    const int isw = static_cast<int>(a.is[4]);
    const int osn = static_cast<int>(a.os[0]);
    const int osk = static_cast<int>(a.os[1]);
    const int osd = static_cast<int>(a.os[2]);
    const int osh = static_cast<int>(a.os[3]);
    const int osw = static_cast<int>(a.os[4]);
    const int rowsAnImage = a.dOut * a.ho;

    // The wave's slice of the prepared filter: [ci][t][r][s][KB], uniform.
    const GvConstF wt = __builtin_bit_cast(GvConstF, static_cast<const float*>(a.wt))
                        + static_cast<long long>(gk) * a.ci * TAPS * KB;

    for(int pr = rowBegin + wave * rpw; pr < rowEnd; pr += NW * rpw)
    {
        const int n0 = pr / rowsAnImage;
        const long long inBase
            = (static_cast<long long>(n0) * a.is[0] + static_cast<long long>(c0) * a.is[1]) * ES;
        const long long outBase
            = (static_cast<long long>(n0) * a.os[0] + static_cast<long long>(k0) * a.os[1]) * ES;
        const __amdgpu_buffer_rsrc_t ir
            = gvRsrc(static_cast<const char*>(a.in) + inBase, a.inBytes - inBase);
        const __amdgpu_buffer_rsrc_t orr
            = gvRsrc(static_cast<char*>(a.out) + outBase, a.outBytes - outBase);

        const int row = pr + slot;
        const bool rowOk = row < rowEnd;
        const int n = row / rowsAnImage;
        const int within = row - n * rowsAnImage;
        int od = within / a.ho;
        int oy = within - od * a.ho;
        if constexpr(UPROWS)
        {
            od = phased(od, a.dOut, a.ud);
            oy = phased(oy, a.ho, a.uh);
        }
        const int dn = n - n0;

        // The input row every filter row (depth tap x row) reads, and whether it exists: inside
        // in, and (upsampled) on a multiple of the stride.
        int rowOff[FD * FR];
        bool rowIn[FD * FR];
        bool rowAny[FD * FR];
#pragma unroll
        for(int t = 0; t < FD; ++t)
        {
            const int md = od * a.sd - a.pd + t * a.dd;
            const int id = UPROWS ? md / a.ud : md;
            const bool dIn = (!UPROWS || md % a.ud == 0)
                             & (static_cast<unsigned>(id) < static_cast<unsigned>(a.di));
#pragma unroll
            for(int r = 0; r < FR; ++r)
            {
                const int mh = oy * a.sh - a.ph + r * a.dh;
                const int iy = UPROWS ? mh / a.uh : mh;
                const int fr = t * FR + r;
                rowIn[fr] = rowOk & dIn & (!UPROWS || mh % a.uh == 0)
                            & (static_cast<unsigned>(iy) < static_cast<unsigned>(a.hi));
                rowOff[fr] = id * isd + iy * ish;
                rowAny[fr] = anyLane(rowIn[fr] & colY);
            }
        }

        float acc[KB][P];
#pragma unroll
        for(int k = 0; k < KB; ++k)
#pragma unroll
            for(int j = 0; j < P; ++j)
                acc[k][j] = 0.0f;

        for(int cc = 0; cc < a.ci; cc += CB)
        {
            // The input windows of CB channels and every filter row (depth tap x row).
            const int inCol = col * OWN; // the lane's first own input column
#pragma unroll
            for(int fr = 0; fr < FD * FR; ++fr)
            {
                // A filter row no lane of the wave reads (padding, or an upsampled row between
                // dy rows): nothing to load or multiply.
                if(!rowAny[fr])
                {
                    continue;
                }
                Raw own[CB][OWN];
                if constexpr(NCHW)
                {
                    const bool ok = rowIn[fr] & (inCol < a.wi);
#pragma unroll
                    for(int c = 0; c < CB; ++c)
                    {
                        const int e = dn * isn + (cc + c) * isc + rowOff[fr] + inCol;
                        loadRaw<OWN, VBX>(ir, ok ? static_cast<unsigned>(e * ES) : OOB, own[c]);
                        if constexpr(RAGGED)
                        {
#pragma unroll
                            for(int u = 0; u < OWN; ++u)
                            {
                                own[c][u] = inCol + u < a.wi ? own[c][u] : Raw(0);
                            }
                        }
                    }
                }
                else
                {
#pragma unroll
                    for(int u = 0; u < OWN; ++u)
                    {
                        const int ix = inCol + u;
                        const bool ok = rowIn[fr] & (ix < a.wi);
                        const int e = dn * isn + cc * isc + rowOff[fr] + ix * isw;
                        Raw v[CB];
                        loadRaw<CB, VBX>(ir, ok ? static_cast<unsigned>(e * ES) : OOB, v);
#pragma unroll
                        for(int c = 0; c < CB; ++c)
                        {
                            own[c][u] = v[c];
                        }
                    }
                }
                // The window: PL columns from the lane before, the lane's own, RH from the lane
                // after (zeros past the row's ends).
                Raw xw[CB][SPAN];
#pragma unroll
                for(int c = 0; c < CB; ++c)
                {
#pragma unroll
                    for(int q = 0; q < PL; ++q)
                    {
                        const unsigned v = fromLanePrev(own[c][OWN - PL + q], lane);
                        xw[c][q] = firstCol ? Raw(0) : static_cast<Raw>(v);
                    }
#pragma unroll
                    for(int u = 0; u < OWN; ++u)
                    {
                        xw[c][PL + u] = own[c][u];
                    }
#pragma unroll
                    for(int q = 0; q < RH; ++q)
                    {
                        const unsigned v = fromLaneNext(own[c][q], lane);
                        xw[c][PL + OWN + q] = lastCol ? Raw(0) : static_cast<Raw>(v);
                    }
                }
                if(rowOk & colY)
                {
#pragma unroll
                    for(int c = 0; c < CB; ++c)
                    {
                        filterColumns<0>(acc, xw[c], wt + ((cc + c) * TAPS + fr * FS) * KB);
                    }
                }
            }
        }

        // Stores: KB output channels x P columns; columns past the row are not stored.
        const bool ok = rowOk & colY;
        if constexpr(NCHW)
        {
#pragma unroll
            for(int k = 0; k < KB; ++k)
            {
                Raw v[P];
#pragma unroll
                for(int j = 0; j < P; ++j)
                {
                    v[j] = fromFloat(acc[k][j]);
                }
                const int e = dn * osn + k * osk + od * osd + oy * osh + ox0;
                if constexpr(COLMASK)
                {
#pragma unroll
                    for(int j = 0; j < P; ++j)
                    {
                        const bool in = ok & (ox0 + j < a.wo);
                        Raw one[1] = {v[j]};
                        storeRaw<1, ES>(orr, in ? static_cast<unsigned>((e + j) * ES) : OOB, one);
                    }
                }
                else
                {
                    storeRaw<P, VBY>(orr, ok ? static_cast<unsigned>(e * ES) : OOB, v);
                }
            }
        }
        else
        {
#pragma unroll
            for(int j = 0; j < P; ++j)
            {
                Raw v[KB];
#pragma unroll
                for(int k = 0; k < KB; ++k)
                {
                    v[k] = fromFloat(acc[k][j]);
                }
                const bool in = ok & (ox0 + j < a.wo);
                const int e = dn * osn + od * osd + oy * osh + (ox0 + j) * osw;
                storeRaw<KB, VBY>(orr, in ? static_cast<unsigned>(e * ES) : OOB, v);
            }
        }
    }
}

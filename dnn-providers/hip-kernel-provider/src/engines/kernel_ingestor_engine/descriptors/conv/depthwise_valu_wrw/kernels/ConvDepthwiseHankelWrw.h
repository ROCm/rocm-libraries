// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The matrix-core partial kernel of the depthwise backward weights (CONV_DW_VARIANT 2):
// hipconv's Hankel construction on 4x4x4 MFMA with 16 channels as the instruction's 16 blocks,
// reimplemented. Channels-last, 16-bit, unit dilation, compile-time filter and stride S (1 or 2,
// both axes).
//
// For input row y, channel c and a 4-column step q of the output tile:
//   A[s][k] = x[y][S (q + k) + s - pw]      (Hankel: the shift s on the column)
//   B[k][r] = dy[(y + ph - r) / S][q + k]   (the filter rows' dy rows as columns; zero where S
//                                           does not divide y + ph - r)
//   C[s][r] += sum_k A[s][k] B[k][r]        = dw[c][r][s] over this row and step
// Filters wider or taller than 4 tile into 4 x 4 tap groups (each its own accumulator).
//
// Non-finite x (parity with the reference): no product whose dy operand is not one of the
// convolution's is kept, so an inf / NaN in x never meets a (row, column) pairing the convolution
// does not have. Columns: where wo is not whole tiles (CONV_DW_COLMASK, set by the host) every
// step ANDs the lane's A with a block-uniform mask of the columns before wo (all ones but in the
// last tile; a branch there would split the step's block and cost more). Rows: an input row
// with a real filter row that has no dy row here (outside the image or the partition, or at
// stride 2 the other parity; a block-uniform test) saves the accumulators before its products
// and restores them after, in the lanes of those filter rows: a few moves a row, not a select a
// product.
//
// At stride 2 the x row is staged in two phase rows (its even and its odd columns), so A[s][k]
// = phase (s % 2) at k + s / 2: again a shift along one row, and the operand read stays one
// 64-bit window per lane.
//
// A block: 4 waves x CONV_DW_NWG groups of 16 channels, a partition of output rows (one image
// segment at a time), every CONV_DW_Q-column tile of those rows. Per step it stages CONV_DW_G
// input rows: their x rows and the dy rows entering the ring (LDS, channel-major rows; dy keeps
// the KH + G - 1 rows a step reads, plus a zero slot for taps with no row). The next step's
// global loads are in flight while a step multiplies. One fp32 partial per (partition, channel,
// tap); the fold adds them.
//
// LDS rows are plain (no swizzle), at a pitch of an odd number of dword pairs, so the 16 channel
// rows of a wave's operand read land on distinct banks; a lane reads its A window from its own
// dword offset (no per-lane select of dwords). Ring slots sit 8 dwords apart modulo 32 banks.
// Staging loads are buffer loads: rows and columns outside the image, and channels past c, read
// zeros through out-of-range offsets (the host keeps each image under 2 GiB).
#pragma once

#ifndef CONV_DW_Q
#define CONV_DW_Q 32
#endif
#ifndef CONV_DW_G
#define CONV_DW_G 1
#endif
#ifndef CONV_DW_NWG
#define CONV_DW_NWG 1
#endif
#ifndef CONV_DW_S
#define CONV_DW_S 1
#endif
#ifndef CONV_DW_COLMASK
#define CONV_DW_COLMASK 0
#endif

namespace hankel
{
constexpr int KH = CONV_DW_KH, KW = CONV_DW_KW, Q = CONV_DW_Q, G = CONV_DW_G;
constexpr int NWG = CONV_DW_NWG; // 16-channel groups a wave
constexpr int RG = (KH + 3) / 4, SG = (KW + 3) / 4;
constexpr int CB = 64 * NWG; // channels a block
constexpr int S = CONV_DW_S; // stride, both axes
constexpr bool COLMASK = CONV_DW_COLMASK != 0; // wo not whole tiles
constexpr int THREADS = 256;
static_assert(CONV_DW_KD == 1, "2-D filters");
static_assert(Q % 4 == 0 && Q >= 4, "whole 4-column steps");
static_assert(S == 1 || S == 2, "unit or 2 stride");
static_assert(S == 1 || SG <= 2, "stride 2 reads two words a tap group");
// Four-column words of an x (phase) row a tile reads: the tile's Q columns and what its taps
// reach past them (KW - 1 columns; at stride 2, (KW - 1) / 2 a phase row, rounded to the words
// a 64-bit window can start in).
constexpr int XG = S == 1 ? Q / 4 + SG : Q / 4 + (SG - 1) / 2 + 1;
constexpr int DG = Q / 4; // of a dy row
// Row pitches (halves): whole dwords, an odd number of dword pairs, so 16 channel rows read at
// one dword offset (or the next) land on distinct banks.
constexpr int pitchOf(int dwords)
{
    return 2 * (((dwords + 2) / 2) % 2 == 1 ? dwords + 2 : dwords + 4);
}
constexpr int XP = pitchOf(2 * XG), DP = pitchOf(2 * DG);
/// A row's skew (halves): every 8 rows (a staging thread's channel vectors) start two dwords
/// further along, so rows 8 apart land on distinct banks. The skew only grows with the row, so
/// rows never overlap and stay contiguous for the window reads.
constexpr int skewOf(int row)
{
    return 4 * (row >> 3);
}
/// The stride-2 odd phase rows continue the even ones: row CB + c (CB / 2 halves of skew on, a
/// bank offset of 16 dwords from the even row).
constexpr int PHASE = CB * XP + CB / 2;
constexpr int XB = S * CB * XP + S * CB / 2; // an x row buffer (halves)
constexpr int SS = (CB * DP + CB / 2 + 63) / 64 * 64 + 16; // ring slot stride (halves)
constexpr int SLOTS = KH + G - 1; // ring slots; slot SLOTS is zeros
// Staging units: 4 x columns x 8 channels. At stride 2 a phase row's XG words take 2 XG units.
constexpr int XU = S * XG * (CB / 8), DU = DG * (CB / 8);
constexpr int XPT = (G * XU + THREADS - 1) / THREADS, DPT = (G * DU + THREADS - 1) / THREADS;
/// The LDS the kernel declares (bytes); the host gates it against the device's.
constexpr int LDS_BYTES = (G * XB + (SLOTS + 1) * SS) * 2;

typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
typedef unsigned short u16;
typedef unsigned long long u64;

constexpr unsigned OOB = 0x80000000u; // a buffer offset past every record: loads zeros

/// A buffer descriptor over [base, base + bytes), bytes < 2^31 (the host gates the images).
__device__ inline __amdgpu_buffer_rsrc_t rsrcOf(const void* base, long bytes)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>((static_cast<unsigned long long>(hi) << 32) | lo),
        static_cast<short>(0),
        __builtin_amdgcn_readfirstlane(static_cast<int>(bytes)),
        0x00020000);
}

template <bool BF16>
__device__ inline f4 mma(u64 a, u64 b, f4 c)
{
    if constexpr(BF16)
    {
        return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
            __builtin_bit_cast(s4, a), __builtin_bit_cast(s4, b), c, 0, 0, 0);
    }
    else
    {
        return __builtin_amdgcn_mfma_f32_4x4x4f16(
            __builtin_bit_cast(h4, a), __builtin_bit_cast(h4, b), c, 0, 0, 0);
    }
}

/// A thread's N staging units (4 columns x 8 channels, 16-byte loads) of a step's G rows of
/// UNITS units each, in registers.
template <int N, int UNITS>
struct Stage
{
    u4 v[N][4];

    /// Row g's elements start at rowOff[g] in the image (-1: no row); columns col0 + ... of
    /// `width`, channels c0 + cv of c. Out-of-range loads read zeros (OOB offsets).
    __device__ void load(__amdgpu_buffer_rsrc_t r,
                         const int (&rowOff)[G],
                         int col0,
                         int width,
                         int c,
                         int c0,
                         int tid)
    {
#pragma unroll
        for(int u = 0; u < N; ++u)
        {
            const int all = tid + u * THREADS, gr = all / UNITS, unit = all % UNITS;
            const int g = unit / (CB / 8), cv = (unit % (CB / 8)) * 8;
            const int off = rowOff[gr < G ? gr : 0];
            const bool live = gr < G && off >= 0 && c0 + cv + 8 <= c;
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                const int ix = col0 + 4 * g + j;
                const unsigned o = live && ix >= 0 && ix < width
                                       ? static_cast<unsigned>((off + ix * c + c0 + cv) * 2)
                                       : OOB;
                v[u][j] = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(r, o, 0, 0));
            }
        }
    }

    /// Writes channel-major rows dst[gr][ch * pitch + column]: per channel one 64-bit word of
    /// its 4 columns (byte permutes of the column vectors); rows with no destination skip.
    __device__ void store(u16* const (&dst)[G], int pitch, int tid) const
    {
#pragma unroll
        for(int u = 0; u < N; ++u)
        {
            const int all = tid + u * THREADS, gr = all / UNITS, unit = all % UNITS;
            if(gr >= G || dst[gr] == nullptr)
            {
                continue;
            }
            const int g = unit / (CB / 8), cv = (unit % (CB / 8)) * 8;
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                const unsigned sel = e % 2 == 0 ? 0x05040100u : 0x07060302u;
                const u64 w
                    = (u64)__builtin_amdgcn_perm(v[u][1][e / 2], v[u][0][e / 2], sel)
                      | ((u64)__builtin_amdgcn_perm(v[u][3][e / 2], v[u][2][e / 2], sel) << 32);
                *reinterpret_cast<u64*>(dst[gr] + (cv + e) * pitch + skewOf(cv + e) + 4 * g) = w;
            }
        }
    }

    /// Stride 2: writes the unit's even columns (0, 2) to phase row 0 and its odd columns (1, 3)
    /// to phase row 1, `phase` halves after it: per channel one 32-bit half word of each.
    __device__ void storePhases(u16* const (&dst)[G], int pitch, int phase, int tid) const
    {
#pragma unroll
        for(int u = 0; u < N; ++u)
        {
            const int all = tid + u * THREADS, gr = all / UNITS, unit = all % UNITS;
            if(gr >= G || dst[gr] == nullptr)
            {
                continue;
            }
            const int g = unit / (CB / 8), cv = (unit % (CB / 8)) * 8;
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                const unsigned sel = e % 2 == 0 ? 0x05040100u : 0x07060302u;
                const int at = (cv + e) * pitch + skewOf(cv + e) + 2 * g;
                *reinterpret_cast<unsigned*>(dst[gr] + at)
                    = __builtin_amdgcn_perm(v[u][2][e / 2], v[u][0][e / 2], sel);
                *reinterpret_cast<unsigned*>(dst[gr] + phase + at)
                    = __builtin_amdgcn_perm(v[u][3][e / 2], v[u][1][e / 2], sel);
            }
        }
    }
};

template <bool BF16>
__device__ void partial(const ConvDepthwiseValuWrwArgs& a)
{
    __shared__ __attribute__((aligned(16))) u16 xs[G][XB];
    __shared__ __attribute__((aligned(16))) u16 ring[(SLOTS + 1) * SS];
    const int tid = static_cast<int>(threadIdx.x), wave = tid / 64, lane = tid % 64;
    const int cBlocks = (a.c + CB - 1) / CB;
    const int partition = static_cast<int>(blockIdx.x) / cBlocks;
    const int c0 = (static_cast<int>(blockIdx.x) % cBlocks) * CB;
    const int b = wave * 16 + lane / 4; // the lane's channel (of its first group) in the block
    const int i = lane % 4; // its A row (s) and B column (r)
    const auto* x = static_cast<const u16*>(a.x);
    const auto* dy = static_cast<const u16*>(a.dy);
    // The lane's A window: stride 1, i halves into the row (its dword offset i / 2 goes in the
    // read address, the odd half is a 2-byte align); stride 2, phase row i % 2, (i / 2) halves
    // in (the align).
    const int laneDword = S == 1 ? i >> 1 : 0;
    const unsigned ab = S == 1 ? 2 * (i & 1) : 2 * (i >> 1);

    for(int e = tid; e < SS; e += THREADS)
    {
        ring[SLOTS * SS + e] = 0;
    }
    f4 acc[NWG][RG][SG];
#pragma unroll
    for(int cg = 0; cg < NWG; ++cg)
#pragma unroll
        for(int rg = 0; rg < RG; ++rg)
#pragma unroll
            for(int sg = 0; sg < SG; ++sg)
                acc[cg][rg][sg] = f4{0, 0, 0, 0};

    Stage<XPT, XU> sx;
    Stage<DPT, DU> sd;
    const int first = partition * a.rowsPerPartition;
    const int last = min(a.rows, first + a.rowsPerPartition);
    for(int seg = first; seg < last;)
    {
        // An image's output rows [o0, o1) of the partition.
        const int n = seg / a.ho, o0 = seg % a.ho;
        const int o1 = min(a.ho, o0 + (last - seg));
        seg += o1 - o0;
        const auto xr = rsrcOf(x + (long)n * a.hi * a.wi * a.c, (long)a.hi * a.wi * a.c * 2);
        const auto dr = rsrcOf(dy + (long)n * a.ho * a.wo * a.c, (long)a.ho * a.wo * a.c * 2);
        for(int q0 = 0; q0 < a.wo; q0 += Q)
        {
            // The input rows whose taps reach [o0, o1): y = S oy - ph + r. Output row oy enters
            // the ring at its first input row, S oy - ph.
            const int y0 = o0 * S - a.ph, y1 = (o1 - 1) * S - a.ph + KH;
            auto entering = [&](int yy, int& oy) {
                const int num = yy + a.ph; // >= S o0 >= 0
                oy = num / S;
                return yy < y1 && num % S == 0 && oy >= o0 && oy < o1;
            };
            auto loadStep = [&](int y) {
                int xo[G], dyo[G];
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g;
                    int oy = 0;
                    xo[g] = yy < y1 && yy >= 0 && yy < a.hi ? yy * a.wi * a.c : -1;
                    dyo[g] = entering(yy, oy) ? oy * a.wo * a.c : -1;
                }
                sx.load(xr, xo, q0 * S - a.pw, a.wi, a.c, c0, tid);
                sd.load(dr, dyo, q0, a.wo, a.c, c0, tid);
            };
            auto storeStep = [&](int y) {
                u16* xd[G];
                u16* dd[G];
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g;
                    int oy = 0;
                    xd[g] = xs[g];
                    dd[g] = entering(yy, oy) ? ring + (oy % SLOTS) * SS : nullptr;
                }
                if constexpr(S == 1)
                {
                    sx.store(xd, XP, tid);
                }
                else
                {
                    sx.storePhases(xd, XP, PHASE, tid);
                }
                sd.store(dd, DP, tid);
            };
            __syncthreads(); // the previous tile's reads are done
            loadStep(y0);
            storeStep(y0);
            __syncthreads();
            for(int y = y0; y < y1; y += G)
            {
                if(y + G < y1)
                {
                    loadStep(y + G);
                }
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g;
                    if(yy >= y1 || yy < 0 || yy >= a.hi)
                    {
                        continue; // a padding row's window is zeros
                    }
                    // The lane's B rows r = 4 rg + i: dy row (yy + ph - r) / S, or the zero slot.
                    // A lane whose real filter row has no dy row keeps its sums (rows past KH
                    // are never written out): the row is careful when any filter row of it has
                    // none (block-uniform; at stride 2 every row, by parity).
                    const u16* bSlot[RG];
                    bool keep[RG];
#pragma unroll
                    for(int rg = 0; rg < RG; ++rg)
                    {
                        const int r = 4 * rg + i, num = yy + a.ph - r, oy = num / S;
                        const bool ok = r < KH && num >= o0 * S && num % S == 0 && oy < o1;
                        bSlot[rg] = ring + (ok ? oy % SLOTS : SLOTS) * SS + b * DP + skewOf(b);
                        keep[rg] = ok || r >= KH;
                    }
                    const bool careful
                        = S != 1 || yy + a.ph - (KH - 1) < o0 * S || yy + a.ph >= o1 * S;
                    f4 saved[NWG][RG][SG];
                    if(careful)
                    {
#pragma unroll
                        for(int cg = 0; cg < NWG; ++cg)
#pragma unroll
                            for(int rg = 0; rg < RG; ++rg)
#pragma unroll
                                for(int sg = 0; sg < SG; ++sg)
                                    saved[cg][rg][sg] = acc[cg][rg][sg];
                    }
#pragma unroll
                    for(int cg = 0; cg < NWG; ++cg)
                    {
                        // The lane's channel row as dwords, from its window's first dword.
                        const unsigned* aRow = reinterpret_cast<const unsigned*>(
                                                   xs[g] + (S == 1 ? 0 : (i & 1) * PHASE)
                                                   + (b + 64 * cg) * XP + skewOf(b + 64 * cg))
                                               + laneDword;
#pragma unroll
                        for(int qq = 0; qq < Q; qq += 4)
                        {
                            // The dwords the tap groups' windows span: stride 1, tap group sg's
                            // window starts at dword 2 sg; stride 2, at dword sg.
                            constexpr int SPAN = S == 1 ? 2 * SG + 1 : SG + 2;
                            unsigned d[SPAN];
#pragma unroll
                            for(int k = 0; k < SPAN; ++k)
                            {
                                d[k] = aRow[qq / 2 + k];
                            }
                            u64 av[SG], bv[RG];
                            // A: 4 halves from the window's first dword, then a 2-byte align.
#pragma unroll
                            for(int sg = 0; sg < SG; ++sg)
                            {
                                const int w0 = S == 1 ? 2 * sg : sg;
                                av[sg]
                                    = (u64)__builtin_amdgcn_alignbyte(d[w0 + 1], d[w0], ab)
                                      | ((u64)__builtin_amdgcn_alignbyte(d[w0 + 2], d[w0 + 1], ab)
                                         << 32);
                            }
                            if constexpr(COLMASK)
                            {
                                // A's k is output column q0 + qq + k: in a step past wo (the last
                                // tile's), zero the halves past it.
                                // (A block-uniform mask, no branch: the step stays one block.)
                                const int live = a.wo - (q0 + qq);
                                const u64 mask = live >= 4   ? ~0ull
                                                 : live <= 0 ? 0ull
                                                             : (1ull << (16 * live)) - 1;
#pragma unroll
                                for(int sg = 0; sg < SG; ++sg)
                                {
                                    av[sg] &= mask;
                                }
                            }
#pragma unroll
                            for(int rg = 0; rg < RG; ++rg)
                            {
                                bv[rg] = *reinterpret_cast<const u64*>(bSlot[rg] + 64 * cg * DP
                                                                       + 32 * cg + qq);
                            }
#pragma unroll
                            for(int rg = 0; rg < RG; ++rg)
#pragma unroll
                                for(int sg = 0; sg < SG; ++sg)
                                    acc[cg][rg][sg] = mma<BF16>(av[sg], bv[rg], acc[cg][rg][sg]);
                        }
                    }
                    if(careful)
                    {
#pragma unroll
                        for(int cg = 0; cg < NWG; ++cg)
#pragma unroll
                            for(int rg = 0; rg < RG; ++rg)
#pragma unroll
                                for(int sg = 0; sg < SG; ++sg)
                                    acc[cg][rg][sg]
                                        = keep[rg] ? acc[cg][rg][sg] : saved[cg][rg][sg];
                    }
                }
                __syncthreads();
                if(y + G < y1)
                {
                    storeStep(y + G);
                }
                __syncthreads();
            }
        }
    }
    // D: lane (channel b, j = lane % 4); VGPR v holds C[s = v][r = j].
#pragma unroll
    for(int cg = 0; cg < NWG; ++cg)
    {
        const int c = c0 + b + 64 * cg;
        if(c >= a.c)
        {
            continue;
        }
        float* out = a.partial + ((long)partition * a.c + c) * a.taps;
#pragma unroll
        for(int rg = 0; rg < RG; ++rg)
#pragma unroll
            for(int sg = 0; sg < SG; ++sg)
#pragma unroll
                for(int v = 0; v < 4; ++v)
                {
                    const int r = 4 * rg + i, s = 4 * sg + v;
                    if(r < KH && s < KW)
                    {
                        out[r * KW + s] = acc[cg][rg][sg][v];
                    }
                }
    }
}
} // namespace hankel

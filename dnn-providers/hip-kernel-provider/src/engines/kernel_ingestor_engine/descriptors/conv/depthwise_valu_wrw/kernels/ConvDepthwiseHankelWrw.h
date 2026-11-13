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
// Non-finite x (parity with the reference): a product whose dy operand is not one of the
// convolution's is never accumulated, so an inf / NaN in x never meets a (row, column) pairing the
// convolution does not have. The staging marks a step whose x rows hold a non-finite element (one
// block-wide vote with the step's barrier); only such a step multiplies carefully, so finite data
// runs the plain loop. Careful columns: past wo the lane's A halves are zeroed (CONV_DW_COLMASK, set
// by the host when wo is not whole tiles; otherwise the tiles never pass wo). Careful rows: a lane
// whose filter row r has no dy row here (outside the image or partition, or at stride 2 the other
// parity) keeps its accumulator instead of the product's.
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
// LDS rows are whole groups of 8 four-column words, the word index XORed with (row / 8) % 8, at
// a pitch of 2 words + 2 dwords; ring slots sit 8 dwords apart modulo 32 banks. The 64-bit
// operand reads of 4 channel rows by 4 ring slots, and the staging writes of 8 rows x 2 words,
// then land on distinct banks.
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
constexpr int XW = (XG + 7) / 8 * 8, DW = (DG + 7) / 8 * 8;
constexpr int XP = 4 * XW + 4, DP = 4 * DW + 4; // row pitches (halves)
constexpr int SS = (CB * DP + 63) / 64 * 64 + 16; // ring slot stride (halves)
constexpr int SLOTS = KH + G - 1; // ring slots; slot SLOTS is zeros
// Staging units: 4 x columns x 8 channels. At stride 2 a phase row's XG words take 2 XG units.
constexpr int XU = S * XG * (CB / 8), DU = DG * (CB / 8);
constexpr int XPT = (G * XU + THREADS - 1) / THREADS, DPT = (G * DU + THREADS - 1) / THREADS;
/// The LDS the kernel declares (bytes); the host gates it against the device's.
constexpr int LDS_BYTES = (G * S * CB * XP + (SLOTS + 1) * SS) * 2;

typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
typedef unsigned short u16;
typedef unsigned long long u64;

__device__ inline int swz(int row, int word)
{
    return word ^ ((row >> 3) & 7);
}

template <bool B>
struct Flag
{
    static constexpr bool value = B;
};

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

    __device__ void load(const u16* const (&row)[G],
                         const bool (&rowOk)[G],
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
            const int grc = gr < G ? gr : 0;
            const u16* src = row[grc];
            const bool live = gr < G && rowOk[grc] && c0 + cv + 8 <= c;
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                const int ix = col0 + 4 * g + j;
                v[u][j] = live && ix >= 0 && ix < width
                              ? *reinterpret_cast<const u4*>(src + (long)ix * c + c0 + cv)
                              : u4{0, 0, 0, 0};
            }
        }
    }

    /// Whether a staged element is inf or NaN (all exponent bits set).
    template <bool BF16>
    __device__ int nonFinite() const
    {
        constexpr unsigned EXP = BF16 ? 0x7f80u : 0x7c00u;
        unsigned any = 0;
#pragma unroll
        for(int u = 0; u < N; ++u)
#pragma unroll
            for(int j = 0; j < 4; ++j)
#pragma unroll
                for(int e = 0; e < 4; ++e)
                {
                    const unsigned w = v[u][j][e];
                    any |= ((w & EXP) == EXP) | (((w >> 16) & EXP) == EXP);
                }
        return static_cast<int>(any);
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
                *reinterpret_cast<u64*>(dst[gr] + (cv + e) * pitch + 4 * swz(cv + e, g)) = w;
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
                const int at = (cv + e) * pitch + 4 * swz(cv + e, g / 2) + 2 * (g % 2);
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
    __shared__ __attribute__((aligned(16))) u16 xs[G][S * CB * XP];
    __shared__ __attribute__((aligned(16))) u16 ring[(SLOTS + 1) * SS];
    const int tid = static_cast<int>(threadIdx.x), wave = tid / 64, lane = tid % 64;
    const int cBlocks = (a.c + CB - 1) / CB;
    const int partition = static_cast<int>(blockIdx.x) / cBlocks;
    const int c0 = (static_cast<int>(blockIdx.x) % cBlocks) * CB;
    const int b = wave * 16 + lane / 4; // the lane's channel (of its first group) in the block
    const int i = lane % 4; // its A row (s) and B column (r)
    const auto* x = static_cast<const u16*>(a.x);
    const auto* dy = static_cast<const u16*>(a.dy);
    // The lane's swizzled word offsets in its channel rows (a group's rows are 64 apart, so
    // the swizzle is the same for each), and its A shift: i halves.
    int offW[XW];
#pragma unroll
    for(int w = 0; w < XW; ++w)
    {
        offW[w] = 4 * swz(b, w);
    }
    // Stride 1: the lane's A starts i halves into the window. Stride 2: in phase row i % 2,
    // (i / 2) halves in, plus 2 halves for an odd tap group (sg / 2 words in).
    const bool hi2 = S == 1 && i >= 2;
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
        const u16* xImg = x + (long)n * a.hi * a.wi * a.c;
        const u16* dyImg = dy + (long)n * a.ho * a.wo * a.c;
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
                const u16* xr[G];
                const u16* dr[G];
                bool xo[G], dyo[G];
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g;
                    int oy = 0;
                    xo[g] = yy < y1 && yy >= 0 && yy < a.hi;
                    dyo[g] = entering(yy, oy);
                    xr[g] = xImg + (long)(xo[g] ? yy : 0) * a.wi * a.c;
                    dr[g] = dyImg + (long)(dyo[g] ? oy : 0) * a.wo * a.c;
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
                    sx.storePhases(xd, XP, CB * XP, tid);
                }
                sd.store(dd, DP, tid);
            };
            __syncthreads(); // the previous tile's reads are done
            loadStep(y0);
            storeStep(y0);
            bool careful = __syncthreads_or(sx.template nonFinite<BF16>()) != 0;
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
                    // are never written out).
                    const u16* bSlot[RG];
                    bool keep[RG];
#pragma unroll
                    for(int rg = 0; rg < RG; ++rg)
                    {
                        const int r = 4 * rg + i, num = yy + a.ph - r, oy = num / S;
                        const bool ok = r < KH && num >= o0 * S && num % S == 0 && oy < o1;
                        bSlot[rg] = ring + (ok ? oy % SLOTS : SLOTS) * SS + b * DP;
                        keep[rg] = ok || r >= KH;
                    }
                    auto multiply = [&](auto tag) {
                        constexpr bool CAREFUL = decltype(tag)::value;
#pragma unroll
                        for(int cg = 0; cg < NWG; ++cg)
                        {
                            const u16* aRow
                                = xs[g] + (S == 1 ? 0 : (i & 1) * CB * XP) + (b + 64 * cg) * XP;
#pragma unroll
                            for(int qq = 0; qq < Q; qq += 4)
                            {
                                // The words the tap groups' windows start in, and the next.
                                constexpr int WORDS = S == 1 ? SG + 1 : (SG - 1) / 2 + 2;
                                unsigned d[2 * WORDS];
#pragma unroll
                                for(int k = 0; k < WORDS; ++k)
                                {
                                    const u64 w
                                        = *reinterpret_cast<const u64*>(aRow + offW[qq / 4 + k]);
                                    d[2 * k] = static_cast<unsigned>(w);
                                    d[2 * k + 1] = static_cast<unsigned>(w >> 32);
                                }
                                u64 av[SG], bv[RG];
                                // A: the 4 halves from the lane's shift: whole dwords, then a 2-byte
                                // align. Stride 1: word sg, i halves in. Stride 2: word sg / 2,
                                // 2 (sg % 2) + i / 2 halves in.
#pragma unroll
                                for(int sg = 0; sg < SG; ++sg)
                                {
                                    const int w0 = S == 1 ? sg : sg / 2;
                                    const bool hi = S == 1 ? hi2 : sg % 2 == 1;
                                    const unsigned e0 = hi ? d[2 * w0 + 1] : d[2 * w0];
                                    const unsigned e1 = hi ? d[2 * w0 + 2] : d[2 * w0 + 1];
                                    const unsigned e2 = hi ? d[2 * w0 + 3] : d[2 * w0 + 2];
                                    av[sg] = (u64)__builtin_amdgcn_alignbyte(e1, e0, ab)
                                             | ((u64)__builtin_amdgcn_alignbyte(e2, e1, ab) << 32);
                                }
                                if constexpr(COLMASK && CAREFUL)
                                {
                                    // A's k is output column q0 + qq + k: zero the halves past wo.
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
                                                                           + offW[qq / 4]);
                                }
#pragma unroll
                                for(int rg = 0; rg < RG; ++rg)
#pragma unroll
                                    for(int sg = 0; sg < SG; ++sg)
                                    {
                                        const f4 sum = mma<BF16>(av[sg], bv[rg], acc[cg][rg][sg]);
                                        acc[cg][rg][sg]
                                            = !CAREFUL || keep[rg] ? sum : acc[cg][rg][sg];
                                    }
                            }
                        }
                    };
                    if(careful)
                    {
                        multiply(Flag<true>{});
                    }
                    else
                    {
                        multiply(Flag<false>{});
                    }
                }
                __syncthreads();
                if(y + G < y1)
                {
                    storeStep(y + G);
                }
                careful = __syncthreads_or(y + G < y1 ? sx.template nonFinite<BF16>() : 0) != 0;
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

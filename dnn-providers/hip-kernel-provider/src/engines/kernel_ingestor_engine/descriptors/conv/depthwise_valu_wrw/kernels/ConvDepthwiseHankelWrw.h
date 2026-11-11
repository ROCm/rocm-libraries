// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The matrix-core partial kernel of the depthwise backward weights (CONV_DW_VARIANT 2):
// hipconv's Hankel construction on 4x4x4 MFMA with 16 channels as the instruction's 16 blocks,
// reimplemented. Channels-last, 16-bit, unit stride and dilation, compile-time filter.
//
// For input row y, channel c and a 4-column step q of the output tile:
//   A[s][k] = x[y][q + k + s - pw]      (Hankel: the shift s on the column)
//   B[k][r] = dy[y + ph - r][q + k]     (the filter rows' dy rows as columns)
//   C[s][r] += sum_k A[s][k] B[k][r]    = dw[c][r][s] over this row and step
// Filters wider or taller than 4 tile into 4 x 4 tap groups (each its own accumulator).
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

namespace hankel
{
constexpr int KH = CONV_DW_KH, KW = CONV_DW_KW, Q = CONV_DW_Q, G = CONV_DW_G;
constexpr int NWG = CONV_DW_NWG; // 16-channel groups a wave
constexpr int RG = (KH + 3) / 4, SG = (KW + 3) / 4;
constexpr int CB = 64 * NWG; // channels a block
constexpr int THREADS = 256;
static_assert(CONV_DW_KD == 1, "2-D filters");
static_assert(Q % 4 == 0 && Q >= 4, "whole 4-column steps");
constexpr int XG = Q / 4 + SG; // four-column words of the x row a tile reads
constexpr int DG = Q / 4; // of a dy row
constexpr int XW = (XG + 7) / 8 * 8, DW = (DG + 7) / 8 * 8;
constexpr int XP = 4 * XW + 4, DP = 4 * DW + 4; // row pitches (halves)
constexpr int SS = (CB * DP + 63) / 64 * 64 + 16; // ring slot stride (halves)
constexpr int SLOTS = KH + G - 1; // ring slots; slot SLOTS is zeros
constexpr int XU = XG * (CB / 8), DU = DG * (CB / 8); // staging units: 4 columns x 8 channels
constexpr int XPT = (G * XU + THREADS - 1) / THREADS, DPT = (G * DU + THREADS - 1) / THREADS;
/// The LDS the kernel declares (bytes); the host gates it against the device's.
constexpr int LDS_BYTES = (G * CB * XP + (SLOTS + 1) * SS) * 2;

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
};

template <bool BF16>
__device__ void partial(const ConvDepthwiseValuWrwArgs& a)
{
    __shared__ __attribute__((aligned(16))) u16 xs[G][CB * XP];
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
    const bool hi2 = i >= 2;
    const unsigned ab = 2 * (i & 1);

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
            // The input rows whose taps reach [o0, o1): y = oy - ph + r.
            const int y0 = o0 - a.ph, y1 = o1 - a.ph + KH - 1;
            auto loadStep = [&](int y) {
                const u16* xr[G];
                const u16* dr[G];
                bool xo[G], dyo[G];
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g, oy = yy + a.ph;
                    xo[g] = yy < y1 && yy >= 0 && yy < a.hi;
                    dyo[g] = yy < y1 && oy >= o0 && oy < o1;
                    xr[g] = xImg + (long)(xo[g] ? yy : 0) * a.wi * a.c;
                    dr[g] = dyImg + (long)(dyo[g] ? oy : 0) * a.wo * a.c;
                }
                sx.load(xr, xo, q0 - a.pw, a.wi, a.c, c0, tid);
                sd.load(dr, dyo, q0, a.wo, a.c, c0, tid);
            };
            auto storeStep = [&](int y) {
                u16* xd[G];
                u16* dd[G];
#pragma unroll
                for(int g = 0; g < G; ++g)
                {
                    const int yy = y + g, oy = yy + a.ph;
                    xd[g] = xs[g];
                    dd[g] = yy < y1 && oy >= o0 && oy < o1 ? ring + (oy % SLOTS) * SS : nullptr;
                }
                sx.store(xd, XP, tid);
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
                    // The lane's B rows r = 4 rg + i: dy row yy + ph - r, or the zero slot.
                    const u16* bSlot[RG];
#pragma unroll
                    for(int rg = 0; rg < RG; ++rg)
                    {
                        const int r = 4 * rg + i, oy = yy + a.ph - r;
                        const bool ok = r < KH && oy >= o0 && oy < o1;
                        bSlot[rg] = ring + (ok ? oy % SLOTS : SLOTS) * SS + b * DP;
                    }
#pragma unroll
                    for(int cg = 0; cg < NWG; ++cg)
                    {
                        const u16* aRow = xs[g] + (b + 64 * cg) * XP;
#pragma unroll
                        for(int qq = 0; qq < Q; qq += 4)
                        {
                            unsigned d[2 * (SG + 1)];
#pragma unroll
                            for(int k = 0; k <= SG; ++k)
                            {
                                const u64 w
                                    = *reinterpret_cast<const u64*>(aRow + offW[qq / 4 + k]);
                                d[2 * k] = static_cast<unsigned>(w);
                                d[2 * k + 1] = static_cast<unsigned>(w >> 32);
                            }
                            u64 av[SG], bv[RG];
                            // A: the 4 halves from the lane's shift i: whole dwords (i / 2),
                            // then a 2-byte align.
#pragma unroll
                            for(int sg = 0; sg < SG; ++sg)
                            {
                                const unsigned e0 = hi2 ? d[2 * sg + 1] : d[2 * sg];
                                const unsigned e1 = hi2 ? d[2 * sg + 2] : d[2 * sg + 1];
                                const unsigned e2 = hi2 ? d[2 * sg + 3] : d[2 * sg + 2];
                                av[sg] = (u64)__builtin_amdgcn_alignbyte(e1, e0, ab)
                                         | ((u64)__builtin_amdgcn_alignbyte(e2, e1, ab) << 32);
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
                                    acc[cg][rg][sg] = mma<BF16>(av[sg], bv[rg], acc[cg][rg][sg]);
                        }
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

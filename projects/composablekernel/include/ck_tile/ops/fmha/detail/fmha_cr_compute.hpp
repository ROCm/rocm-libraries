// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/detail/fmha_cr_memory.hpp"

// Keep QK/PV operand lifetimes and scheduling regions together.
namespace ck_tile::detail::fmha_cr {

struct Arguments
{
    const void *q, *k, *v;
    void* o;
    float* lse;
    int batch, heads, sq, sk;
    float scale;
};

// Runtime tensor views for GQA, strided layouts and packed variable-length batches.
// The compact Arguments ABI remains available for the measured BSHD specializations.
struct StridedArguments : Arguments
{
    int kv_heads;
    int stride_q, stride_k, stride_v, stride_o;
    int64_t head_q, head_k, head_v, head_o, head_lse;
    int64_t batch_q, batch_k, batch_v, batch_o, batch_lse;
    const int32_t *start_q, *start_k, *length_q, *length_k, *cumulative_q, *cumulative_k;
    int mask_type, left, right;
    bool group;
};

using F8 = float __attribute__((ext_vector_type(8)));
using F2 = float __attribute__((ext_vector_type(2)));

template <class T>
CK_TILE_DEVICE F8 mma(V16<T> a, V16<T> b, F8 c)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    if constexpr(std::is_same_v<T, __bf16>)
        return __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, 0, c, 0, 0);
    else
        return __builtin_amdgcn_wmma_f32_16x16x32_f16(0, a, 0, b, 0, c, 0, 0);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

template <int Width = 128, class T>
CK_TILE_DEVICE V16<T> read_k32(const T* p, int pitch, int row, int k)
{
    const int half = (threadIdx.x % 32) / 16;
    if constexpr(Width == 192)
        k += (k / 64) * 8;
    V16<T> r{};
    V8<T> a, b;
    __builtin_memcpy(&a, __builtin_assume_aligned(p + row * pitch + k + half * 8, 16), sizeof(a));
    __builtin_memcpy(
        &b, __builtin_assume_aligned(p + row * pitch + k + 16 + half * 8, 16), sizeof(b));
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        r[i]     = a[i];
        r[8 + i] = b[i];
    }
    return r;
}

// Optional cooperative TDM: each issuing wave owns disjoint rows of the tile.
// All issuing waves drain their own TENSORcnt before the workgroup barrier.
template <int Rows, int Waves, bool Distributed, int PadBytes = 16, int Width = 128, class T>
CK_TILE_DEVICE void load_tile(T* lds, const T* global, int remaining, int stride)
{
    if constexpr(Distributed)
    {
        static_assert(Rows % Waves == 0);
        constexpr int Part = Rows / Waves;
        const int offset   = (threadIdx.x / 32) * Part;
        load_strided<Part, 0, PadBytes, Width>(
            lds + offset * (Width + (Width == 192 ? 3 : 1) * PadBytes / 2),
            global + int64_t(offset) * stride,
            max(0, remaining - offset),
            stride);
    }
    else
    {
        load_strided<Rows, 0, PadBytes, Width>(lds, global, remaining, stride);
    }
}

// Swap the two 16-lane halves through VALU, without entering the LDS queue.
CK_TILE_DEVICE float swap16(float x)
{
    return __builtin_bit_cast(
        float,
        __builtin_amdgcn_permlanex16(
            0, __builtin_bit_cast(int, x), 0x76543210, 0xfedcba98, false, true));
}

template <bool Causal,
          bool Pad,
          bool FullQ,
          bool StoreLSE,
          bool FirstScore,
          int GridGroup,
          int RegPad,
          int StaticS,
          int LoopUnroll,
          bool PeelCausal,
          int QKPriority,
          int PVPriority,
          bool SetPVPriority,
          typename T         = __bf16,
          int DQ             = 128,
          int DV             = 128,
          int Tile           = 128,
          typename Args      = Arguments,
          int QueryTiles     = 0,
          int KeyTiles       = 0,
          int KeyTile        = Tile,
          bool TailFragments = false>
CK_TILE_DEVICE void run(Args a)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    constexpr bool Generic = !std::is_same_v<Args, Arguments>;
    constexpr int D = DQ, QM = Tile / 64, Waves = 4, BM = Tile, BN = KeyTile;
    constexpr int Pitch = DQ + (DQ == 192 ? 24 : 8), VP = DV + 16;
    constexpr int QKFragments = DQ / 16, VFragments = DV / 16;
    static_assert(LoopUnroll > 0);
    // The launcher checks five Q/K tiles, Q % 16 == 15 and K % 128 > 112.
    static_assert(!TailFragments || (!Generic && !Causal && Pad && !FullQ && BM == 128 &&
                                     BN == 128 && QueryTiles == 5));
    static_assert(QKPriority >= 0 && QKPriority <= 3 && PVPriority >= 0 && PVPriority <= 3);
    static_assert(SetPVPriority || PVPriority == 0);
    static_assert(!PeelCausal || (Causal && !Pad && BM == BN && StaticS > 0 && StaticS % BN == 0));
    // The host dispatch checks this fixed B32/H8 square shape before launch.
    // Runtime scale and tensor pointers are preserved.
    if constexpr(StaticS)
    {
        a.sq    = StaticS;
        a.sk    = StaticS;
        a.heads = 8;
        a.batch = 32;
    }
    if constexpr(QueryTiles)
    {
        static_assert(!StaticS && !Generic);
        if constexpr(KeyTiles)
        {
            static_assert(FullQ && !Pad);
            a.sq = QueryTiles * BM;
            a.sk = KeyTiles * BN;
        }
        a.heads = 8;
    }
    const unsigned grid_x = StaticS ? (StaticS + BM - 1) / BM : QueryTiles ? QueryTiles : gridDim.x;
    const unsigned grid_y = StaticS ? 256 : QueryTiles && KeyTiles ? a.batch * 8 : gridDim.y;

    // A rectangular query tile temporarily spans both K buffers. Q is held
    // in registers before either K buffer is filled; O reuses them after PV.
    __shared__ __align__(16) union
    {
        T blocks[2][BN * Pitch];
        T rows[BM * Pitch];
    } k_storage;
    auto& ks = k_storage.blocks;
    __shared__ __align__(16) T vs[2][BN * VP];
    const int heads = a.heads;
    auto tile_div   = [&](unsigned x) { return x / grid_x; };
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, tiles = grid_x;

    int linear = blockIdx.x + tiles * blockIdx.y, bh, tile;
    {
        if constexpr(Causal)
        {
            const int per_batch = tiles * heads, within = linear % per_batch;
            bh   = linear / per_batch * heads + within % heads;
            tile = tiles - 1 - within / heads;
        }
        else
        {
            const int total = tiles * grid_y, xcd = linear % 8;
            linear = xcd * (total / 8) + min(xcd, total % 8) + linear / 8;
            {
                bh   = linear / tiles;
                tile = linear % tiles;
            }
        }
    }
    if constexpr(GridGroup > 0)
    {
        // Whole batch/head rows per XCD, Q tiles traversed in bounded row groups.
        // This is a bijection when each XCD owns a multiple of GridGroup rows.
        if(grid_y % (8 * GridGroup) == 0)
        {
            const unsigned physical = blockIdx.x + tiles * blockIdx.y;
            const unsigned slot = physical / 8, depth = slot / GridGroup;
            const unsigned chunk = tile_div(depth);
            bh   = (physical % 8) * (grid_y / 8) + chunk * GridGroup + slot % GridGroup;
            tile = Causal ? tiles - 1 - (depth - chunk * tiles) : depth - chunk * tiles;
        }
    }
    bh              = __builtin_amdgcn_readfirstlane(bh);
    tile            = __builtin_amdgcn_readfirstlane(tile);
    const int row0  = tile * BM + wave * 16 + lane % 16;
    const int batch = bh / heads, head = bh % heads;
    int qstride = heads * D, kstride = qstride, vstride = qstride, row_stride = qstride;
    int64_t qbase = int64_t(batch) * a.sq * qstride + head * D;
    int64_t kbase = int64_t(batch) * a.sk * kstride + head * D;
    int64_t vbase = kbase, obase = qbase, lbase = int64_t(bh) * a.sq;
    if constexpr(Generic)
    {
        qstride                 = a.stride_q;
        kstride                 = a.stride_k;
        vstride                 = a.stride_v;
        row_stride              = a.stride_o;
        const int kv_head       = head / (heads / a.kv_heads);
        const int64_t qs        = a.group ? a.start_q[batch] : 0;
        const int64_t key_start = a.group ? a.start_k[batch] : 0;
        qbase                   = (a.group ? qs * qstride : batch * a.batch_q) + head * a.head_q;
        kbase = (a.group ? key_start * kstride : batch * a.batch_k) + kv_head * a.head_k;
        vbase = (a.group ? key_start * vstride : batch * a.batch_v) + kv_head * a.head_v;
        obase = (a.group ? qs * row_stride : batch * a.batch_o) + head * a.head_o;
        lbase = (a.group ? qs : batch * a.batch_lse) + head * a.head_lse;
        if(a.length_q)
            a.sq = a.length_q[batch];
        else if(a.cumulative_q)
            a.sq = a.cumulative_q[batch + 1] - a.cumulative_q[batch];
        else if(a.group)
            a.sq = a.start_q[batch + 1] - qs;
        if(a.length_k)
            a.sk = a.length_k[batch];
        else if(a.cumulative_k)
            a.sk = a.cumulative_k[batch + 1] - a.cumulative_k[batch];
        else if(a.group)
            a.sk = a.start_k[batch + 1] - key_start;
        if(tile * BM >= a.sq)
            return;
    }
    const auto* q = static_cast<const T*>(a.q) + qbase;
    const auto* k = static_cast<const T*>(a.k) + kbase;
    const auto* v = static_cast<const T*>(a.v) + vbase;
    auto* o       = static_cast<T*>(a.o) + obase;
    auto mask     = [&]() {
        if constexpr(Generic)
            return make_generic_attention_mask_from_lr_window<
                    SimplifiedGenericAttentionMask<Causal>>(
                a.left, a.right, a.sq, a.sk, a.mask_type == 1);
        else
            return SimplifiedGenericAttentionMask<Causal>{a.sq, a.sk};
    }();
    // The tail contract fixes the number of K tiles. Keep the actual K length
    // for TDM bounds and masking, and expose the fixed trip count to unrolling.
    int begin = 0, end = TailFragments ? QueryTiles * BN
                         : Causal      ? min(a.sk, (tile + 1) * BM)
                                       : a.sk;
    if constexpr(Generic)
    {
        if constexpr(Causal)
        {
            auto range = mask.GetTileRangeAlongX(tile * BM, number<BM>{}, number<BN>{});
            begin      = min(a.sk, max(0, range.at(number<0>{})));
            end        = min(a.sk, max(0, range.at(number<1>{})));
        }
        else
            end = a.sk;
        if(begin >= end)
        {
            if constexpr(DQ == 128 || DQ == 192)
            {
                for(int i = threadIdx.x; i < BM * (DV / 8); i += 128)
                {
                    const int row = tile * BM + i / (DV / 8);
                    if(row < a.sq)
                    {
                        V8<T> zero{};
                        __builtin_memcpy(__builtin_assume_aligned(o + int64_t(row) * row_stride +
                                                                      (i % (DV / 8)) * 8,
                                                                  16),
                                         &zero,
                                         sizeof(zero));
                    }
                }
            }
            else
            {
                for(int i = threadIdx.x; i < BM * DV; i += 128)
                {
                    const int row = tile * BM + i / DV;
                    if(row < a.sq)
                        o[int64_t(row) * row_stride + i % DV] = T(0);
                }
            }
            if constexpr(StoreLSE)
                for(int r = threadIdx.x; r < BM; r += 128)
                    if(tile * BM + r < a.sq)
                        a.lse[lbase + tile * BM + r] = -INFINITY;
            return;
        }
    }
    // Distribute the wider D192 descriptors across three waves. Each issuer
    // drains its own TENSORcnt below before any wave reads the shared tiles.
    if constexpr(BM > BN)
    {
        load_tile<BM, Waves, true, 16, DQ>(
            k_storage.rows, q + int64_t(tile * BM) * qstride, a.sq - tile * BM, qstride);
    }
    else if constexpr(Generic && DQ == 192)
    {
        if(wave == 0)
            load_tile<BN, Waves, false, 16, DQ>(
                ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride);
        if(wave == 1)
            load_tile<BN, Waves, false, 32, DV>(
                vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride);
        if(wave == 2)
            load_tile<BM, Waves, false, 16, DQ>(
                ks[1], q + int64_t(tile * BM) * qstride, a.sq - tile * BM, qstride);
    }
    else
    {
        // A single issuer avoids redundant descriptors on short dense tiles.
        constexpr bool Cooperative = !(Generic && DQ <= 128 && BM == 64 && !Causal);
        if(Cooperative || wave == 0)
        {
            load_tile<BN, Waves, Cooperative, 16, DQ>(
                ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride);
            load_tile<BN, Waves, Cooperative, 32, DV>(
                vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride);
            load_tile<BM, Waves, Cooperative, 16, DQ>(
                ks[1], q + int64_t(tile * BM) * qstride, a.sq - tile * BM, qstride);
        }
    }

    wait();

    __syncthreads();

    V16<T> qr[QM][DQ / 32];
    F8 out[QM][VFragments] = {};
    float maxima[QM], sums[QM] = {};
#pragma unroll
    for(int qm = 0; qm < QM; ++qm)
    {
        maxima[qm] = -INFINITY;
#pragma unroll
        for(int t = 0; t < DQ / 32; ++t)
        {
            qr[qm][t] = read_k32<DQ>(BM > BN ? k_storage.rows : ks[1],
                                     Pitch,
                                     wave * 16 + lane % 16 + qm * 16 * Waves,
                                     t * 32);
        }
    }

    if constexpr(BM > BN)
    {
        __syncthreads();
        load_tile<BN, Waves, true, 16, DQ>(
            ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride);
        load_tile<BN, Waves, true, 32, DV>(
            vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride);
        wait();
        __syncthreads();
    }

    const float scale2 = a.scale * 1.4426950408889634f;
    // Traverse all key columns before returning to an accumulator on long
    // fixed shapes, increasing the distance between dependent QK instructions.
    constexpr bool DenseRectangle = !Generic && !Causal && QueryTiles == 16 && KeyTiles == 8;
    constexpr bool DepthFirst = StaticS == 32768 || (StaticS == 16384 && !Causal) || DenseRectangle;
    static_assert(!DepthFirst || (DQ == 128 && BN == 128));
    // Reverse the four depth slices on the longest dense specialization. This
    // preserves the QK reduction while improving its measured operand schedule.
    constexpr bool ReverseDepth = StaticS == 32768 && !Causal;
    // Keep the expensive rescale path out of the common instruction sequence.
    // The expectation must stay at the branch site to preserve its code placement.
    constexpr bool HintRescale =
        (StaticS == 32768 && !Causal) || (!StoreLSE && !Generic && !Causal &&
                                          (DenseRectangle || (QueryTiles == 2 && KeyTiles == 2) ||
                                           (QueryTiles == 5 && KeyTiles == 5)));
    V16<T> kc[QKFragments];
    auto read_slice = [&](int buffer, int slice, int i) {
        return read_k32<DQ>(ks[buffer],
                            Pitch,
                            (DepthFirst ? i * 16 : slice * 32 + (i % 2) * 16) + lane % 16,
                            DepthFirst ? (ReverseDepth ? (slice ^ 3) : slice) * 32 : (i / 2) * 32);
    };
    constexpr bool HoistInitialK = StaticS == 32768 && !Causal;
    if constexpr(HoistInitialK)
    {
        // Initial K is already complete. Keep its read and the Q-buffer reuse
        // barrier outside the unrolled loop; later slices arrive during PV.
#pragma unroll
        for(int i = 0; i < QKFragments; ++i)
            kc[i] = read_slice(0, 0, i);
        __syncthreads();
    }
    int stage = 0;
    // The fixed-shape path can favor QK over PV to improve overlap between waves.
    // Softmax runs at priority zero; generic configurations keep both matrices at one.
    __builtin_amdgcn_s_setprio(QKPriority);
    auto iteration = [&](auto mask_last, auto last_iteration, int start)
        __attribute__((always_inline))
    {
        if constexpr(!HoistInitialK)
        {
            if(start == begin)
            {
                if(wave == 0)
                    wait();
                __syncthreads();
            }
        }
        if(wave == 0 && !decltype(last_iteration)::value && start + BN < end)
        {
            // Every prefetched tile is complete on this fixed dense shape.
            // A constant extent avoids rebuilding the TDM boundary fields.
            const int remaining = StaticS == 32768 && !Causal ? BN : a.sk - start - BN;
            load_tile<BN, Waves, false, 16, DQ>(
                ks[stage ^ 1], k + int64_t(start + BN) * kstride, remaining, kstride);
            load_tile<BN, Waves, false, 32, DV>(
                vs[stage ^ 1], v + int64_t(start + BN) * vstride, remaining, vstride);
        }

        V16<T> vr0[VFragments];
        // Keep a small register range live across QK to tune clang's allocation.
        // Empty asm emits no arithmetic; the reserved values never enter attention.
        // This is compiler-sensitive and must be checked with same-device timing.
        using Padding = float __attribute__((ext_vector_type(RegPad ? RegPad : 1)));
        Padding reg_padding;

#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr(RegPad)
            asm volatile("" : "=v"(reg_padding));
#endif
        F8 s[QM][BN / 16] = {};

        // Keep the next K slice live while the current slice executes.
#pragma unroll
        for(int i = 0; i < QKFragments; ++i)
        {
            if constexpr(!HoistInitialK)
            {
                if(start == begin)
                    kc[i] = read_slice(stage, 0, i);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int ns = 0; ns < BN / 32; ++ns)
        {

            V16<T> kn[QKFragments];
            if(ns + 1 < BN / 32)
            {
#pragma unroll
                for(int i = 0; i < QKFragments; ++i)
                    kn[i] = read_slice(stage, ns + 1, i);
            }
            else if constexpr(StaticS != 32768)
            {
#pragma unroll
                for(int d = 0; d < VFragments; ++d)
                    vr0[d] = read_v32<VP>(vs[stage], d * 16, 0);
            }

            auto accumulate = [&](int qm, int i) {
                const int column = DepthFirst ? i : ns * 2 + i % 2;
                const int depth  = DepthFirst ? (ReverseDepth ? (ns ^ 3) : ns) : i / 2;
                s[qm][column]    = mma(kc[i], qr[qm][depth], s[qm][column]);
            };
            if constexpr(DepthFirst && Causal)
            {
#pragma unroll
                for(int qm = 0; qm < QM; ++qm)
                {
#pragma unroll
                    for(int i = 0; i < QKFragments; ++i)
                        accumulate(qm, i);
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < QKFragments; ++i)
                {
#pragma unroll
                    for(int qm = 0; qm < QM; ++qm)
                        accumulate(qm, i);
                }
            }

            constexpr int chunk = 16;
#pragma unroll
            for(int j = 0; j < 16 / chunk; ++j)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, chunk, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, chunk / 2 * QM, 0);
            }

            __builtin_amdgcn_sched_barrier(0);

            if(ns + 1 < BN / 32)
            {
#pragma unroll
                for(int i = 0; i < QKFragments; ++i)
                    kc[i] = kn[i];
            }
        }

#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr(RegPad)
            asm volatile("" : : "v"(reg_padding));
#endif
        __builtin_amdgcn_s_setprio(0);
        V16<T> p[QM][BN / 32];
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const int row = row0 + qm * 16 * Waves;
            if constexpr(Generic && Causal)
            {
                if(mask.IsEdgeTile(tile * BM, start, number<BM>{}, number<BN>{}))
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                        {
                            const int col = start + n * 16 + lane / 16 * 8 + i;
                            if(mask.IsOutOfBound(row, col))
                                s[qm][n][i] = -INFINITY;
                        }
                }
            }
            else if constexpr((Causal && decltype(mask_last)::value) || Pad)
            {
                if((Pad && start + BN > a.sk) || (Causal && start + BN > tile * BM))
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
                    {
                        if constexpr(TailFragments)
                        {
                            // The first seven fragments are complete under the tail contract.
                            if(n + 1 < BN / 16)
                                continue;
                        }
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                        {
                            const int col = start + n * 16 + lane / 16 * 8 + i;
                            if((Pad && col >= a.sk) || (Causal && col > row))
                                s[qm][n][i] = -INFINITY;
                        }
                    }
                }
            }

            auto local_max = [&](const auto& logits) {
                float local[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
                for(int n = 0; n < BN / 16; ++n)
                {
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        local[n % 4] = fmaxf(local[n % 4], logits[n][i]);
                }
                const float lm = fmaxf(fmaxf(local[0], local[1]), fmaxf(local[2], local[3]));
                return fmaxf(lm, swap16(lm)) * scale2;
            };
            auto exponentiate = [&](const auto& logits, float base) {
                constexpr int Parts = StaticS == 32768 ? 4 : 2;
                F2 partial[Parts]   = {};
#pragma unroll
                for(int n = 0; n < BN / 16; ++n)
                {
#pragma unroll
                    for(int i = 0; i < 8; i += 2)
                    {
                        const F2 x = __builtin_elementwise_fma(
                            F2{logits[n][i], logits[n][i + 1]},
                            F2{scale2, scale2},
                            F2{-(Generic && base == -INFINITY ? 0.f : base),
                               -(Generic && base == -INFINITY ? 0.f : base)});
                        const float e0                    = __builtin_amdgcn_exp2f(x[0]);
                        const F2 e                        = {e0, __builtin_amdgcn_exp2f(x[1])};
                        p[qm][n / 2][(n % 2) * 8 + i]     = T(e[0]);
                        p[qm][n / 2][(n % 2) * 8 + i + 1] = T(e[1]);
                        if constexpr(StaticS == 32768)
                        {
                            // The first nonnegative exp initializes each part;
                            // adding it to zero otherwise survives codegen.
                            if(!Causal && n * 4 + i / 2 < Parts)
                                partial[(n * 4 + i / 2) % Parts] = e;
                            else
                                partial[(n * 4 + i / 2) % Parts] += e;
                        }
                        else
                        {
                            if(n < Parts && i == 0)
                                partial[n % Parts] = e;
                            else
                                partial[n % Parts] += e;
                        }
                    }
                }
                if constexpr(StaticS == 32768)
                {
                    // Independent accumulators shorten the softmax dependency
                    // chain; combine them before the same partial-sum guard.
                    partial[0] += partial[2];
                    partial[1] += partial[3];
                }
                F2 total = partial[0];
                total += partial[1];
                return total[0] + total[1];
            };
            float base = maxima[qm];
            if(start == begin || (Generic && base == -INFINITY))
            {
                if constexpr(FirstScore)
                {
                    // Column zero exists for every valid row, including top-left causal.
                    // Both half-waves choose its score and share the same initial base.
                    const float first       = s[qm][0][0];
                    const float other       = swap16(first);
                    const float valid_first = lane < 16 ? first : other;
                    // The known finite score keeps at least one weight at 2^-8.
                    // A larger score is handled by the guard before multiplying V.
                    base = valid_first * scale2 + 8.f;
                }
                else
                    base = local_max(s[qm]);
                maxima[qm] = base;
            }
            float sum     = exponentiate(s[qm], base);
            float row_sum = sum;

            // A nonnegative partial sum <= 2^8 proves each of its exp values <= 2^8.
            // Only the slow path needs the logit maximum. Overflow also enters it.
            if((FirstScore || start != begin) &&
               (HintRescale ? __builtin_expect(bool(__any(!(row_sum <= 256.f))), false)
                            : __any(!(row_sum <= 256.f))))
            {
                auto correct = [&](const auto& logits) {
                    const float candidate = fmaxf(base, local_max(logits));
                    const bool update     = candidate > base + 8.f;
                    if(HintRescale ? __builtin_expect(bool(__any(update)), false) : __any(update))
                    {
                        const float next  = update ? candidate : base;
                        const float alpha = __builtin_amdgcn_exp2f(base - next);
#pragma unroll
                        for(int d = 0; d < VFragments; ++d)
                            out[qm][d] *= alpha;
                        sums[qm] *= alpha;
                        maxima[qm] = next;
                        sum        = exponentiate(logits, next);
                        row_sum    = sum;
                    }
                };
                {
                    correct(s[qm]);
                }
            }
            sums[qm] += row_sum;
            if constexpr(StaticS == 32768 && !Causal)
            {
                // Let the second row fragment's softmax hide the first V read.
                // Delaying this read also shortens V's lifetime across QK.
                if(qm == 0)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                        vr0[d] = read_v32<VP>(vs[stage], d * 16, 0);
                }
            }
        }

        if constexpr(StaticS == 32768 && Causal)
        {
            // The peeled causal schedule benefits from keeping V out of softmax.
#pragma unroll
            for(int d = 0; d < VFragments; ++d)
                vr0[d] = read_v32<VP>(vs[stage], d * 16, 0);
        }
        if constexpr(SetPVPriority)
            __builtin_amdgcn_s_setprio(PVPriority);
        {
            V16<T> vc[VFragments];
#pragma unroll
            for(int d = 0; d < VFragments; ++d)
            {
                vc[d] = vr0[d];
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int t = 0; t < BN / 32; ++t)
            {
                V16<T> vn[VFragments];
                if(t + 1 < BN / 32)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                        vn[d] = read_v32<VP>(vs[stage], d * 16, (t + 1) * 32);
                }
                else if constexpr(!decltype(last_iteration)::value)
                {
                    // All reads of the old V tile finish before its buffer is reused.
                    // Next K is ready here; its first slice overlaps this final PV stage.
                    {
                        if(wave == 0)
                            wait();
                    }
                    __syncthreads();
#pragma unroll
                    for(int i = 0; i < QKFragments; ++i)
                        kc[i] = read_slice(stage ^ 1, 0, i);
                }

                // Keep each query fragment's PV instructions in one scheduling
                // region. The same P operand feeds all eight output fragments.
                // This is a compiler scheduling boundary, not a workgroup sync.
#pragma unroll
                for(int qm = 0; qm < QM; ++qm)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                        out[qm][d] = mma(vc[d], p[qm][t], out[qm][d]);
                    __builtin_amdgcn_sched_barrier(0);
                }

                {

                    constexpr int chunk = 16;
#pragma unroll
                    for(int j = 0; j < 16 / chunk; ++j)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, chunk, 1);
                        __builtin_amdgcn_sched_group_barrier(0x008, chunk / 2 * QM, 1);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                if(t + 1 < BN / 32)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                        vc[d] = vn[d];
                }
            }
        }

        if constexpr(QKPriority != PVPriority)
            __builtin_amdgcn_s_setprio(QKPriority);
    };
    if constexpr(!Generic && Pad && !Causal && BM == 128)
    {
        // A separate final iteration avoids prefetching an unused K slice.
        int start = begin;
#pragma unroll LoopUnroll
        for(; start + BN < end; start += BN, stage ^= 1)
            iteration(std::true_type{}, std::false_type{}, start);
        if(start < end)
            iteration(std::true_type{}, std::true_type{}, start);
    }
    else if constexpr(PeelCausal)
    {
        // For square, aligned M=N tiles, only the final K tile meets the diagonal.
        // Keep its mask outside the unrolled prefix to shorten register lifetimes.
        int start = begin;
#pragma unroll LoopUnroll
        for(; start + BN < end; start += BN, stage ^= 1)
            iteration(std::false_type{}, std::false_type{}, start);
        if(start < end)
            iteration(std::true_type{}, std::false_type{}, start);
    }
    else
    {
#pragma unroll LoopUnroll
        for(int start = begin; start < end; start += BN, stage ^= 1)
            iteration(std::true_type{}, std::false_type{}, start);
    }

    __builtin_amdgcn_s_setprio(0);

    if constexpr(Generic && DQ == 128 && BM == 64)
    {
        // The WMMA result already holds eight contiguous values per row/lane.
        // Short strided tiles benefit from writing these vectors without an LDS shuffle.
        wait();
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            sums[qm] += swap16(sums[qm]);
            const float inv = sums[qm] == 0.f ? 0.f : 1.f / sums[qm];
            const int row   = row0 + qm * 16 * Waves;
            if(row < a.sq)
            {
#pragma unroll
                for(int d = 0; d < VFragments; ++d)
                {
                    V8<T> result;
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        result[i] = T(out[qm][d][i] * inv);
                    __builtin_memcpy(__builtin_assume_aligned(o + int64_t(row) * row_stride +
                                                                  d * 16 + lane / 16 * 8,
                                                              16),
                                     &result,
                                     sizeof(result));
                }
            }
        }
    }
    else
    {
        static_assert(BM <= 2 * BN);
        constexpr int OPitch = Pitch;
        T* output_lds        = k_storage.rows;
        // Reuse the now-dead K0 storage after all waves finish their LDS accesses.
        wait();
        __syncthreads();
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            sums[qm] += swap16(sums[qm]);
            // These compact BF16 tiles have nonempty rows. Native reciprocal
            // avoids the general division sequence in their output epilogue.
            constexpr bool NativeReciprocal =
                TailFragments ||
                (!StoreLSE && !Causal && !Generic &&
                 ((QueryTiles == 4 && KeyTiles == 4) || (QueryTiles == 5 && KeyTiles == 5)));
            const float inv     = NativeReciprocal             ? ck_tile::rcp(sums[qm])
                                  : Generic && sums[qm] == 0.f ? 0.f
                                                               : 1.f / sums[qm];
            const int local_row = wave * 16 + lane % 16 + qm * 16 * Waves;
#pragma unroll
            for(int d = 0; d < VFragments; ++d)
            {
                V8<T> result{};
#pragma unroll
                for(int i = 0; i < 8; ++i)
                    result[i] = T(out[qm][d][i] * inv);
                __builtin_memcpy(__builtin_assume_aligned(
                                     output_lds + local_row * OPitch + d * 16 + lane / 16 * 8, 16),
                                 &result,
                                 sizeof(result));
            }
        }

        // The row redistribution stays within each wave's own 16-row fragments.
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_wait_dscnt 0" ::: "memory");

        constexpr bool Dense64 = Generic && DQ == 64 && BM == 64 && !Causal;
        // Async stores use a 32-bit byte offset; wider output strides retain
        // the 64-bit vector-store path below.
        constexpr uint64_t MaxAsyncStride = ((uint64_t{1} << 32) - DV * 2) / (2 * (BM - 1));
        const bool async_rows =
            (!Generic && BM == 128) || (Dense64 && uint64_t(row_stride) <= MaxAsyncStride);
        // A complete 16-row wave fragment can use the asynchronous burst
        // even when a different fragment intersects the query boundary.
        constexpr bool PartialAsync = !Generic && BM == 128 && !Causal;
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const bool full_fragment = tile * BM + wave * 16 + qm * 16 * Waves + 16 <= a.sq;
            const bool fifteen_rows =
                TailFragments && tile * BM + wave * 16 + qm * 16 * Waves + 15 == a.sq;
            const bool async_fragment = TailFragments
                                            ? (full_fragment || fifteen_rows)
                                            : (FullQ || (async_rows && tile * BM + BM <= a.sq) ||
                                               (PartialAsync && (full_fragment || fifteen_rows)));
            if(async_fragment)
            {
                uint32_t offset[VFragments], local[VFragments];
#pragma unroll
                for(int t = 0; t < VFragments; ++t)
                {
                    const int local_row =
                        wave * 16 + qm * 16 * Waves + (256 / DV) * t + lane / (DV / 8);
                    offset[t] = uint32_t(local_row * row_stride + (lane % (DV / 8)) * 8) * 2;
                    local[t]  = uint32_t(reinterpret_cast<uintptr_t>(
                        output_lds + local_row * OPitch + (lane % (DV / 8)) * 8));
                }
                if constexpr(VFragments == 8)
                {
                    if constexpr(TailFragments)
                    {
                        if(!full_fragment)
                            store_async_15(o + int64_t(tile * BM) * row_stride, offset, local);
                        else
                            store_async_x8(o + int64_t(tile * BM) * row_stride, offset, local);
                    }
                    else
                        store_async_x8(o + int64_t(tile * BM) * row_stride, offset, local);
                }
                else
                    store_async_x4(o + int64_t(tile * BM) * row_stride, offset, local);
            }
            else if constexpr(!TailFragments)
            {
                // With Q % 16 == 15, a fragment that is neither full nor 15 rows
                // has no valid rows, so the tail specialization needs no fallback.
                // Keep only one vector live to avoid private scratch/LDS materialization.
#pragma unroll
                for(int t = 0; t < VFragments; ++t)
                {
                    const int local_row =
                        wave * 16 + qm * 16 * Waves + (256 / DV) * t + lane / (DV / 8);
                    const int row = tile * BM + local_row;
                    if(row < a.sq)
                    {
                        V8<T> result;
                        __builtin_memcpy(&result,
                                         __builtin_assume_aligned(output_lds + local_row * OPitch +
                                                                      (lane % (DV / 8)) * 8,
                                                                  16),
                                         sizeof(result));
                        __builtin_memcpy(__builtin_assume_aligned(o + int64_t(row) * row_stride +
                                                                      (lane % (DV / 8)) * 8,
                                                                  16),
                                         &result,
                                         sizeof(result));
                    }
                }
            }
        }
        if constexpr(FullQ || (!Generic && BM == 128) || Dense64)
            __builtin_amdgcn_s_wait_asynccnt(0);
    }

    if constexpr(StoreLSE)
    {
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const int row = row0 + qm * 16 * Waves;
            if((FullQ || row < a.sq) && a.lse && lane < 16)
                a.lse[lbase + row] = maxima[qm] * 0.6931471805599453f + logf(sums[qm]);
        }
    }

#else
    (void)a;
#endif
}

} // namespace ck_tile::detail::fmha_cr

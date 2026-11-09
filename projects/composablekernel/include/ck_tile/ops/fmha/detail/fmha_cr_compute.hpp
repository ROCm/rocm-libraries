// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
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

using F8        = float __attribute__((ext_vector_type(8)));
using F2        = float __attribute__((ext_vector_type(2)));
constexpr int D = 128;

template <class T>
CK_TILE_DEVICE F8 mma(V16<T> a, V16<T> b, F8 c)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    static_assert(std::is_same_v<T, __bf16>);
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(0, a, 0, b, 0, c, 0, 0);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

template <class T>
CK_TILE_DEVICE V16<T> read_k32(const T* p, int pitch, int row, int k)
{
    const int half = (threadIdx.x % 32) / 16;
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
template <int Rows, int Waves, bool Distributed, int PadBytes = 16, class T>
CK_TILE_DEVICE void load_tile(T* lds, const T* global, int remaining, int stride)
{
    if constexpr(Distributed)
    {
        static_assert(Rows % Waves == 0);
        constexpr int Part = Rows / Waves;
        const int offset   = (threadIdx.x / 32) * Part;
        load_strided<Part, 0, PadBytes>(lds + offset * (128 + PadBytes / 2),
                                        global + int64_t(offset) * stride,
                                        max(0, remaining - offset),
                                        stride);
    }
    else
    {
        load_strided<Rows, 0, PadBytes>(lds, global, remaining, stride);
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
          bool SetPVPriority>
CK_TILE_DEVICE void run(Arguments a)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    using T          = __bf16;
    constexpr int QM = 2, Waves = 4, BM = 128, BN = 128, Pitch = 136, VP = 144;
    static_assert(LoopUnroll > 0);
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
    const unsigned grid_x = StaticS ? (StaticS + BM - 1) / BM : gridDim.x;
    const unsigned grid_y = StaticS ? 256 : gridDim.y;

    __shared__ __align__(16) T ks[2][BN * Pitch], vs[2][BN * VP];
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
    const int batch = bh / heads, head = bh % heads, row_stride = heads * D;
    const int64_t qbase = int64_t(batch) * a.sq * row_stride + head * D;
    const int64_t kbase = int64_t(batch) * a.sk * row_stride + head * D;
    const auto* q       = static_cast<const T*>(a.q) + qbase;
    const auto* k       = static_cast<const T*>(a.k) + kbase;
    const auto* v       = static_cast<const T*>(a.v) + kbase;
    auto* o             = static_cast<T*>(a.o) + qbase;
    const int end       = Causal ? min(a.sk, (tile + 1) * BM) : a.sk;

    load_tile<BN, Waves, true>(ks[0], k, a.sk, row_stride);
    load_tile<BN, Waves, true, (VP - 128) * 2>(vs[0], v, a.sk, row_stride);
    load_tile<BM, Waves, true>(
        ks[1], q + int64_t(tile * BM) * row_stride, a.sq - tile * BM, row_stride);

    wait();

    __syncthreads();

    V16<T> qr[QM][4];
    F8 out[QM][8] = {};
    float maxima[QM], sums[QM] = {};
#pragma unroll
    for(int qm = 0; qm < QM; ++qm)
    {
        maxima[qm] = -INFINITY;
#pragma unroll
        for(int t = 0; t < 4; ++t)
        {
            qr[qm][t] = read_k32(ks[1], Pitch, wave * 16 + lane % 16 + qm * 16 * Waves, t * 32);
        }
    }

    const float scale2 = a.scale * 1.4426950408889634f;
    V16<T> kc[8];
    int stage = 0;
    // The fixed-shape path can favor QK over PV to improve overlap between waves.
    // Softmax runs at priority zero; generic configurations keep both matrices at one.
    __builtin_amdgcn_s_setprio(QKPriority);
    auto iteration = [&](auto mask_last, int start) __attribute__((always_inline))
    {
        if(start == 0)
        {
            if(wave == 0)
                wait();
            __syncthreads();
        }
        if(wave == 0 && start + BN < end)
        {
            load_tile<BN, Waves, false>(
                ks[stage ^ 1], k + int64_t(start + BN) * row_stride, a.sk - start - BN, row_stride);
            load_tile<BN, Waves, false, (VP - 128) * 2>(
                vs[stage ^ 1], v + int64_t(start + BN) * row_stride, a.sk - start - BN, row_stride);
        }

        V16<T> vr0[8];
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

        // Keep the next N32 slice live while the current slice executes.
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            if(start == 0)
                kc[i] = read_k32(ks[stage], Pitch, (i % 2) * 16 + lane % 16, (i / 2) * 32);
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int ns = 0; ns < BN / 32; ++ns)
        {

            V16<T> kn[8];
            if(ns + 1 < BN / 32)
            {
#pragma unroll
                for(int i = 0; i < 8; ++i)
                    kn[i] = read_k32(
                        ks[stage], Pitch, (ns + 1) * 32 + (i % 2) * 16 + lane % 16, (i / 2) * 32);
            }
            else
            {
#pragma unroll
                for(int d = 0; d < 8; ++d)
                    vr0[d] = read_v32<VP>(vs[stage], d * 16, 0);
            }

            {
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
#pragma unroll
                    for(int qm = 0; qm < QM; ++qm)
                        s[qm][ns * 2 + i % 2] = mma(kc[i], qr[qm][i / 2], s[qm][ns * 2 + i % 2]);
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
                for(int i = 0; i < 8; ++i)
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
            if constexpr((Causal && decltype(mask_last)::value) || Pad)
            {
                if((Pad && start + BN > a.sk) || (Causal && start + BN > tile * BM))
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
                    {
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
                constexpr int Parts = 2;
                F2 partial[Parts]   = {};
#pragma unroll
                for(int n = 0; n < BN / 16; ++n)
                {
#pragma unroll
                    for(int i = 0; i < 8; i += 2)
                    {
                        const F2 x = __builtin_elementwise_fma(F2{logits[n][i], logits[n][i + 1]},
                                                               F2{scale2, scale2},
                                                               F2{-base, -base});
                        const float e0                    = __builtin_amdgcn_exp2f(x[0]);
                        const F2 e                        = {e0, __builtin_amdgcn_exp2f(x[1])};
                        p[qm][n / 2][(n % 2) * 8 + i]     = T(e[0]);
                        p[qm][n / 2][(n % 2) * 8 + i + 1] = T(e[1]);
                        if(n < Parts && i == 0)
                            partial[n % Parts] = e;
                        else
                            partial[n % Parts] += e;
                    }
                }
                F2 total = partial[0];
                total += partial[1];
                return total[0] + total[1];
            };
            float base = maxima[qm];
            if(start == 0)
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
            if((FirstScore || start != 0) && __any(!(row_sum <= 256.f)))
            {
                auto correct = [&](const auto& logits) {
                    const float candidate = fmaxf(base, local_max(logits));
                    const bool update     = candidate > base + 8.f;
                    if(__any(update))
                    {
                        const float next  = update ? candidate : base;
                        const float alpha = __builtin_amdgcn_exp2f(base - next);
#pragma unroll
                        for(int d = 0; d < 8; ++d)
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
        }

        if constexpr(SetPVPriority)
            __builtin_amdgcn_s_setprio(PVPriority);
        {
            V16<T> vc[8];
#pragma unroll
            for(int d = 0; d < 8; ++d)
            {
                vc[d] = vr0[d];
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int t = 0; t < BN / 32; ++t)
            {
                V16<T> vn[8];
                if(t + 1 < BN / 32)
                {
#pragma unroll
                    for(int d = 0; d < 8; ++d)
                        vn[d] = read_v32<VP>(vs[stage], d * 16, (t + 1) * 32);
                }
                else
                {
                    // All reads of the old V tile finish before its buffer is reused.
                    // Next K is ready here; its first slice overlaps this final PV stage.
                    {
                        if(wave == 0)
                            wait();
                    }
                    __syncthreads();
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        kc[i] =
                            read_k32(ks[stage ^ 1], Pitch, (i % 2) * 16 + lane % 16, (i / 2) * 32);
                }

                // Keep each query fragment's PV instructions in one scheduling
                // region. The same P operand feeds all eight output fragments.
                // This is a compiler scheduling boundary, not a workgroup sync.
#pragma unroll
                for(int qm = 0; qm < QM; ++qm)
                {
#pragma unroll
                    for(int d = 0; d < 8; ++d)
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
                    for(int d = 0; d < 8; ++d)
                        vc[d] = vn[d];
                }
            }
        }

        if constexpr(QKPriority != PVPriority)
            __builtin_amdgcn_s_setprio(QKPriority);
    };
    if constexpr(PeelCausal)
    {
        // For square, aligned M=N tiles, only the final K tile meets the diagonal.
        // Keep its mask outside the unrolled prefix to shorten register lifetimes.
        int start = 0;
#pragma unroll LoopUnroll
        for(; start + BN < end; start += BN, stage ^= 1)
            iteration(std::false_type{}, start);
        if(start < end)
            iteration(std::true_type{}, start);
    }
    else
    {
#pragma unroll LoopUnroll
        for(int start = 0; start < end; start += BN, stage ^= 1)
            iteration(std::true_type{}, start);
    }

    __builtin_amdgcn_s_setprio(0);

    static_assert(BM <= BN);
    constexpr int OPitch = Pitch;
    // Reuse the now-dead K0 storage after all waves finish their LDS accesses.
    wait();
    __syncthreads();
#pragma unroll
    for(int qm = 0; qm < QM; ++qm)
    {
        sums[qm] += swap16(sums[qm]);
        const float inv     = 1.f / sums[qm];
        const int local_row = wave * 16 + lane % 16 + qm * 16 * Waves;
#pragma unroll
        for(int d = 0; d < 8; ++d)
        {
            V8<T> result{};
#pragma unroll
            for(int i = 0; i < 8; ++i)
                result[i] = T(out[qm][d][i] * inv);
            __builtin_memcpy(
                __builtin_assume_aligned(ks[0] + local_row * OPitch + d * 16 + lane / 16 * 8, 16),
                &result,
                sizeof(result));
        }
    }

    // The row redistribution stays within each wave's own 16-row fragments.
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_wait_dscnt 0" ::: "memory");

    if constexpr(FullQ)
    {
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            uint32_t offset[8], local[8];
#pragma unroll
            for(int t = 0; t < 8; ++t)
            {
                const int local_row = wave * 16 + qm * 16 * Waves + 2 * t + lane / 16;
                offset[t]           = uint32_t(local_row * row_stride + (lane % 16) * 8) * 2;
                local[t]            = uint32_t(
                    reinterpret_cast<uintptr_t>(ks[0] + local_row * OPitch + (lane % 16) * 8));
            }
            store_async_x8(o + int64_t(tile * BM) * row_stride, offset, local);
        }
        __builtin_amdgcn_s_wait_asynccnt(0);
    }
    else
    {
        // Redistribute each row to 16 lanes, making each store cover two full rows.
        V8<T> result[QM][8];
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
#pragma unroll
            for(int t = 0; t < 8; ++t)
            {
                const int local_row = wave * 16 + qm * 16 * Waves + 2 * t + lane / 16;
                __builtin_memcpy(
                    &result[qm][t],
                    __builtin_assume_aligned(ks[0] + local_row * OPitch + (lane % 16) * 8, 16),
                    sizeof(result[qm][t]));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
#pragma unroll
            for(int t = 0; t < 8; ++t)
            {
                const int row = tile * BM + wave * 16 + qm * 16 * Waves + 2 * t + lane / 16;
                if(row < a.sq)
                    __builtin_memcpy(__builtin_assume_aligned(
                                         o + int64_t(row) * row_stride + (lane % 16) * 8, 16),
                                     &result[qm][t],
                                     sizeof(result[qm][t]));
            }
        }
    }

    if constexpr(StoreLSE)
    {
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const int row = row0 + qm * 16 * Waves;
            if((FullQ || row < a.sq) && a.lse && lane < 16)
                a.lse[int64_t(bh) * a.sq + row] = maxima[qm] * 0.6931471805599453f + logf(sums[qm]);
        }
    }

#else
    (void)a;
#endif
}

} // namespace ck_tile::detail::fmha_cr

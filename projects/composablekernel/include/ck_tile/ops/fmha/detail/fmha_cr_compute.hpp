// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/detail/fmha_cr_memory.hpp"

// Keep QK/PV operand lifetimes and scheduling regions together.
namespace ck_tile::detail::fmha_cr {

struct Arguments
{
    using RowStride                         = index_t;
    static constexpr bool kPrepared         = false;
    static constexpr auto BiasEnum          = BlockAttentionBiasEnum::NO_BIAS;
    static constexpr bool kHasLogitsSoftCap = false;
    static constexpr bool kHasSink          = false;
    static constexpr bool kVirtualSink      = false;
    const void *q, *k, *v;
    void* o;
    float* lse;
    int batch, heads, sq, sk;
    float scale;
};

// Head-local tensor views and logical lengths prepared by FmhaFwdKernel.
// Compact Arguments retain their launch ABI for the BSHD specializations.
struct StridedArguments : Arguments
{
    static constexpr bool kPrepared = true;
    int query_tile;
    int stride_q, stride_k, stride_v, stride_o;
    int mask_type, left, right;
};

template <typename Base>
struct WideStrideArguments : Base
{
    using RowStride = long_index_t;
};

// The general path scales scores before masking and uses scalar-aligned O
// stores. Keep it separate so the vector/TDM path retains its schedule and ABI.
struct GeneralArguments : StridedArguments
{
};

struct HeadPaddedArguments : GeneralArguments
{
    int hdim_q, hdim_v;
};

template <typename Base, BlockAttentionBiasEnum Bias, bool SoftCap>
struct ScoreArguments : Base
{
    static constexpr auto BiasEnum          = Bias;
    static constexpr bool kHasLogitsSoftCap = SoftCap;
    const void* bias;
    int64_t stride_bias;
    float logits_soft_cap;
};

// Prefix sink tokens are independent of a virtual sink logit.
// Keep their metadata out of the ordinary and compact argument layouts.
template <typename Base, bool Sink, bool VirtualSink>
struct SequenceArguments : Base
{
    static constexpr bool kHasSink     = Sink;
    static constexpr bool kVirtualSink = VirtualSink;
    int sink_size;
    const float* sink;
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
    k              = LdsLayout<Width, 16>::Column(k);
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
template <int Rows,
          int Waves,
          bool Distributed,
          int PadBytes = 16,
          int Width    = 128,
          class T,
          class Stride>
CK_TILE_DEVICE void
load_tile(T* lds, const T* global, int remaining, Stride stride, int valid_width = Width)
{
    if constexpr(Distributed)
    {
        static_assert(Rows % Waves == 0);
        constexpr int Part = Rows / Waves;
        const int offset   = (threadIdx.x / 32) * Part;
        load_strided<Part, 0, PadBytes, Width>(lds + offset * LdsLayout<Width, PadBytes>::kPitch,
                                               global + int64_t(offset) * stride,
                                               max(0, remaining - offset),
                                               stride,
                                               valid_width);
    }
    else
    {
        load_strided<Rows, 0, PadBytes, Width>(lds, global, remaining, stride, valid_width);
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
          typename T          = __bf16,
          int DQ              = 128,
          int DV              = 128,
          int Tile            = 128,
          typename Args       = Arguments,
          int QueryTiles      = 0,
          int KeyTiles        = 0,
          int KeyTile         = Tile,
          bool TailFragments  = false,
          bool PreferNonempty = false>
CK_TILE_DEVICE void run(Args a)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    constexpr bool Generic    = !std::is_same_v<Args, Arguments>;
    constexpr bool General    = std::is_base_of_v<GeneralArguments, Args>;
    constexpr bool HeadPadded = std::is_base_of_v<HeadPaddedArguments, Args>;
    constexpr bool CooperativePrefetch =
        Generic &&
        (HeadPadded || DQ == 32 || DQ == 96 || DQ == 160 ||
         (DQ == 64 && Tile == 64 && KeyTile == 128 && Args::kVirtualSink) ||
         (Tile == 128 && (Args::BiasEnum != BlockAttentionBiasEnum::NO_BIAS ||
                          Args::kHasLogitsSoftCap || Args::kVirtualSink))) &&
        !(DQ == 32 && KeyTile == 64 && !HeadPadded);
    constexpr int D = DQ, QM = Tile / 64, Waves = 4, BM = Tile, BN = KeyTile;
    // TDM completes in issue order: optional bias, next K, then next V.
    // Count 1 releases next K at PV's end; count 2 before reading V leaves only
    // the following K/V in flight (count 0 when there is no following tile).
    constexpr bool DeferredValueWait =
        CooperativePrefetch && (DQ != 32 || (HeadPadded && std::is_same_v<T, __bf16>));
    constexpr bool SoftmaxV = DeferredValueWait && BM == 128;
    // Wide causal ALiBi tiles rescale often enough to favor a maximum before
    // every exponentiation. A finite base also avoids per-lane reinitialization
    // while a row is fully masked; those tiles still contribute zero weight.
    constexpr bool EagerMax = DQ == 160 && BM == 128 && Causal && !HeadPadded &&
                              Args::BiasEnum == BlockAttentionBiasEnum::ALIBI &&
                              !Args::kHasLogitsSoftCap && !Args::kHasSink;
    // Dense virtual-sink rows initialize their base from the first K tile.
    constexpr bool FiniteBase =
        CooperativePrefetch && (BM == 128 || (DQ == 32 && BN == 128)) &&
        (std::is_same_v<T, __bf16> || EagerMax ||
         (DQ == 96 && Args::kVirtualSink && !Causal && !HeadPadded &&
          Args::BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !Args::kHasLogitsSoftCap));
    constexpr int Pitch = LdsLayout<DQ, 16>::kPitch, VP = LdsLayout<DV, 32>::kPitch;
    constexpr int QKFragments = DQ / 16, VFragments = DV / 16;
    constexpr bool EvenKeyPrefetch =
        HeadPadded && (DQ == 96 || DQ == 192) &&
        !(DQ == 96 && BM == 64 && Args::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
          !Args::kHasLogitsSoftCap && !Args::kHasSink && !Args::kVirtualSink);
    const int query_width = [&] {
        if constexpr(HeadPadded)
            return a.hdim_q;
        else
            return DQ;
    }();
    const int value_width = [&] {
        if constexpr(HeadPadded)
            return a.hdim_v;
        else
            return DV;
    }();
    static_assert(LoopUnroll > 0);
    static_assert(!Args::kPrepared || GridGroup == 0,
                  "Prepared tensor views use the shared kernel's grid mapping");
    // The launcher checks five Q/K tiles, Q % 16 == 15 and K % 128 > 112.
    static_assert(!TailFragments || (!Generic && !Causal && Pad && !FullQ && BM == 128 &&
                                     BN == 128 && QueryTiles == 5));
    static_assert(QKPriority >= 0 && QKPriority <= 3 && PVPriority >= 0 && PVPriority <= 3);
    static_assert(SetPVPriority || PVPriority == 0);
    static_assert(!PeelCausal ||
                  (!Generic && Causal && !Pad && BM == BN && (!StaticS || StaticS % BN == 0)));
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
    if constexpr(Args::kPrepared)
    {
        bh   = 0;
        tile = a.query_tile;
    }
    else
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
    typename Args::RowStride qstride = heads * D, kstride = qstride, vstride = qstride;
    index_t row_stride = heads * D;
    int64_t qbase      = int64_t(batch) * a.sq * qstride + head * D;
    int64_t kbase      = int64_t(batch) * a.sk * kstride + head * D;
    int64_t vbase = kbase, obase = qbase, lbase = int64_t(bh) * a.sq;
    if constexpr(Args::kPrepared)
    {
        qstride    = a.stride_q;
        kstride    = a.stride_k;
        vstride    = a.stride_v;
        row_stride = a.stride_o;
        qbase = kbase = vbase = obase = lbase = 0;
    }
    const auto* q = static_cast<const T*>(a.q) + qbase;
    const auto* k = static_cast<const T*>(a.k) + kbase;
    const auto* v = static_cast<const T*>(a.v) + vbase;
    auto* o       = static_cast<T*>(a.o) + obase;
    auto mask     = [&]() {
        if constexpr(Args::kHasSink)
            return make_generic_attention_mask_from_lr_window<
                    SimplifiedGenericAttentionMask<Causal>>(
                a.left, a.right, a.sink_size, a.sq, a.sk, a.mask_type == 1);
        else if constexpr(Generic)
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
    int sink_end = 0, window_begin = 0;
    if constexpr(Generic)
    {
        if constexpr(Args::kHasSink && Causal)
        {
            auto range   = mask.GetSinkTileRangeAlongX(tile * BM, number<BM>{}, number<BN>{});
            sink_end     = min(a.sk, max(0, range.at(number<0>{})));
            window_begin = min(a.sk, max(0, range.at(number<1>{})));
            begin        = sink_end ? 0 : window_begin;
            end          = max(sink_end, min(a.sk, max(0, range.at(number<2>{}))));
        }
        else if constexpr(Causal)
        {
            auto range = mask.GetTileRangeAlongX(tile * BM, number<BM>{}, number<BN>{});
            begin      = min(a.sk, max(0, range.at(number<0>{})));
            end        = min(a.sk, max(0, range.at(number<1>{})));
        }
        else
            end = a.sk;
        if(PreferNonempty ? __builtin_expect(begin >= end, false) : begin >= end)
        {
            const float cached_virtual_sink = [&] {
                if constexpr(PreferNonempty && Args::kVirtualSink)
                    return a.sink[head];
                else
                    return 0.f;
            }();
            const bool vector_output = [&] {
                if constexpr(!General && (DQ == 128 || DQ == 192))
                    return true;
                else if constexpr(Args::kVirtualSink && DQ == 192 && !HeadPadded &&
                                  Args::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                                  !Args::kHasLogitsSoftCap && !Args::kHasSink)
                    return (reinterpret_cast<uintptr_t>(o) % 16 == 0) && row_stride % 8 == 0;
                else
                    return false;
            }();
            if(vector_output)
            {
                for(int i = threadIdx.x; i < BM * (DV / 8); i += 128)
                {
                    const int row = tile * BM + i / (DV / 8);
                    if(row < a.sq)
                    {
                        V8<T> zero{};
                        if constexpr(Args::kVirtualSink)
                        {
                            const T value = T(
                                __builtin_isnan(PreferNonempty ? cached_virtual_sink : a.sink[head])
                                    ? NAN
                                    : 0.f);
                            zero = V8<T>{value, value, value, value, value, value, value, value};
                        }
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
                    if(row < a.sq && i % DV < value_width)
                    {
                        if constexpr(Args::kVirtualSink)
                            o[int64_t(row) * row_stride + i % DV] = T(
                                __builtin_isnan(PreferNonempty ? cached_virtual_sink : a.sink[head])
                                    ? NAN
                                    : 0.f);
                        else
                            o[int64_t(row) * row_stride + i % DV] = T(0);
                    }
                }
            }
            if constexpr(StoreLSE)
                for(int r = threadIdx.x; r < BM; r += 128)
                    if(tile * BM + r < a.sq)
                    {
                        if constexpr(Args::kVirtualSink)
                            a.lse[lbase + tile * BM + r] =
                                PreferNonempty ? cached_virtual_sink : a.sink[head];
                        else
                            a.lse[lbase + tile * BM + r] = -INFINITY;
                    }
            return;
        }
    }
    auto next_tile = [&](int start) {
        if constexpr(Args::kHasSink && Causal)
            if(start + BN == sink_end)
                return max(start + BN, window_begin);
        return start + BN;
    };
    // Distribute the wider D192 descriptors across three waves. Each issuer
    // drains its own TENSORcnt below before any wave reads the shared tiles.
    if constexpr(BM > BN)
    {
        load_tile<BM, Waves, true, 16, DQ>(k_storage.rows,
                                           q + int64_t(tile * BM) * qstride,
                                           a.sq - tile * BM,
                                           qstride,
                                           query_width);
    }
    else if constexpr(Generic && DQ == 192)
    {
        if(wave == 0)
            load_tile<BN, Waves, false, 16, DQ>(
                ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride, query_width);
        if(wave == 1)
            load_tile<BN, Waves, false, 32, DV>(
                vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride, value_width);
        if(wave == 2)
            load_tile<BM, Waves, false, 16, DQ>(
                ks[1], q + int64_t(tile * BM) * qstride, a.sq - tile * BM, qstride, query_width);
    }
    else
    {
        // A single issuer avoids redundant descriptors on short dense tiles.
        constexpr bool Cooperative = !(Generic && DQ <= 128 && BM == 64 && !Causal);
        if(Cooperative || wave == 0)
        {
            load_tile<BN, Waves, Cooperative, 16, DQ>(
                ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride, query_width);
            load_tile<BN, Waves, Cooperative, 32, DV>(
                vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride, value_width);
            load_tile<BM, Waves, Cooperative, 16, DQ>(
                ks[1], q + int64_t(tile * BM) * qstride, a.sq - tile * BM, qstride, query_width);
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
            ks[0], k + int64_t(begin) * kstride, a.sk - begin, kstride, query_width);
        load_tile<BN, Waves, true, 32, DV>(
            vs[0], v + int64_t(begin) * vstride, a.sk - begin, vstride, value_width);
        wait();
        __syncthreads();
    }

    constexpr bool FoldScaleIntoExp = DQ == 32 && BM == 128 && BN == 64 && Args::kVirtualSink &&
                                      Args::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                                      !Args::kHasLogitsSoftCap;
    const bool scale_before_mask = General && (!FoldScaleIntoExp || a.scale <= 0.f);
    const float scale2           = scale_before_mask ? 1.f : a.scale * 1.4426950408889634f;
    const auto* bias             = [&]() {
        if constexpr(Args::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            return static_cast<const T*>(a.bias);
        else
            return static_cast<const T*>(nullptr);
    }();
    auto position_encoding = [&]() {
        if constexpr(Args::BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            const float slope = *static_cast<const float*>(a.bias) * ck_tile::log2e_v<float>;
            if constexpr(Causal)
                return make_alibi_from_lr_mask<float, true, 32>(
                    slope,
                    a.left,
                    a.right,
                    a.sq,
                    a.sk,
                    static_cast<GenericAttentionMaskEnum>(a.mask_type));
            else
                return Alibi<float, true, 32>{slope, a.sq, a.sk, AlibiMode::FROM_BOTTOM_RIGHT};
        }
        else
            return EmptyPositionEncoding<float>{};
    }();
    // Elementwise bias may mask just the first key. Without a finite base,
    // initialize from the row maximum so later tiles use the same normalization.
    constexpr bool InitialScore =
        FirstScore || (std::is_same_v<T, __bf16> && Generic &&
                       (HeadPadded || DQ == 32 || DQ == 96 || DQ == 160) &&
                       (Args::BiasEnum != BlockAttentionBiasEnum::ELEMENTWISE_BIAS || FiniteBase));
    const bool use_initial_score = [&] {
        if constexpr(Generic)
            return begin == 0 && (!Causal || a.left < 0);
        else
            return true;
    }();
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
    // Native vector assignments preserve D32 padded bias's LDS/WMMA schedule.
    // Even identity bit_casts change LLVM's vector decomposition on this path.
    constexpr bool PaddedBias32 = DQ == 32 && BM == 128 && BN == 128 && HeadPadded &&
                                  Args::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                                  !Args::kHasLogitsSoftCap && !Args::kHasSink &&
                                  !Args::kVirtualSink;
    // Padded FP16 causal tiles also favor native vectors for operand reuse.
    constexpr bool PaddedPlain32 =
        DQ == 32 && BM == 128 && BN == 128 && HeadPadded && Causal && std::is_same_v<T, _Float16> &&
        Args::BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !Args::kHasLogitsSoftCap &&
        !Args::kHasSink && !Args::kVirtualSink;
    constexpr bool NativeOperands = PaddedBias32 || PaddedPlain32;
    using Operand = std::conditional_t<(DQ == 32 || DQ == 64) && !NativeOperands, I8, V16<T>>;
    // Keep casts at their use sites to preserve LLVM's loop operand ordering.
    Operand kc[QKFragments];
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
        {
            if constexpr(NativeOperands)
                kc[i] = read_slice(0, 0, i);
            else
                kc[i] = bit_cast<Operand>(read_slice(0, 0, i));
        }
        __syncthreads();
    }
    constexpr bool BiasTdm = Args::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
    using BiasLayout       = LdsLayout<BN, 16>;
    static_assert(BiasLayout::kChunk == BN);
    auto* bias_lds = [] {
        if constexpr(BiasTdm)
        {
            __shared__ __align__(16) T storage[BM * BiasLayout::kPitch];
            return storage;
        }
        else
            return static_cast<T*>(nullptr);
    }();
    const bool bias_tdm = [&] {
        if constexpr(BiasTdm)
            return a.stride_bias <= INT32_MAX;
        else
            return false;
    }();
    int stage = 0;
    // The fixed-shape path can favor QK over PV to improve overlap between waves.
    // Softmax runs at priority zero; generic configurations keep both matrices at one.
    __builtin_amdgcn_s_setprio(QKPriority);
    auto iteration_body = [&](auto mask_last, auto last_iteration, int start, auto first_iteration)
        __attribute__((always_inline))
    {
        // Tensor waits ignore EXEC; keep the next-tile choice uniform so
        // the short sink path does not execute both wait thresholds.
        // Cache it only for cooperative waits; the single-issuer path evaluates
        // its predicate inside wave 0 to avoid extending VGPR lifetimes.
        // D32 padded bias recomputes it to keep the LDS buffer selector scalar.
        const bool has_next = [&] {
            if constexpr(!CooperativePrefetch || PaddedBias32)
                return false;
            else if constexpr(DQ == 96 && Args::kHasSink && Causal && BM == 64 && BN == 64)
                return bool(__builtin_amdgcn_readfirstlane(int(next_tile(start) < end)));
            else
                return next_tile(start) < end;
        }();
        if constexpr(BiasTdm)
        {
            // One shared bias tile; finish all readers before reusing it.
            __syncthreads();
            if(bias_tdm && amd_wave_read_first_lane(wave) == 1)
                load_tile<BM, Waves, false, 16, BN>(bias_lds,
                                                    bias + int64_t(tile * BM) * a.stride_bias +
                                                        start,
                                                    a.sq - tile * BM,
                                                    a.stride_bias,
                                                    min(BN, a.sk - start));
        }
        if constexpr(!HoistInitialK)
        {
            if((CooperativePrefetch ? decltype(first_iteration)::value : start == begin))
            {
                if(wave == 0)
                    wait();
                __syncthreads();
            }
        }
        if((CooperativePrefetch ? wave < 4 : wave == 0) && !decltype(last_iteration)::value &&
           (CooperativePrefetch && !PaddedBias32 ? has_next : next_tile(start) < end))
        {
            // Every prefetched tile is complete on this fixed dense shape.
            // A constant extent avoids rebuilding the TDM boundary fields.
            const int remaining = StaticS == 32768 && !Causal ? BN
                                  : Args::kHasSink && Causal  ? a.sk - next_tile(start)
                                                              : a.sk - start - BN;
            // Another logical row guarantees storage for the additional element,
            // including the final head. Broadcast rows keep their exact extent.
            const int prefetch_width = query_width + (EvenKeyPrefetch && (query_width & 1) &&
                                                      kstride > 0 && remaining > BN);
            load_tile<BN, Waves, CooperativePrefetch, 16, DQ>(ks[stage ^ 1],
                                                              k + int64_t(next_tile(start)) *
                                                                      kstride,
                                                              remaining,
                                                              kstride,
                                                              prefetch_width);
            load_tile<BN, Waves, CooperativePrefetch, 32, DV>(vs[stage ^ 1],
                                                              v + int64_t(next_tile(start)) *
                                                                      vstride,
                                                              remaining,
                                                              vstride,
                                                              value_width);
        }

        Operand vr0[VFragments];
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
                if((CooperativePrefetch ? decltype(first_iteration)::value : start == begin))
                {
                    if constexpr(NativeOperands)
                        kc[i] = read_slice(stage, 0, i);
                    else
                        kc[i] = bit_cast<Operand>(read_slice(stage, 0, i));
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int ns = 0; ns < BN / 32; ++ns)
        {

            Operand kn[QKFragments];
            if(ns + 1 < BN / 32)
            {
#pragma unroll
                for(int i = 0; i < QKFragments; ++i)
                {
                    if constexpr(NativeOperands)
                        kn[i] = read_slice(stage, ns + 1, i);
                    else
                        kn[i] = bit_cast<Operand>(read_slice(stage, ns + 1, i));
                }
            }
            else if constexpr(StaticS != 32768 && !SoftmaxV)
            {
                if constexpr(DeferredValueWait)
                {
                    if(PaddedBias32 ? next_tile(start) < end : has_next)
                        __builtin_amdgcn_s_wait_tensorcnt(2);
                    else
                        __builtin_amdgcn_s_wait_tensorcnt(0);
                    __syncthreads();
                }
#pragma unroll
                for(int d = 0; d < VFragments; ++d)
                {
                    if constexpr(NativeOperands)
                        vr0[d] = read_v32<VP, DV>(vs[stage], d * 16, 0);
                    else
                        vr0[d] = bit_cast<Operand>(read_v32<VP, DV>(vs[stage], d * 16, 0));
                }
            }

            auto accumulate = [&](int qm, int i) {
                const int column = DepthFirst ? i : ns * 2 + i % 2;
                const int depth  = DepthFirst ? (ReverseDepth ? (ns ^ 3) : ns) : i / 2;
                {
                    if constexpr(NativeOperands)
                        s[qm][column] = mma(kc[i], qr[qm][depth], s[qm][column]);
                    else
                        s[qm][column] = mma(bit_cast<V16<T>>(kc[i]), qr[qm][depth], s[qm][column]);
                }
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
        if constexpr(BiasTdm)
        {
            if(bias_tdm && __builtin_amdgcn_readfirstlane(wave) == 1)
            {
                if constexpr(CooperativePrefetch)
                {
                    if(PaddedBias32 ? next_tile(start) < end : has_next)
                        __builtin_amdgcn_s_wait_tensorcnt(2);
                    else
                        wait();
                }
                else
                    wait();
            }
            __syncthreads();
        }
        V16<T> p[QM][BN / 32];
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const int row = row0 + qm * 16 * Waves;
            // A nonpositive scale cannot be applied to the -infinity mask
            // sentinel. Scale finite QK scores first on the general path.
            if constexpr(Args::kHasLogitsSoftCap)
            {
                const LogitsSoftCapParams<decltype(mask), true> params{
                    mask, a.scale * ck_tile::log2e_v<float>, a.logits_soft_cap};
                const LogitsSoftCap<true> transform;
#pragma unroll
                for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        s[qm][n][i] = transform.LogitsTransform(params, s[qm][n][i], 0, 0, 0);
            }
            else if constexpr(General)
            {
                if(scale_before_mask)
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                            s[qm][n][i] *= a.scale * 1.4426950408889634f;
                }
            }
            if constexpr(BiasTdm)
            {
                if(bias_tdm)
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
                    {
                        V8<T> fragment;
                        const auto* address = bias_lds + (row - tile * BM) * BiasLayout::kPitch +
                                              BiasLayout::Column(n * 16 + lane / 16 * 8);
                        __builtin_memcpy(
                            &fragment, __builtin_assume_aligned(address, 16), sizeof(fragment));
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                            s[qm][n][i] += float(fragment[i]) * ck_tile::log2e_v<float>;
                    }
                }
                else
                {
#pragma unroll
                    for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                        for(int i = 0; i < 8; ++i)
                        {
                            const int col = start + n * 16 + lane / 16 * 8 + i;
                            if(row < a.sq && col < a.sk)
                                s[qm][n][i] += float(bias[int64_t(row) * a.stride_bias + col]) *
                                               ck_tile::log2e_v<float>;
                        }
                }
            }
            else if constexpr(Args::BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
#pragma unroll
                for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                    {
                        const int col = start + n * 16 + lane / 16 * 8 + i;
                        float score   = s[qm][n][i];
                        position_encoding.update(score, row, col);
                        s[qm][n][i] = score;
                    }
            }
            if constexpr(Generic && Causal)
            {
                if constexpr(decltype(mask_last)::value)
                    if(mask.IsEdgeTile(tile * BM, start, number<BM>{}, number<BN>{}))
                    {
#pragma unroll
                        for(int n = 0; n < BN / 16; ++n)
#pragma unroll
                            for(int i = 0; i < 8; ++i)
                            {
                                const int col     = start + n * 16 + lane / 16 * 8 + i;
                                const bool masked = Args::kHasSink ? mask.IsOutOfSinkBound(row, col)
                                                                   : mask.IsOutOfBound(row, col);
                                if(masked)
                                    s[qm][n][i] = -INFINITY;
                            }
                    }
            }
            else if constexpr((Causal && decltype(mask_last)::value) || Pad)
            {
                if((Pad && start + BN > a.sk) || (Causal && start + BN > tile * BM))
                {
                    // Combine padding and causal bounds once for this score row.
                    const int last_key = Causal ? (Pad ? min(a.sk - 1, row) : row) : a.sk - 1;
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
                            if(col > last_key)
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
                            F2{-(Generic && !FiniteBase && base == -INFINITY ? 0.f : base),
                               -(Generic && !FiniteBase && base == -INFINITY ? 0.f : base)});
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
            const bool initialized =
                (CooperativePrefetch ? decltype(first_iteration)::value : start == begin) ||
                (Generic && !FiniteBase && base == -INFINITY);
            if((CooperativePrefetch ? decltype(first_iteration)::value : start == begin) ||
               (Generic && !FiniteBase && base == -INFINITY))
            {
                if constexpr(InitialScore && !EagerMax)
                {
                    if(use_initial_score)
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
                }
                else
                    base = local_max(s[qm]);
                if constexpr(FiniteBase)
                    base = fmaxf(base, -__FLT_MAX__);
                maxima[qm] = base;
            }
            if constexpr(Args::BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                if(!initialized && (EagerMax || (position_encoding.mode == AlibiMode::VERTICAL &&
                                                 position_encoding.slope > 0.f)))
                {
                    const float estimate = local_max(s[qm]);
                    const bool update    = estimate > base;
                    if(__any(update))
                    {
                        const float next_base = update ? estimate : base;
                        const float alpha = update ? __builtin_amdgcn_exp2f(base - next_base) : 1.f;
#pragma unroll
                        for(int d = 0; d < VFragments; ++d)
                            out[qm][d] *= alpha;
                        sums[qm] *= alpha;
                        maxima[qm] = base = next_base;
                    }
                }
            }
            float sum     = exponentiate(s[qm], base);
            float row_sum = sum;

            // A nonnegative partial sum <= 2^8 proves each of its exp values <= 2^8.
            // Only the slow path needs the logit maximum. Overflow also enters it.
            if constexpr(!EagerMax)
            {
                if(((InitialScore && use_initial_score) ||
                    (CooperativePrefetch ? !decltype(first_iteration)::value : start != begin)) &&
                   (HintRescale ? __builtin_expect(bool(__any(!(row_sum <= 256.f))), false)
                                : __any(!(row_sum <= 256.f))))
                {
                    auto correct = [&](const auto& logits) {
                        const float candidate = fmaxf(base, local_max(logits));
                        const bool update     = candidate > base + 8.f;
                        if(HintRescale ? __builtin_expect(bool(__any(update)), false)
                                       : __any(update))
                        {
                            const float next = update ? candidate : base;
                            // Another row can enter this wave-wide branch while this
                            // row is fully masked. Preserve it instead of evaluating -inf - -inf.
                            const float alpha =
                                Generic && !update ? 1.f : __builtin_amdgcn_exp2f(base - next);
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
            }
            // Consume the reduction before PV so its partial sums can die here.
            if constexpr(EagerMax)
                asm volatile("" : "+v"(row_sum));
            sums[qm] += row_sum;
            if constexpr(SoftmaxV)
            {
                // Reuse the first query fragment's dead score registers for V.
                if(qm == 0)
                {
                    if(PaddedBias32 ? next_tile(start) < end : has_next)
                        __builtin_amdgcn_s_wait_tensorcnt(2);
                    else
                        __builtin_amdgcn_s_wait_tensorcnt(0);
                    __syncthreads();
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                    {
                        if constexpr(NativeOperands)
                            vr0[d] = read_v32<VP, DV>(vs[stage], d * 16, 0);
                        else
                            vr0[d] = bit_cast<Operand>(read_v32<VP, DV>(vs[stage], d * 16, 0));
                    }
                }
            }
            if constexpr(StaticS == 32768 && !Causal)
            {
                // Let the second row fragment's softmax hide the first V read.
                // Delaying this read also shortens V's lifetime across QK.
                if(qm == 0)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                    {
                        if constexpr(NativeOperands)
                            vr0[d] = read_v32<VP, DV>(vs[stage], d * 16, 0);
                        else
                            vr0[d] = bit_cast<Operand>(read_v32<VP, DV>(vs[stage], d * 16, 0));
                    }
                }
            }
        }

        if constexpr(StaticS == 32768 && Causal)
        {
            // The peeled causal schedule benefits from keeping V out of softmax.
#pragma unroll
            for(int d = 0; d < VFragments; ++d)
            {
                if constexpr(NativeOperands)
                    vr0[d] = read_v32<VP, DV>(vs[stage], d * 16, 0);
                else
                    vr0[d] = bit_cast<Operand>(read_v32<VP, DV>(vs[stage], d * 16, 0));
            }
        }
        if constexpr(SetPVPriority)
            __builtin_amdgcn_s_setprio(PVPriority);
        {
            Operand vc[VFragments];
#pragma unroll
            for(int d = 0; d < VFragments; ++d)
            {
                vc[d] = vr0[d];
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int t = 0; t < BN / 32; ++t)
            {
                Operand vn[VFragments];
                if(t + 1 < BN / 32)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                    {
                        if constexpr(NativeOperands)
                            vn[d] = read_v32<VP, DV>(vs[stage], d * 16, (t + 1) * 32);
                        else
                            vn[d] = bit_cast<Operand>(
                                read_v32<VP, DV>(vs[stage], d * 16, (t + 1) * 32));
                    }
                }
                else if constexpr(!decltype(last_iteration)::value)
                {
                    // All reads of the old V tile finish before its buffer is reused.
                    // Next K is ready here; its first slice overlaps this final PV stage.
                    {
                        if constexpr(DeferredValueWait)
                            __builtin_amdgcn_s_wait_tensorcnt(1);
                        else if(CooperativePrefetch || wave == 0)
                            wait();
                    }
                    if constexpr(EvenKeyPrefetch)
                    {
                        // Each issuing wave repairs its own rows after its next-K
                        // TDM completes. The existing workgroup barrier publishes
                        // the zero before any wave consumes that column.
                        static_assert(CooperativePrefetch);
                        if((query_width & 1) && has_next)
                        {
                            constexpr int Part = BN / Waves;
                            for(int r = lane; r < Part; r += 32)
                                ks[stage ^ 1][(wave * Part + r) * Pitch +
                                              LdsLayout<DQ, 16>::Column(query_width)] = T(0);
                            ck_tile::s_wait_dscnt<0>();
                        }
                    }
                    __syncthreads();
#pragma unroll
                    for(int i = 0; i < QKFragments; ++i)
                    {
                        if constexpr(NativeOperands)
                            kc[i] = read_slice(stage ^ 1, 0, i);
                        else
                            kc[i] = bit_cast<Operand>(read_slice(stage ^ 1, 0, i));
                    }
                }

                // Keep each query fragment's PV instructions in one scheduling
                // region. The same P operand feeds all eight output fragments.
                // This is a compiler scheduling boundary, not a workgroup sync.
#pragma unroll
                for(int qm = 0; qm < QM; ++qm)
                {
#pragma unroll
                    for(int d = 0; d < VFragments; ++d)
                    {
                        if constexpr(NativeOperands)
                            out[qm][d] = mma(vc[d], p[qm][t], out[qm][d]);
                        else
                            out[qm][d] = mma(bit_cast<V16<T>>(vc[d]), p[qm][t], out[qm][d]);
                    }
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
    auto iteration = [&](auto mask_last, auto last_iteration, int start) {
        iteration_body(mask_last, last_iteration, start, std::false_type{});
    };
    if constexpr(CooperativePrefetch)
    {
        int start = begin;
        iteration_body(std::true_type{}, std::false_type{}, start, std::true_type{});
        start = next_tile(start);
        stage ^= 1;
        if constexpr(Causal && (BM == 128 || !std::is_same_v<T, __bf16> ||
                                (DQ == 64 && BM == 64 && BN == 128 && Args::kVirtualSink)))
        {
            // With no left window, the prefix is fully visible to every valid Q row.
            const int full_end =
                a.left < 0 && a.right == 0
                    ? min(a.sk, tile * BM + (a.mask_type == 1 ? 0 : a.sk - a.sq) + 1)
                    : begin;
#pragma unroll LoopUnroll
            for(; start + BN <= full_end; start += BN, stage ^= 1)
                iteration_body(std::false_type{}, std::false_type{}, start, std::false_type{});
        }
#pragma unroll LoopUnroll
        for(; start < end; start = next_tile(start), stage ^= 1)
            iteration_body(std::true_type{}, std::false_type{}, start, std::false_type{});
    }
    else if constexpr(!Generic && Pad && !Causal && BM == 128)
    {
        // A separate final iteration avoids prefetching an unused K slice.
        int start = begin;
#pragma unroll LoopUnroll
        for(; next_tile(start) < end; start += BN, stage ^= 1)
            iteration(std::true_type{}, std::false_type{}, start);
        if(start < end)
            iteration(std::true_type{}, std::true_type{}, start);
    }
    else if constexpr(PeelCausal)
    {
        // For aligned M=N block tiles, only the final K tile can meet the diagonal.
        // Keep its mask outside the unrolled prefix to shorten register lifetimes.
        int start = begin;
#pragma unroll LoopUnroll
        for(; next_tile(start) < end; start += BN, stage ^= 1)
            iteration(std::false_type{}, std::false_type{}, start);
        if(start < end)
            iteration(std::true_type{}, std::false_type{}, start);
    }
    else
    {
#pragma unroll LoopUnroll
        for(int start = begin; start < end; start = next_tile(start), stage ^= 1)
            iteration(std::true_type{}, std::false_type{}, start);
    }

    __builtin_amdgcn_s_setprio(0);

    if constexpr(Args::kVirtualSink)
    {
        // Merge the zero-valued virtual token after the attention loop. Each
        // lane holds half of the denominator until the epilogue reduction.
        const float sink = a.sink[head] * ck_tile::log2e_v<float>;
        if(sink != -INFINITY)
        {
#pragma unroll
            for(int qm = 0; qm < QM; ++qm)
            {
                const float base   = maxima[qm];
                const float next   = __builtin_isnan(sink) ? sink : max(base, sink);
                const float alpha  = next == INFINITY || base == -INFINITY
                                         ? 0.f
                                         : __builtin_amdgcn_exp2f(base - next);
                const float weight = sink == INFINITY ? 1.f : __builtin_amdgcn_exp2f(sink - next);
                sums[qm]           = sums[qm] * alpha + 0.5f * weight;
                maxima[qm]         = next;
#pragma unroll
                for(int d = 0; d < VFragments; ++d)
                    out[qm][d] *= alpha;
            }
        }
    }
    constexpr bool DirectOutput = General || (Generic && DQ == 128 && BM == 64);
    if constexpr(DirectOutput)
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
                    auto* destination = o + int64_t(row) * row_stride + d * 16 + lane / 16 * 8;
                    if constexpr(HeadPadded)
                    {
                        const int column = d * 16 + lane / 16 * 8;
                        if(column + 8 <= value_width)
                            __builtin_memcpy(destination, &result, sizeof(result));
                        else
                            for(int i = 0; i < 8; ++i)
                                if(column + i < value_width)
                                    destination[i] = result[i];
                    }
                    else if constexpr(General)
                        __builtin_memcpy(destination, &result, sizeof(result));
                    else
                        __builtin_memcpy(
                            __builtin_assume_aligned(destination, 16), &result, sizeof(result));
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
            // Compact rows are nonempty. Correct the reciprocal with one Newton
            // step without the general division sequence's exceptional-value fixup.
            auto refined_reciprocal = [](float sum) {
                float r = ck_tile::rcp(sum);
                r       = __builtin_fmaf(r, __builtin_fmaf(-sum, r, 1.f), r);
                return r;
            };
            const float inv     = NativeReciprocal             ? ck_tile::rcp(sums[qm])
                                  : !Generic && !StoreLSE      ? refined_reciprocal(sums[qm])
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

        // Each wave stores only its own 16-row fragments, including a partial tail.
#pragma unroll
        for(int qm = 0; qm < QM; ++qm)
        {
            const int local_row = get_warp_id() * 16 + qm * 16 * Waves;
            const int row       = tile * BM + local_row;
            if(FullQ || row < a.sq)
                store_strided<16, DV, OPitch>(output_lds + local_row * OPitch,
                                              o + int64_t(row) * row_stride,
                                              FullQ ? 16 : a.sq - row,
                                              uint32_t(row_stride));
        }
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

    // LSE writes do not access output LDS and can overlap the pending TDM stores.
    if constexpr(!DirectOutput)
        wait();
#else
    (void)a;
#endif
}

} // namespace ck_tile::detail::fmha_cr

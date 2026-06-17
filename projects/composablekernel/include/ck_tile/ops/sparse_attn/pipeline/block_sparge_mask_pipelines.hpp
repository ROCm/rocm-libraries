// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/topk.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"

#include <cassert>

namespace ck_tile {

// Block reduction: intra-warp shuffle, cross-warp via scratch[0..numWarps); result in scratch[0].
template <index_t BlockSize, typename T, typename Op>
CK_TILE_DEVICE T
block_reduce_smem(T val, T* scratch, index_t tid, Op op, T identity)
{
    for(index_t off = get_warp_size() / 2; off > 0; off /= 2)
        val = op(val, warp_shuffle_down(val, static_cast<unsigned>(off)));
    if(get_lane_id() == 0)
        scratch[get_warp_id()] = val;
    block_sync_lds();

    constexpr index_t num_warps = (BlockSize + get_warp_size() - 1) / get_warp_size();
    static_assert((num_warps & (num_warps - 1)) == 0,
                  "cross-warp reduction assumes num_warps is a power of two");
    if(tid < static_cast<index_t>(get_warp_size()))
    {
        T v = (tid < num_warps) ? scratch[tid] : identity;
        for(index_t off = num_warps / 2; off > 0; off /= 2)
            v = op(v, warp_shuffle_down(v, static_cast<unsigned>(off)));
        if(tid == 0)
            scratch[0] = v;
    }
    block_sync_lds();
    return scratch[0];
}

template <index_t BlockSize>
CK_TILE_DEVICE float
block_reduce_max_f32(float val, float* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](float a, float b) { return a > b ? a : b; },
        -INFINITY);
}
template <index_t BlockSize>
CK_TILE_DEVICE float
block_reduce_sum_f32(float val, float* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](float a, float b) { return a + b; },
        0.0f);
}
template <index_t BlockSize>
CK_TILE_DEVICE int32_t
block_reduce_min_i32(int32_t val, int32_t* scratch, index_t tid)
{
    return block_reduce_smem<BlockSize>(
        val, scratch, tid,
        [](int32_t a, int32_t b) { return a < b ? a : b; },
        INT32_MAX);
}

// In-place descending bitonic sort with int32 payload; N power of two.
// Explicit LDS (not tile-distribution): the stage compare-exchanges i with (i ^ j) where j is a
// runtime loop var, a runtime cross-thread access ck_tile distributed tensors can't express (they
// only allow compile-time-indexed access to a thread's own registers, and have no tile-level sort).
// kStride = block thread count. N <= kStride: one-thread-per-element, inner stages (j < warp_size)
// stay within a wavefront so wave_barrier suffices. N > kStride (strided): a thread may own both i
// and (i ^ j) and warp neighbors aren't consecutive, so wave_barrier is unsafe -> CTA-wide
// block_sync_lds between every stage.
template <index_t N, index_t kStride>
CK_TILE_DEVICE void
bitonic_sort_desc_smem(float* keys, int32_t* vals, index_t tid)
{
    static_assert((N & (N - 1)) == 0 && N > 0, "N must be power of two");
    static_assert((kStride & (kStride - 1)) == 0 && kStride > 0,
                  "kStride must be power of two");

    constexpr bool strided = (N > kStride);
    const index_t wsz      = static_cast<index_t>(get_warp_size());

    for(index_t k = 2; k <= N; k <<= 1)
    {
        for(index_t j = k >> 1; j > 0; j >>= 1)
        {
            for(index_t i = tid; i < N; i += kStride)
            {
                const index_t ixj = i ^ j;
                if(ixj < N && ixj > i)
                {
                    const bool flip = ((i & k) == 0)
                                          ? (keys[i] < keys[ixj])
                                          : (keys[i] > keys[ixj]);
                    if(flip)
                    {
                        float kt   = keys[i]; keys[i] = keys[ixj]; keys[ixj] = kt;
                        int32_t vt = vals[i]; vals[i] = vals[ixj]; vals[ixj] = vt;
                    }
                }
            }
            if(strided || j >= wsz)
                block_sync_lds();
            else
                __builtin_amdgcn_wave_barrier();
        }
    }
}

// Hillis-Steele inclusive prefix sum on first N elements; aux is scratch of size >= N.
// Explicit LDS for the same reason as bitonic_sort_desc_smem: thread i reads (i - d) with d a
// runtime loop var, a cross-thread read tile-distribution can't express (no tile-level scan).
// kStride = block thread count; threads stride over [0, N) so N may exceed kStride.
template <index_t N, index_t kStride>
CK_TILE_DEVICE void
block_scan_inclusive_sum_smem(float* buf, float* aux, index_t tid)
{
    for(index_t e = tid; e < N; e += kStride) aux[e] = buf[e];
    block_sync_lds();

    for(index_t d = 1; d < N; d <<= 1)
    {
        // Read aux, write buf, fence: a strided thread never reads a slot updated this iteration.
        for(index_t e = tid; e < N; e += kStride)
        {
            float add  = (e >= d) ? aux[e - d] : 0.0f;
            buf[e]     = aux[e] + add;
        }
        block_sync_lds();
        for(index_t e = tid; e < N; e += kStride) aux[e] = buf[e];
        block_sync_lds();
    }
    for(index_t e = tid; e < N; e += kStride) buf[e] = aux[e];
    block_sync_lds();
}

// Quant absmax divisor: int8 -> 127, fp8 -> fp8_t max. fp8 uses a literal because
// numeric<fp8_t>::max() routes through bit_cast (not constexpr / __host__ here). Must match the
// host reference numeric<fp8_t>::max(): OCP E4M3 = 448, FNUZ E4M3 = 240, so branch on the macro.
template <typename QuantType>
CK_TILE_HOST_DEVICE constexpr float sparge_quant_absmax_divisor()
{
    if constexpr(std::is_same_v<QuantType, fp8_t>)
#if defined(CK_TILE_USE_OCP_FP8)
        return 448.0f;
#else
        return 240.0f;
#endif
    else
        return 127.0f;
}

// Per-block mean + optional cosine similarity (+ optional row-wise quant). Tile-programmed:
// reduces along the token axis with BlockReduce2d; matches the CPU reference bit-for-bit.
// Requires hdim == block_size == 128 (runtime-asserted).
template <typename InputType_,
          index_t kBlockSize_                            = 256,
          BlockSageAttentionQuantScaleEnum QScale        = BlockSageAttentionQuantScaleEnum::NO_SCALE,
          typename QuantType_                            = int8_t>
struct BlockSpargePreprocessPipeline
{
    using InputType = remove_cvref_t<InputType_>;
    using QuantType = remove_cvref_t<QuantType_>;
    static constexpr float kQuantDivisor = sparge_quant_absmax_divisor<QuantType>();

    static constexpr bool kDoQuant =
        (QScale != BlockSageAttentionQuantScaleEnum::NO_SCALE);

    static constexpr index_t kBlockSize   = kBlockSize_;
    static constexpr index_t kHdim        = 128;
    static constexpr index_t kBlock       = 128;
    static constexpr float   kNormEpsilon = 1e-8f; // sqrt argument floor

    using ComputeDataType = float;

    // Tile [M = hdim, N = token], reduce along N. M must stay within a warp (WarpPerBlock_N = 1):
    // a cross-warp N reduce would broadcast each warp's lane-0 partial over every M position and
    // collapse the result to mean[d mod period]. So warps go on M, N reduces wholly in one warp.
    struct BlockShape
    {
        static constexpr index_t Block_M = kHdim;
        static constexpr index_t Block_N = kBlock;

        static constexpr index_t WarpPerBlock_M = 4;
        static constexpr index_t WarpPerBlock_N = 1;

        static constexpr index_t ThreadPerWarp_M = 16;
        static constexpr index_t ThreadPerWarp_N = 4;

        static constexpr index_t Vector_M = 1;
        static constexpr index_t Vector_N = 2;

        static constexpr index_t Repeat_M =
            Block_M / (WarpPerBlock_M * ThreadPerWarp_M * Vector_M);
        static constexpr index_t Repeat_N =
            Block_N / (WarpPerBlock_N * ThreadPerWarp_N * Vector_N);

        static constexpr index_t BlockSize = kBlockSize;
    };

    using ReduceProblem =
        BlockReduce2dProblem<ComputeDataType, ComputeDataType, BlockShape>;

    static constexpr bool kNeedCrossWarpSync = (BlockShape::WarpPerBlock_N > 1);

    // X-tile distribution [M, N], reducing N. Mirrors the rmsnorm2d default policy encoding.
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    struct Params
    {
        index_t seqlen;
        index_t hdim;
        index_t block_size;
        index_t block_id;
        index_t stride_seq;     // tokens stride in elements; hdim for BHSD, H*hdim for BSHD
        float simthreshold;
        const float* km_ptr;    // [hdim] K-mean; nullptr disables (Q always nullptr)

        // Quantization (QScale != NO_SCALE); unused on the NO_SCALE path.
        QuantType* quant_out;      // [block_size, hdim] quant out (token-major, hdim stride 1)
        float*  scale_out;         // [block_size / tokens_per_scale] scales for this block
        index_t tokens_per_scale;  // PERWARP: Q=32, K=64 (kBlockScaleSize)
        index_t quant_stride_seq;  // token stride of quant_out in elements
    };

    // Stage the [token, hidden] block in LDS ONCE (coalesced non-transposed load); mean/quant/sim
    // all re-read it from LDS rather than re-loading global. 128*128 bf16 = 32KB, under the 64KB
    // budget alongside the ~1.5KB (km|inv_norm|reduce|absmax) scratch.
    static constexpr index_t kStageBytes =
        static_cast<index_t>(kBlock * kHdim * sizeof(InputType));

    // LDS: stage[kBlock*kHdim InputType] | km[hdim] | per-token inv-norm[block_size] |
    //      cross-warp reduce scratch | (quant) per-token absmax[block_size].
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(index_t /*hdim*/)
    {
        using x_block_tile =
            decltype(make_static_distributed_tensor<ComputeDataType>(
                MakeXBlockTileDistribution()));
        using y_block_tile =
            decltype(BlockReduce2d<ReduceProblem>::template MakeYBlockTile<x_block_tile>());
        constexpr index_t reduce_bytes =
            BlockReduce2dCrossWarpSync<ReduceProblem>::template GetSmemSize<y_block_tile>();
        // Quant additionally needs a per-token absmax[block] scratch.
        constexpr index_t quant_bytes =
            kDoQuant ? static_cast<index_t>(kBlock * sizeof(float)) : 0;
        return kStageBytes +
               static_cast<index_t>((kHdim + kBlock) * sizeof(float)) + reduce_bytes +
               quant_bytes;
    }

    // Non-transposed LDS view of the staged block: [token (M), hidden (N)], stride [kHdim, 1].
    CK_TILE_DEVICE static auto MakeStageViewTN(InputType* p_stage)
    {
        return make_tensor_view<address_space_enum::lds>(
            p_stage,
            make_naive_tensor_descriptor(
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                make_tuple(number<kHdim>{}, number<1>{})));
    }

    // Transposed LDS view of the staged block: [hidden (M), token (N)] over the same storage,
    // so the mean/sim reduce-along-token tile can read it.
    CK_TILE_DEVICE static auto MakeStageViewTransposed(InputType* p_stage)
    {
        const auto tn = MakeStageViewTN(p_stage);
        return transform_tensor_view(
            tn,
            make_tuple(make_pass_through_transform(number<kHdim>{}),
                       make_pass_through_transform(number<kBlock>{})),
            make_tuple(sequence<1>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // Per-warp row-wise quant: per-token absmax over hidden, grouped over tokens_per_scale tokens
    // -> scale = absmax/divisor, then quant = round(val / scale). smooth_k (K side, km_ptr != null):
    // the centered value (val - km[channel]) feeds BOTH absmax and quant, matching official
    // SpargeAttn; km is staged in s_km (indexed by hidden channel). Q side passes km_ptr == nullptr.
    // s_absmax is a [kBlock] LDS scratch (one slot per token-within-block).
    CK_TILE_DEVICE void quantize_block(InputType*        p_stage,
                                       const Params&     params,
                                       index_t           count,
                                       index_t           s_start,
                                       float*            s_absmax,
                                       const float*      s_km,
                                       void*             s_reduce) const
    {
        const index_t tid     = get_thread_id();
        const index_t tps     = params.tokens_per_scale;
        const bool    has_km  = (params.km_ptr != nullptr);

        const auto stage_tn = MakeStageViewTN(p_stage);
        auto q_window = make_tile_window(
            stage_tn,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            {0, 0},
            MakeXBlockTileDistribution());

        auto reduce       = BlockReduce2d<ReduceProblem>{};
        auto reduce_sync  = BlockReduce2dSync<ReduceProblem>{};
        auto reduce_xwarp = BlockReduce2dCrossWarpSync<ReduceProblem>{};
        auto absmax_func  = ReduceOp::AbsMax{};

        auto q_tile = load_tile(q_window);

        // per-token absmax over hidden (N); smooth_k centers by km[c] first. OOB tokens reduce to 0.
        auto abs_tile = make_static_distributed_tensor<ComputeDataType>(
            decltype(q_tile)::get_tile_distribution());
        sweep_tile(q_tile, [&](auto idx) {
            float v = type_convert<ComputeDataType>(q_tile[idx]);
            if(has_km)
            {
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    q_tile.get_tile_distribution(), idx);
                const index_t c = tile_idx.at(number<1>{});
                v -= s_km[c];
            }
            abs_tile(idx) = v;
        });
        auto amax_tile =
            reduce(abs_tile, absmax_func.GetIdentityValue<ComputeDataType>(), absmax_func);
        reduce_sync(amax_tile, absmax_func);
        if constexpr(kNeedCrossWarpSync)
            reduce_xwarp(amax_tile, s_reduce, absmax_func);

        // stage per-token absmax in LDS.
        for(index_t t = tid; t < kBlock; t += kBlockSize)
            s_absmax[t] = 0.0f;
        block_sync_lds();
        sweep_tile_span(decltype(amax_tile)::get_distributed_spans()[number<0>{}], [&](auto idx0) {
            constexpr auto t_idx = make_tuple(idx0);
            const auto tile_idx  = get_x_indices_from_distributed_indices(
                amax_tile.get_tile_distribution(), t_idx);
            const index_t t = tile_idx.at(number<0>{});
            if(t < count)
                s_absmax[t] = amax_tile[t_idx];
        });
        block_sync_lds();

        // group absmax over tokens_per_scale consecutive tokens -> scale = absmax/divisor.
        const index_t num_scale = kBlock / tps;
        for(index_t g = tid; g < num_scale; g += kBlockSize)
        {
            float a = 0.0f;
            const index_t g0 = g * tps;
            for(index_t t = 0; t < tps; ++t)
                a = max(a, s_absmax[g0 + t]);
            params.scale_out[g] = a / kQuantDivisor;
        }
        block_sync_lds();

        // sweep tile -> quant = round(val / scale[token-group]). OOB tokens skipped.
        const auto quant_view = make_naive_tensor_view<address_space_enum::global>(
            params.quant_out,
            make_tuple(params.seqlen, kHdim),
            make_tuple(params.quant_stride_seq, 1),
            number<1>{},
            number<1>{});
        const auto quant_padded = pad_tensor_view(
            quant_view,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            sequence<1, 0>{});
        auto quant_window = make_tile_window(
            quant_padded,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            {s_start, 0},
            MakeXBlockTileDistribution());

        auto out_tile = make_static_distributed_tensor<QuantType>(
            decltype(q_tile)::get_tile_distribution());
        sweep_tile(q_tile, [&](auto idx) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                q_tile.get_tile_distribution(), idx);
            const index_t t = tile_idx.at(number<0>{}); // token-within-block (M)
            QuantType q8    = QuantType{0};
            if(t < count)
            {
                const index_t c = tile_idx.at(number<1>{}); // hidden channel (N)
                const float sc  = params.scale_out[t / tps];
                float v         = type_convert<ComputeDataType>(q_tile[idx]);
                if(has_km)
                    v -= s_km[c];
                const float r   = (sc > 0.0f) ? (v / sc) : 0.0f;
                if constexpr(std::is_same_v<QuantType, fp8_t>)
                    q8 = type_convert<fp8_t>(r);
                else
                    q8 = type_convert<int8_t>(saturates<int8_t>{}(r));
            }
            out_tile(idx) = q8;
        });
        store_tile(quant_window, out_tile);
    }

    CK_TILE_DEVICE void operator()(
        const InputType* slice,
        float* mean_out,
        float* sim_out,
        const Params& params,
        void* smem) const
    {
        const index_t tid  = get_thread_id();
        const index_t hdim = params.hdim;

        assert(params.hdim == kHdim && params.block_size == kBlock &&
               "sparge preprocess tile path requires hdim == block_size == 128");

        InputType* s_stage = reinterpret_cast<InputType*>(smem);       // [kBlock*kHdim]
        float* s_km        = reinterpret_cast<float*>(
            reinterpret_cast<char*>(smem) + kStageBytes);              // [kHdim]
        float* s_inv_norm  = s_km + kHdim;                             // [kBlock]
        void*  s_reduce    = reinterpret_cast<void*>(s_inv_norm + kBlock);

        const index_t s_start = params.block_id * params.block_size;
        const index_t s_end   = min(s_start + params.block_size, params.seqlen);
        const index_t count   = s_end - s_start;

        if(count <= 0)
        {
            for(index_t d = tid; d < hdim; d += kBlockSize)
                mean_out[d] = 0.0f;
            if(sim_out != nullptr && tid == 0)
                *sim_out = 1.0f;
            return;
        }

        // Stage the [token, hidden] block from global into LDS once. OOB tokens read 0 via the
        // padded view (zero sum/absmax/Gram contributions); all later passes re-read this buffer.
        {
            const auto naive_in = make_naive_tensor_view<address_space_enum::global>(
                slice,
                make_tuple(params.seqlen, kHdim),
                make_tuple(params.stride_seq, 1),
                number<1>{},
                number<1>{});
            const auto padded_in = pad_tensor_view(
                naive_in,
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                sequence<1, 0>{}); // pad token axis only; hidden is exact 128
            auto in_window = make_tile_window(
                padded_in,
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                {s_start, 0},
                MakeXBlockTileDistribution());

            auto stage_view = MakeStageViewTN(s_stage);
            auto stage_window = make_tile_window(
                stage_view,
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                {0, 0},
                MakeXBlockTileDistribution());

            auto in_tile = load_tile(in_window);
            store_tile(stage_window, in_tile);
            block_sync_lds();
        }

        // Stage WG-uniform km_ptr in LDS so the tile sweeps index it by channel without per-element
        // global reads. Staged before quant so smooth_k centering can read it; Q side -> 0.
        const bool has_km = (params.km_ptr != nullptr);
        for(index_t d = tid; d < kHdim; d += kBlockSize)
            s_km[d] = has_km ? params.km_ptr[d] : 0.0f;
        block_sync_lds();

        // Quant sub-pass; its absmax scratch sits past the cross-warp reduce region.
        if constexpr(kDoQuant)
        {
            using x_block_tile =
                decltype(make_static_distributed_tensor<ComputeDataType>(
                    MakeXBlockTileDistribution()));
            using y_block_tile =
                decltype(BlockReduce2d<ReduceProblem>::template MakeYBlockTile<x_block_tile>());
            constexpr index_t reduce_floats =
                BlockReduce2dCrossWarpSync<ReduceProblem>::template GetSmemSize<y_block_tile>() /
                static_cast<index_t>(sizeof(float));
            float* s_absmax =
                reinterpret_cast<float*>(s_reduce) + reduce_floats;
            quantize_block(s_stage, params, count, s_start, s_absmax, s_km, s_reduce);
            block_sync_lds();
        }

        // Transposed view of the staged block: tile axes [hdim (M), token (N)] for the reduces.
        const auto transposed = MakeStageViewTransposed(s_stage);
        auto x_window = make_tile_window(
            transposed,
            make_tuple(number<kHdim>{}, number<kBlock>{}),
            {0, 0},
            MakeXBlockTileDistribution());

        auto reduce        = BlockReduce2d<ReduceProblem>{};
        auto reduce_sync   = BlockReduce2dSync<ReduceProblem>{};
        auto reduce_xwarp  = BlockReduce2dCrossWarpSync<ReduceProblem>{};
        auto add_func      = ReduceOp::Add{};

        // Pass 1: block mean = (1/count) * sum_token (t - km); reduce-sum along token (N).
        auto x_tile = load_tile(x_window);
        auto centered = make_static_distributed_tensor<ComputeDataType>(
            decltype(x_tile)::get_tile_distribution());
        sweep_tile(x_tile, [&](auto idx) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                x_tile.get_tile_distribution(), idx);
            const index_t m = tile_idx.at(number<0>{}); // hidden channel (M)
            centered(idx) = type_convert<ComputeDataType>(x_tile[idx]) - s_km[m];
        });

        auto mean_tile = reduce(centered, add_func.GetIdentityValue<ComputeDataType>(), add_func);
        reduce_sync(mean_tile, add_func);
        if constexpr(kNeedCrossWarpSync)
            reduce_xwarp(mean_tile, s_reduce, add_func);

        const float inv_count = 1.0f / static_cast<float>(count);
        // Write per-channel mean; mean_tile is replicated across the threads that reduced each m,
        // so duplicate stores carry identical values.
        constexpr auto mean_spans = decltype(mean_tile)::get_distributed_spans();
        sweep_tile_span(mean_spans[number<0>{}], [&](auto idx0) {
            constexpr auto m_idx = make_tuple(idx0);
            const auto tile_idx  = get_x_indices_from_distributed_indices(
                mean_tile.get_tile_distribution(), m_idx);
            const index_t m = tile_idx.at(number<0>{});
            mean_out[m]     = mean_tile[m_idx] * inv_count;
        });

        if(!(sim_out != nullptr && params.simthreshold > 0.0f))
        {
            if(sim_out != nullptr && tid == 0)
                *sim_out = 1.0f;
            return;
        }

        // Pass 2: block self-similarity = (1/count^2) sum_{i,j} cos(t_i, t_j), t = token - km.
        // Identity sum_{i,j}<t_i/|t_i|, t_j/|t_j|> = ||sum_i t_i/|t_i|||^2: accumulate the unit-vector
        // sum u[d] and report ||u||^2 / count^2 (matches upstream SpargeAttn).

        // Step 1: per-token inv-norm 1/|t_s|. BlockReduce2d only reduces N, so for a per-token (not
        // per-channel) squared sum read the [token(M), hidden(N)] non-transposed tile from LDS.
        const auto stage_tn = MakeStageViewTN(s_stage);
        auto xt_window = make_tile_window(
            stage_tn,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            {0, 0},
            MakeXBlockTileDistribution());

        auto xt_tile = load_tile(xt_window);
        // squared centered, reduce-sum hidden (N) -> per-token norm^2
        auto sq_tile = make_static_distributed_tensor<ComputeDataType>(
            decltype(xt_tile)::get_tile_distribution());
        sweep_tile(xt_tile, [&](auto idx) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                xt_tile.get_tile_distribution(), idx);
            const index_t d = tile_idx.at(number<1>{});
            const float v   = type_convert<ComputeDataType>(xt_tile[idx]) - s_km[d];
            sq_tile(idx)    = v * v;
        });

        auto norm_tile = reduce(sq_tile, add_func.GetIdentityValue<ComputeDataType>(), add_func);
        reduce_sync(norm_tile, add_func);
        if constexpr(kNeedCrossWarpSync)
            reduce_xwarp(norm_tile, s_reduce, add_func);

        // Stage per-token inv-norm into LDS.
        constexpr auto norm_spans = decltype(norm_tile)::get_distributed_spans();
        sweep_tile_span(norm_spans[number<0>{}], [&](auto idx0) {
            constexpr auto t_idx = make_tuple(idx0);
            const auto tile_idx  = get_x_indices_from_distributed_indices(
                norm_tile.get_tile_distribution(), t_idx);
            const index_t t = tile_idx.at(number<0>{});
            if(t < count)
                s_inv_norm[t] = 1.0f / ck_tile::sqrt(norm_tile[t_idx] + kNormEpsilon);
        });
        block_sync_lds();

        // Step 2: u[d] = sum_token normalize(t)[d]; multiply each centered element by its token's
        // inv-norm and reduce-sum along token (N).
        auto unit_tile = make_static_distributed_tensor<ComputeDataType>(
            decltype(x_tile)::get_tile_distribution());
        sweep_tile(x_tile, [&](auto idx) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                x_tile.get_tile_distribution(), idx);
            const index_t m = tile_idx.at(number<0>{}); // hidden channel (M)
            const index_t n = tile_idx.at(number<1>{}); // token (N)
            const float v   = type_convert<ComputeDataType>(x_tile[idx]) - s_km[m];
            const float inv = (n < count) ? s_inv_norm[n] : 0.0f;
            unit_tile(idx)  = v * inv;
        });

        auto u_tile = reduce(unit_tile, add_func.GetIdentityValue<ComputeDataType>(), add_func);
        reduce_sync(u_tile, add_func);
        if constexpr(kNeedCrossWarpSync)
            reduce_xwarp(u_tile, s_reduce, add_func);

        // sim = ||u||^2 / count^2. Stash each channel's u into LDS by overwriting s_km; the barrier
        // below makes this reuse of the km slots safe (the X-tile distribution already gives each warp
        // a disjoint set of M channels for the read above and the write below, so this is defensive),
        // then sum u[m]^2 over the hidden axis in one strided pass.
        block_sync_lds();
        sweep_tile_span(decltype(u_tile)::get_distributed_spans()[number<0>{}], [&](auto idx0) {
            constexpr auto m_idx = make_tuple(idx0);
            const auto tile_idx  = get_x_indices_from_distributed_indices(
                u_tile.get_tile_distribution(), m_idx);
            const index_t m = tile_idx.at(number<0>{});
            s_km[m]         = u_tile[m_idx];
        });
        block_sync_lds();

        float local_u_sq = 0.0f;
        for(index_t m = tid; m < kHdim; m += kBlockSize)
            local_u_sq += s_km[m] * s_km[m];

        float u_norm_sq =
            block_reduce_sum_f32<kBlockSize>(local_u_sq, reinterpret_cast<float*>(s_reduce), tid);
        if(tid == 0)
            *sim_out = u_norm_sq / (static_cast<float>(count) * static_cast<float>(count));
    }

};

// PERTENSOR Q/K quant: one (batch,head) slice [seqlen, hdim] per work-group, with a single global
// scale = absmax_over_all_tokens_and_hidden(X) / kQuantDivisor, x_q = round(X / scale). Pass 1
// accumulates per-channel absmax over all blocks then maxes across channels; Pass 2 re-sweeps to
// write the quantized output. hdim == 128.
template <typename InputType_, index_t kBlockSize_ = 256, typename QuantType_ = int8_t>
struct BlockSpargeQKQuantPipeline
{
    using InputType       = remove_cvref_t<InputType_>;
    using QuantType       = remove_cvref_t<QuantType_>;
    using ComputeDataType = float;
    static constexpr float kQuantDivisor = sparge_quant_absmax_divisor<QuantType>();

    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kHdim      = 128;
    static constexpr index_t kBlock     = 128;

    struct BlockShape
    {
        static constexpr index_t Block_M = kHdim;
        static constexpr index_t Block_N = kBlock;

        static constexpr index_t WarpPerBlock_M = 4;
        static constexpr index_t WarpPerBlock_N = 1;

        static constexpr index_t ThreadPerWarp_M = 16;
        static constexpr index_t ThreadPerWarp_N = 4;

        static constexpr index_t Vector_M = 1;
        static constexpr index_t Vector_N = 2;

        static constexpr index_t Repeat_M =
            Block_M / (WarpPerBlock_M * ThreadPerWarp_M * Vector_M);
        static constexpr index_t Repeat_N =
            Block_N / (WarpPerBlock_N * ThreadPerWarp_N * Vector_N);

        static constexpr index_t BlockSize = kBlockSize;
    };

    using ReduceProblem = BlockReduce2dProblem<ComputeDataType, ComputeDataType, BlockShape>;
    static constexpr bool kNeedCrossWarpSync = (BlockShape::WarpPerBlock_N > 1);

    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    struct Params
    {
        index_t seqlen;
        index_t hdim;
        index_t stride_seq;      // token stride in elements of the bf16 input
        index_t quant_stride_seq; // token stride in elements of the quant output
        const float* km_ptr;     // [hdim] per-channel K-mean (smooth_k); nullptr disables (Q side).
    };

    // LDS: per-channel absmax[hdim] | block reduce scratch[kBlockSize] | (smooth_k) km[hdim].
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(index_t /*hdim*/)
    {
        using x_block_tile =
            decltype(make_static_distributed_tensor<ComputeDataType>(
                MakeXBlockTileDistribution()));
        using y_block_tile =
            decltype(BlockReduce2d<ReduceProblem>::template MakeYBlockTile<x_block_tile>());
        constexpr index_t reduce_bytes =
            BlockReduce2dCrossWarpSync<ReduceProblem>::template GetSmemSize<y_block_tile>();
        return static_cast<index_t>((2 * kHdim + kBlockSize) * sizeof(float)) + reduce_bytes;
    }

    // slice/quant_out: bf16 X and quantized X for this (batch, head) = [seqlen, hdim].
    // scale_out: single global scale for this (batch, head).
    CK_TILE_DEVICE void operator()(const InputType* slice,
                                   QuantType*        quant_out,
                                   float*            scale_out,
                                   const Params&     params,
                                   void*             smem) const
    {
        const index_t tid = get_thread_id();
        assert(params.hdim == kHdim && "sparge QK quant tile path requires hdim == 128");

        float* s_absmax = reinterpret_cast<float*>(smem);            // [kHdim] per-channel
        float* s_scr    = s_absmax + kHdim;                          // [kBlockSize] reduce scratch
        float* s_km     = s_scr + kBlockSize;                        // [kHdim] staged km (smooth_k)
        void*  s_reduce = reinterpret_cast<void*>(s_km + kHdim);

        // smooth_k (K side): stage per-channel km in LDS; Q side passes km_ptr == nullptr -> 0.
        const bool has_km = (params.km_ptr != nullptr);
        for(index_t d = tid; d < kHdim; d += kBlockSize)
        {
            s_absmax[d] = 0.0f;
            s_km[d]     = has_km ? params.km_ptr[d] : 0.0f;
        }
        block_sync_lds();

        auto reduce       = BlockReduce2d<ReduceProblem>{};
        auto reduce_sync  = BlockReduce2dSync<ReduceProblem>{};
        auto reduce_xwarp = BlockReduce2dCrossWarpSync<ReduceProblem>{};
        auto absmax_func  = ReduceOp::AbsMax{};

        // Transposed view [hdim(M), token(N)]; AbsMax over N -> per-channel [M].
        const auto naive_t = make_naive_tensor_view<address_space_enum::global>(
            slice,
            make_tuple(params.seqlen, kHdim),
            make_tuple(params.stride_seq, 1),
            number<1>{},
            number<1>{});
        const auto transposed = transform_tensor_view(
            naive_t,
            make_tuple(make_pass_through_transform(kHdim),
                       make_pass_through_transform(params.seqlen)),
            make_tuple(sequence<1>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        const auto padded_t = pad_tensor_view(
            transposed,
            make_tuple(number<kHdim>{}, number<kBlock>{}),
            sequence<0, 1>{});

        // Pass 1: per-channel absmax across all token blocks.
        for(index_t s_start = 0; s_start < params.seqlen; s_start += kBlock)
        {
            auto x_window = make_tile_window(
                padded_t,
                make_tuple(number<kHdim>{}, number<kBlock>{}),
                {0, s_start},
                MakeXBlockTileDistribution());
            auto x_tile = load_tile(x_window);

            // smooth_k: center K by km[m] before absmax.
            auto abs_tile = make_static_distributed_tensor<ComputeDataType>(
                decltype(x_tile)::get_tile_distribution());
            sweep_tile(x_tile, [&](auto idx) {
                float v = type_convert<ComputeDataType>(x_tile[idx]);
                if(has_km)
                {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        x_tile.get_tile_distribution(), idx);
                    const index_t m = tile_idx.at(number<0>{});
                    v -= s_km[m];
                }
                abs_tile(idx) = v;
            });
            auto amax_tile =
                reduce(abs_tile, absmax_func.GetIdentityValue<ComputeDataType>(), absmax_func);
            reduce_sync(amax_tile, absmax_func);
            if constexpr(kNeedCrossWarpSync)
                reduce_xwarp(amax_tile, s_reduce, absmax_func);

            sweep_tile_span(decltype(amax_tile)::get_distributed_spans()[number<0>{}],
                            [&](auto idx0) {
                constexpr auto m_idx = make_tuple(idx0);
                const auto tile_idx  = get_x_indices_from_distributed_indices(
                    amax_tile.get_tile_distribution(), m_idx);
                const index_t m = tile_idx.at(number<0>{});
                s_absmax[m]     = max(s_absmax[m], amax_tile[m_idx]);
            });
            block_sync_lds();
        }

        // Reduce per-channel absmax -> global scalar -> scale = absmax/kQuantDivisor.
        float local_max = 0.0f;
        for(index_t d = tid; d < kHdim; d += kBlockSize)
            local_max = max(local_max, s_absmax[d]);
        const float global_amax = block_reduce_max_f32<kBlockSize>(local_max, s_scr, tid);
        const float scale = (global_amax > 0.0f) ? (global_amax / kQuantDivisor) : 1.0f;
        if(tid == 0)
            scale_out[0] = scale;
        block_sync_lds();

        // Pass 2: write quant = round(X / scale) in natural [token, hidden].
        const auto naive_in = make_naive_tensor_view<address_space_enum::global>(
            slice,
            make_tuple(params.seqlen, kHdim),
            make_tuple(params.stride_seq, 1),
            number<1>{},
            number<1>{});
        const auto padded_in = pad_tensor_view(
            naive_in,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            sequence<1, 0>{}); // pad token axis only; hidden exact 128

        const auto naive_out = make_naive_tensor_view<address_space_enum::global>(
            quant_out,
            make_tuple(params.seqlen, kHdim),
            make_tuple(params.quant_stride_seq, 1),
            number<1>{},
            number<1>{});
        const auto padded_out = pad_tensor_view(
            naive_out,
            make_tuple(number<kBlock>{}, number<kHdim>{}),
            sequence<1, 0>{});

        for(index_t s_start = 0; s_start < params.seqlen; s_start += kBlock)
        {
            auto in_window = make_tile_window(
                padded_in,
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                {s_start, 0},
                MakeXBlockTileDistribution());
            auto out_window = make_tile_window(
                padded_out,
                make_tuple(number<kBlock>{}, number<kHdim>{}),
                {s_start, 0},
                MakeXBlockTileDistribution());

            auto in_tile  = load_tile(in_window);
            auto out_tile = make_static_distributed_tensor<QuantType>(
                decltype(in_tile)::get_tile_distribution());
            sweep_tile(in_tile, [&](auto idx) {
                // smooth_k: center K by km[c].
                float v = type_convert<ComputeDataType>(in_tile[idx]);
                if(has_km)
                {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        in_tile.get_tile_distribution(), idx);
                    const index_t c = tile_idx.at(number<1>{});
                    v -= s_km[c];
                }
                const float r = (scale > 0.0f) ? (v / scale) : 0.0f;
                if constexpr(std::is_same_v<QuantType, fp8_t>)
                    out_tile(idx) = type_convert<fp8_t>(r);
                else
                    out_tile(idx) = type_convert<int8_t>(saturates<int8_t>{}(r));
            });
            store_tile(out_window, out_tile);
        }
    }
};

// Per Q-block: scores + softmax + sort-based selection -> delta-encoded LUT.
// kMaxKBlocksPow2_ caps the K-block sort capacity (power of two; BLKK=128: 256->32k, 512->64k,
// 1024->128k seqlen_k). The sort/scan/select loops stride over the array (e += kBlockSize), so it
// may exceed kBlockSize but must then be a multiple of it so every element is covered.
template <index_t kMaxKBlocksPow2_ = 256, index_t kBlockSize_ = 256>
struct BlockSpargeMaskPredictionPipeline
{
    static constexpr index_t kBlockSize      = kBlockSize_;
    static constexpr index_t kMaxKBlocksPow2 = kMaxKBlocksPow2_;
    static_assert((kMaxKBlocksPow2 & (kMaxKBlocksPow2 - 1)) == 0 && kMaxKBlocksPow2 > 0,
                  "kMaxKBlocksPow2 must be a power of two");
    static_assert(kMaxKBlocksPow2 <= kBlockSize || (kMaxKBlocksPow2 % kBlockSize) == 0,
                  "when kMaxKBlocksPow2 > kBlockSize it must be a multiple of kBlockSize "
                  "so the strided sort/scan/select loops cover every element");
    // Cross-warp scratch: kBlockSize / min_warp_size, rounded up.
    static constexpr index_t kReduceScratchSlots = (kBlockSize + 31) / 32;
    // OOB sentinel: finite (not -INF) so softmax max-subtract avoids inf-inf=NaN.
    static constexpr float   kScoreOOB           = -1.0e30f;
    // scores_smem is reused across three phases for one M-tile of K-block scores:
    //   1. raw scores      = dot(q_mean, k_means[k])      -> any real value (incl. kScoreOOB).
    //   2. exp probs       = softmax over the raw scores  -> values in (0, 1].
    //   3. selection flag  = kScoreSelected               -> marks a K-block as picked.
    // The phase-3 sentinel is -2.0, which can never appear as a phase-2 prob (those are in
    // (0, 1]) nor as a phase-1 OOB score (-1e30), so the float `== kScoreSelected` test that
    // reads back the selection flag is unambiguous and collision-free.
    static constexpr float   kScoreSelected      = -2.0f;

    static constexpr index_t kHdim = 128;

    // score = dot(q_mean, k_means[k]): load the k_mean rows for kScoreTileM K-blocks as a
    // [kScoreTileM (M), hdim (N)] tile, multiply by the WG-uniform q_mean, reduce along N.
    using ComputeDataType = float;

    static constexpr index_t kScoreTileM = 16; // K-blocks reduced per tile window

    // Tile [M = K-blocks, N = hdim], reduce N. M on the warps (WarpPerBlock_N = 1) so N reduces
    // wholly within a warp and each M-block's score scatters directly (no M-collapse).
    struct ScoreBlockShape
    {
        static constexpr index_t Block_M = kScoreTileM;
        static constexpr index_t Block_N = kHdim;

        static constexpr index_t WarpPerBlock_M = 4;
        static constexpr index_t WarpPerBlock_N = 1;

        static constexpr index_t ThreadPerWarp_M = 4;
        static constexpr index_t ThreadPerWarp_N = 16;

        static constexpr index_t Vector_M = 1;
        static constexpr index_t Vector_N = 4;

        static constexpr index_t Repeat_M =
            Block_M / (WarpPerBlock_M * ThreadPerWarp_M * Vector_M);
        static constexpr index_t Repeat_N =
            Block_N / (WarpPerBlock_N * ThreadPerWarp_N * Vector_N);

        static constexpr index_t BlockSize = kBlockSize;
    };

    using ScoreReduceProblem =
        BlockReduce2dProblem<ComputeDataType, ComputeDataType, ScoreBlockShape>;

    static constexpr bool kScoreNeedCrossWarpSync = (ScoreBlockShape::WarpPerBlock_N > 1);

    CK_TILE_DEVICE static constexpr auto MakeScoreXBlockTileDistribution()
    {
        using S = ScoreBlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetScoreReduceSmemSize()
    {
        using x_block_tile =
            decltype(make_static_distributed_tensor<ComputeDataType>(
                MakeScoreXBlockTileDistribution()));
        using y_block_tile =
            decltype(BlockReduce2d<ScoreReduceProblem>::template MakeYBlockTile<x_block_tile>());
        return BlockReduce2dCrossWarpSync<ScoreReduceProblem>::template GetSmemSize<y_block_tile>();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(index_t hdim, index_t num_k_blocks)
    {
        // [q_mean | scores | sort_keys | sort_vals | aux | scratch | n_target]. The score-reduce
        // cross-warp scratch overlaps the trailing scratch (unused during the score pass), so size
        // the buffer to the larger of the two.
        const index_t reduce_scratch =
            max(static_cast<index_t>((kReduceScratchSlots + 1) * sizeof(int32_t)),
                GetScoreReduceSmemSize());
        return static_cast<index_t>(
            (hdim + num_k_blocks + 3 * kMaxKBlocksPow2) * sizeof(float)) + reduce_scratch;
    }

    struct MaskRunArgs
    {
        const float* k_means;
        const float* q_means;
        const float* k_sim;     // [B, Hk, num_k_blocks] or nullptr
        const float* q_sim;     // [B, Hq, num_q_blocks] or nullptr
        int32_t*     lut_row;
        int32_t*     vbn_ptr;
        index_t b, head, kv_head, q_block;
        index_t nhead_q, nhead_k, num_q_blocks, num_k_blocks, hdim;
        float head_cdfthreshd, head_topk, head_simthreshold;
        index_t causal_type;
        bool    attention_sink;
        index_t seqlen_q, seqlen_k, block_size;
        index_t window_left, window_right;
        float   scale;          // softmax scale for q/k mean scores
    };

    CK_TILE_DEVICE void run_with_indices(const MaskRunArgs& args, void* smem) const
    {
        const float* __restrict__ k_means = args.k_means;
        const float* __restrict__ q_means = args.q_means;
        const float* __restrict__ k_sim   = args.k_sim;
        const float* __restrict__ q_sim   = args.q_sim;
        int32_t*     __restrict__ lut_row = args.lut_row;
        int32_t*     __restrict__ vbn_ptr = args.vbn_ptr;
        const index_t b              = args.b;
        const index_t head           = args.head;
        const index_t kv_head        = args.kv_head;
        const index_t q_block        = args.q_block;
        const index_t nhead_q        = args.nhead_q;
        const index_t nhead_k        = args.nhead_k;
        const index_t num_q_blocks   = args.num_q_blocks;
        const index_t num_k_blocks   = args.num_k_blocks;
        const index_t hdim           = args.hdim;
        const float head_cdfthreshd   = args.head_cdfthreshd;
        const float head_topk          = args.head_topk;
        const float head_simthreshold = args.head_simthreshold;
        const index_t causal_type    = args.causal_type;
        const bool    attention_sink = args.attention_sink;
        const index_t seqlen_q       = args.seqlen_q;
        const index_t seqlen_k       = args.seqlen_k;
        const index_t block_size     = args.block_size;
        const index_t window_left    = args.window_left;
        const index_t window_right   = args.window_right;

        const index_t tid = get_thread_id();

        assert(hdim == kHdim && "sparge mask tile score path requires hdim == 128");
        assert(num_k_blocks <= kMaxKBlocksPow2 && "num_k_blocks exceeds sort capacity");

        float* q_mean_smem = reinterpret_cast<float*>(smem);
        float* scores_smem = q_mean_smem + hdim;
        float*   sort_keys_smem = scores_smem + num_k_blocks;
        int32_t* sort_vals_smem = reinterpret_cast<int32_t*>(sort_keys_smem + kMaxKBlocksPow2);
        float*   aux_smem       = reinterpret_cast<float*>(sort_vals_smem + kMaxKBlocksPow2);
        int32_t* scratch_i32    = reinterpret_cast<int32_t*>(aux_smem + kMaxKBlocksPow2);
        int32_t* n_target_smem  = scratch_i32 + kReduceScratchSlots;
        float*   scratch_f32    = reinterpret_cast<float*>(scratch_i32);

        {
            const float* q_src =
                q_means +
                (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks * hdim +
                static_cast<long_index_t>(q_block) * hdim;
            for(index_t d = tid; d < hdim; d += kBlockSize)
                q_mean_smem[d] = q_src[d];
            block_sync_lds();
        }

        // Causal bounds
        const index_t q_start      = q_block * block_size;
        const index_t causal_delta = (causal_type == 2) ? (seqlen_k - seqlen_q) : index_t{0};
        const index_t right_ext    = (causal_type && window_right >= 0) ? window_right : index_t{0};
        const index_t causal_max_k = causal_type
            ? min(num_k_blocks - 1,
                  (q_start + block_size - 1 + right_ext + causal_delta) / block_size)
            : (num_k_blocks - 1);
        const index_t causal_min_k = (causal_type && window_left >= 0)
            ? max(index_t{0}, (q_start - window_left + causal_delta) / block_size)
            : index_t{0};

        const float* k_mean_base =
            k_means +
            (static_cast<long_index_t>(b) * nhead_k + kv_head) * num_k_blocks * hdim;
        const float scale = args.scale;

        // score[k] = dot(q_mean, k_means[k]) * scale, reduced along hidden (N). OOB K-blocks read 0
        // (padded view) and are stamped kScoreOOB below, as are causal-excluded K-blocks.
        {
            const auto km_view = make_naive_tensor_view<address_space_enum::global>(
                k_mean_base,
                make_tuple(num_k_blocks, kHdim),
                make_tuple(static_cast<index_t>(kHdim), 1),
                number<1>{},
                number<1>{});
            const auto km_padded = pad_tensor_view(
                km_view,
                make_tuple(number<kScoreTileM>{}, number<kHdim>{}),
                sequence<1, 0>{}); // pad K-block axis only; hidden exact 128

            auto reduce       = BlockReduce2d<ScoreReduceProblem>{};
            auto reduce_sync  = BlockReduce2dSync<ScoreReduceProblem>{};
            auto add_func     = ReduceOp::Add{};

            for(index_t k0 = 0; k0 < num_k_blocks; k0 += kScoreTileM)
            {
                auto km_window = make_tile_window(
                    km_padded,
                    make_tuple(number<kScoreTileM>{}, number<kHdim>{}),
                    {k0, 0},
                    MakeScoreXBlockTileDistribution());

                auto km_tile = load_tile(km_window);
                // elementwise q_mean[d] * k_mean[k][d], q_mean broadcast by channel
                auto prod_tile = make_static_distributed_tensor<ComputeDataType>(
                    decltype(km_tile)::get_tile_distribution());
                sweep_tile(km_tile, [&](auto idx) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        km_tile.get_tile_distribution(), idx);
                    const index_t d = tile_idx.at(number<1>{});
                    prod_tile(idx)  = q_mean_smem[d] *
                                      type_convert<ComputeDataType>(km_tile[idx]);
                });

                // N reduces wholly within a warp -> complete per-M-block score, scatter directly.
                auto score_tile =
                    reduce(prod_tile, add_func.GetIdentityValue<ComputeDataType>(), add_func);
                reduce_sync(score_tile, add_func);

                sweep_tile_span(
                    decltype(score_tile)::get_distributed_spans()[number<0>{}],
                    [&](auto idx0) {
                        constexpr auto m_idx = make_tuple(idx0);
                        const auto tile_idx  = get_x_indices_from_distributed_indices(
                            score_tile.get_tile_distribution(), m_idx);
                        const index_t k = k0 + tile_idx.at(number<0>{});
                        if(k < num_k_blocks)
                            scores_smem[k] = score_tile[m_idx] * scale;
                    });
                block_sync_lds();
            }
        }

        // Stamp causal-excluded K-blocks with the OOB sentinel (finite -> no NaN in softmax).
        if(causal_type)
        {
            for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
                if(k < causal_min_k || k > causal_max_k)
                    scores_smem[k] = kScoreOOB;
            block_sync_lds();
        }

        // Exclude low-similarity K blocks (sim <= threshold) from the softmax/selection competition
        // (official pooled_score[~sim]=-inf); they are force-selected via the K-sim union below.
        // Finite OOB sentinel avoids NaN if a whole row is excluded.
        if(k_sim != nullptr && head_simthreshold > 0.0f)
        {
            const float* k_sim_row =
                k_sim +
                static_cast<long_index_t>(b) * nhead_k * num_k_blocks +
                static_cast<long_index_t>(kv_head) * num_k_blocks;
            for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
                if(k_sim_row[k] <= head_simthreshold)
                    scores_smem[k] = kScoreOOB;
            block_sync_lds();
        }

        // Softmax; normalize only for CDF mode (TopK is scaling-invariant).
        float local_max = -INFINITY;
        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
            local_max = (scores_smem[k] > local_max) ? scores_smem[k] : local_max;
        const float max_score = block_reduce_max_f32<kBlockSize>(local_max, scratch_f32, tid);

        float local_sum = 0.0f;
        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
        {
            float p = ck_tile::exp(scores_smem[k] - max_score);
            scores_smem[k] = p;
            local_sum += p;
        }
        const float sum_exp = block_reduce_sum_f32<kBlockSize>(local_sum, scratch_f32, tid);

        const bool topk_mode = (head_topk > 0.0f);
        // CDF runs on the unnormalized exp scores (each in (0,1] after the max-shift): rather than
        // dividing every score by sum_exp (an extra LDS pass + sync), searchsorted below compares the
        // unnormalized cumsum against head_cdfthreshd * sum_exp. TopK is scale-invariant either way.

        // Dispatch sort+select to smallest pow-of-2 >= num_k_blocks.
        int32_t n_target = 0;
        auto do_select = [&](auto N_const) {
            constexpr index_t N_pow2 = decltype(N_const)::value;

            for(index_t e = tid; e < N_pow2; e += kBlockSize)
            {
                const bool valid = e < num_k_blocks;
                float p          = valid ? scores_smem[e] : -1.0f;
                if(!(p == p)) p = -1.0f;
                sort_keys_smem[e] = p;
                sort_vals_smem[e] = valid ? static_cast<int32_t>(e) : int32_t{-1};
            }
            block_sync_lds();

            // Both TopK and CDF go through the bitonic sort: the BlockTopkStream2D fast path
            // returned the same argmax index repeatedly under near-tied scores (duplicate indices ->
            // LUT/VBN under-count), whereas the sort yields distinct sorted indices.
            bitonic_sort_desc_smem<N_pow2, kBlockSize>(sort_keys_smem, sort_vals_smem, tid);
            // Cumsum scan only needed for the CDF threshold path.
            if(!topk_mode)
                block_scan_inclusive_sum_smem<N_pow2, kBlockSize>(sort_keys_smem, aux_smem, tid);

            if(topk_mode)
            {
                n_target = max(int32_t{1},
                               static_cast<int32_t>(head_topk * static_cast<float>(num_k_blocks)));
            }
            else
            {
                // official CDF: num_to_select = searchsorted(cdf, thr, right=True) = first index
                // whose inclusive prefix exceeds thr (the crossing block is NOT selected); clamp >=1.
                // cumsum is unnormalized (see above), so scale the threshold by sum_exp.
                const float cdf_abs_threshd = head_cdfthreshd * sum_exp;
                int32_t cand = num_k_blocks;
                for(index_t e = tid; e < static_cast<index_t>(num_k_blocks); e += kBlockSize)
                    if(sort_keys_smem[e] > cdf_abs_threshd)
                    {
                        cand = static_cast<int32_t>(e);
                        break; // sorted desc -> first crossing is this thread's min
                    }
                int32_t reduced = block_reduce_min_i32<kBlockSize>(cand, scratch_i32, tid);
                if(tid == 0)
                {
                    if(reduced > num_k_blocks) reduced = num_k_blocks;
                    if(reduced < 1)            reduced = 1;
                    n_target_smem[0] = reduced;
                }
                block_sync_lds();
                n_target = n_target_smem[0];
            }

            for(index_t e = tid;
                e < N_pow2 && static_cast<int32_t>(e) < n_target;
                e += kBlockSize)
            {
                int32_t orig = sort_vals_smem[e];
                if(orig >= 0)
                {
                    const bool causal_ok = !causal_type ||
                        (orig >= causal_min_k && orig <= causal_max_k);
                    if(causal_ok)
                        scores_smem[orig] = kScoreSelected;
                }
            }
            block_sync_lds();
        };

        if(num_k_blocks <= 32)        do_select(integral_constant<index_t, 32>{});
        else if(num_k_blocks <= 64)   do_select(integral_constant<index_t, 64>{});
        else if(num_k_blocks <= 128)  do_select(integral_constant<index_t, 128>{});
        else                          do_select(integral_constant<index_t, kMaxKBlocksPow2>{});

        // K-sim union: force-select the low-sim blocks excluded above (within causal range).
        if(k_sim != nullptr && head_simthreshold > 0.0f)
        {
            const float* k_sim_row =
                k_sim +
                static_cast<long_index_t>(b) * nhead_k * num_k_blocks +
                static_cast<long_index_t>(kv_head) * num_k_blocks;
            for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
            {
                const bool causal_ok = !causal_type ||
                    (k >= causal_min_k && k <= causal_max_k);
                if(k_sim_row[k] <= head_simthreshold && causal_ok &&
                   scores_smem[k] != kScoreSelected)
                    scores_smem[k] = kScoreSelected;
            }
            block_sync_lds();
        }

        // (Empty-selection fallback runs after the LUT scan below.)

        // Q-sim union: low-sim Q block -> select its whole causal K range.
        if(q_sim != nullptr && head_simthreshold > 0.0f)
        {
            const float q_block_sim =
                q_sim[static_cast<long_index_t>(b) * nhead_q * num_q_blocks +
                      static_cast<long_index_t>(head) * num_q_blocks +
                      q_block];
            if(q_block_sim <= head_simthreshold)
            {
                const index_t lo = causal_type ? causal_min_k : index_t{0};
                const index_t hi = causal_type ? causal_max_k : (num_k_blocks - 1);
                for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
                    if(k >= lo && k <= hi && scores_smem[k] != kScoreSelected)
                        scores_smem[k] = kScoreSelected;
            }
            block_sync_lds();
        }

        // Attention sink: always keep block 0.
        if(tid == 0 && attention_sink && num_k_blocks > 0 &&
           scores_smem[0] != kScoreSelected)
            scores_smem[0] = kScoreSelected;
        block_sync_lds();

        // LUT build: flag -> scan -> compact -> delta. Flag+scan run on the smallest pow-of-2
        // bucket covering num_k_blocks (same dispatch as do_select); the scan's CTA barriers scale
        // with that bucket, not kMaxKBlocksPow2, so short/medium seqlens skip the zero tail.
        auto build_lut_flags_and_scan = [&](auto N_const) {
            constexpr index_t N_pow2 = decltype(N_const)::value;
            for(index_t e = tid; e < N_pow2; e += kBlockSize)
                sort_keys_smem[e] = (e < num_k_blocks &&
                                     scores_smem[e] == kScoreSelected) ? 1.0f : 0.0f;
            block_sync_lds();
            block_scan_inclusive_sum_smem<N_pow2, kBlockSize>(sort_keys_smem, aux_smem, tid);
        };
        if(num_k_blocks <= 32)        build_lut_flags_and_scan(integral_constant<index_t, 32>{});
        else if(num_k_blocks <= 64)   build_lut_flags_and_scan(integral_constant<index_t, 64>{});
        else if(num_k_blocks <= 128)  build_lut_flags_and_scan(integral_constant<index_t, 128>{});
        else build_lut_flags_and_scan(integral_constant<index_t, kMaxKBlocksPow2>{});

        const index_t n_after_scan = (num_k_blocks > 0)
            ? static_cast<index_t>(sort_keys_smem[num_k_blocks - 1]) : 0;

        // Fallback: every selection pass produced zero. Emit a single-element LUT and exit.
        if(n_after_scan == 0)
        {
            if(tid == 0)
            {
                if(causal_min_k <= causal_max_k && causal_min_k < num_k_blocks)
                {
                    lut_row[0] = static_cast<int32_t>(causal_min_k);
                    *vbn_ptr   = 1;
                }
                else
                {
                    *vbn_ptr = 0;
                }
            }
            return;
        }

        for(index_t k = tid; k < num_k_blocks; k += kBlockSize)
            if(scores_smem[k] == kScoreSelected)
            {
                int pos = static_cast<int>(sort_keys_smem[k]) - 1;
                sort_vals_smem[pos] = static_cast<int32_t>(k);
            }
        block_sync_lds();

        for(index_t e = tid; e < n_after_scan; e += kBlockSize)
        {
            int curr = sort_vals_smem[e];
            int prev = (e == 0) ? 0 : sort_vals_smem[e - 1];
            lut_row[e] = curr - prev;
        }
        if(tid == 0) *vbn_ptr = static_cast<int32_t>(n_after_scan);
    }

    CK_TILE_DEVICE void operator()(
        const float* __restrict__ k_means,
        const float* __restrict__ q_means,
        const float* __restrict__ k_sim,
        const float* __restrict__ q_sim,
        int32_t* __restrict__ lut_out,
        int32_t* __restrict__ valid_block_num_out,
        index_t nhead_q,
        index_t nhead_k,
        index_t nhead_ratio_qk,
        index_t num_q_blocks,
        index_t num_k_blocks,
        index_t hdim,
        float cdfthreshd,
        float topk,
        float simthreshold,
        const float* __restrict__ cdfthreshd_per_head,    // nullable [H_q]
        const float* __restrict__ topk_per_head,           // nullable [H_q]
        const float* __restrict__ simthreshold_per_head,  // nullable [H_q]
        index_t causal_type,
        bool attention_sink,
        index_t seqlen_q,
        index_t seqlen_k,
        index_t block_size,
        index_t window_left,
        index_t window_right,
        float scale,
        void* smem) const
    {
        const index_t gid = get_block_id();

        const index_t q_block = gid % num_q_blocks;
        const index_t head    = (gid / num_q_blocks) % nhead_q;
        const index_t b       = gid / (nhead_q * num_q_blocks);
        const index_t kv_head = head / nhead_ratio_qk;

        int32_t* lut_row =
            lut_out +
            (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks * num_k_blocks +
            static_cast<long_index_t>(q_block) * num_k_blocks;
        int32_t* vbn_ptr =
            valid_block_num_out +
            (static_cast<long_index_t>(b) * nhead_q + head) * num_q_blocks +
            q_block;

        // Per-head overrides fall back to the scalar args when the pointer is null.
        const float head_cdfthreshd    = cdfthreshd_per_head   ? cdfthreshd_per_head[head]   : cdfthreshd;
        const float head_topk           = topk_per_head          ? topk_per_head[head]          : topk;
        const float head_simthreshold  = simthreshold_per_head ? simthreshold_per_head[head] : simthreshold;

        MaskRunArgs args;
        args.k_means      = k_means;
        args.q_means      = q_means;
        args.k_sim        = k_sim;
        args.q_sim        = q_sim;
        args.lut_row      = lut_row;
        args.vbn_ptr      = vbn_ptr;
        args.b            = b;
        args.head         = head;
        args.kv_head      = kv_head;
        args.q_block      = q_block;
        args.nhead_q      = nhead_q;
        args.nhead_k      = nhead_k;
        args.num_q_blocks = num_q_blocks;
        args.num_k_blocks = num_k_blocks;
        args.hdim         = hdim;
        args.head_cdfthreshd   = head_cdfthreshd;
        args.head_topk          = head_topk;
        args.head_simthreshold = head_simthreshold;
        args.causal_type    = causal_type;
        args.attention_sink = attention_sink;
        args.seqlen_q       = seqlen_q;
        args.seqlen_k       = seqlen_k;
        args.block_size     = block_size;
        args.window_left    = window_left;
        args.window_right   = window_right;
        args.scale          = scale;
        run_with_indices(args, smem);
    }
};

} // namespace ck_tile

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

// Merge of the two KV-split partial outputs of the gfx1250 batch-prefill page tile
// (FmhaBatchPrefillQrTdmPageTileKernel with KvSplits 2). For every (query, head) row:
//   O = (w0 * O0 + w1 * O1) / (w0 + w1),  w_s = exp(lse_s - max(lse_0, lse_1)).
// A split without keys (O_s = 0, lse_s = -inf) weighs 0; a row whose splits are both empty is 0.
// Workspace written by the split kernel: partial O [2][rows][nhead][HeadDim] and LSE
// [2][nhead][rows], FP32, where row is the query's position in the packed (group-mode) query
// dimension and rows >= the packed query count. One wave32 per (query, head) row, four rows per
// workgroup, the batch in grid y; max_seqlen_q * nhead must fit index_t (the KV split is for small
// grids).
template <typename ODataType_, index_t HeadDim_>
struct FmhaBatchPrefillSplitMergeKernel
{
    using ODataType                        = remove_cvref_t<ODataType_>;
    static constexpr index_t kHeadDim      = HeadDim_;
    static constexpr index_t kWaveSize     = 32;
    static constexpr index_t kRowsPerBlock = 4;
    static constexpr index_t kBlockSize    = kWaveSize * kRowsPerBlock;
    static constexpr index_t kBlockPerCu   = 1;
    static_assert(kHeadDim % kWaveSize == 0, "each lane merges whole columns");

    struct Kargs
    {
        const float* partial_o;
        const float* lse;
        void* o_ptr;
        const int32_t* seqstart_q; // [batch + 1] packed query offsets
        index_t nhead;
        index_t rows;           // query rows of one workspace plane
        index_t stride_o;       // O row stride (elements)
        index_t nhead_stride_o; // O head stride (elements)
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* partial_o,
                                                  const void* lse,
                                                  void* o_ptr,
                                                  const void* seqstart_q,
                                                  index_t nhead,
                                                  index_t rows,
                                                  index_t stride_o,
                                                  index_t nhead_stride_o)
    {
        return Kargs{reinterpret_cast<const float*>(partial_o),
                     reinterpret_cast<const float*>(lse),
                     o_ptr,
                     reinterpret_cast<const int32_t*>(seqstart_q),
                     nhead,
                     rows,
                     stride_o,
                     nhead_stride_o};
    }

    CK_TILE_HOST static dim3 GridSize(index_t batch, index_t nhead, index_t max_seqlen_q)
    {
        return dim3(integer_divide_ceil(max_seqlen_q * nhead, kRowsPerBlock), batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // Launch functor for ck_tile::launch_kernel; grids from GridSize().
    CK_TILE_HOST static auto MakeKernel(const Kargs& kargs, dim3 grids)
    {
        return make_kernel<kBlockPerCu, gfx125_t>(
            FmhaBatchPrefillSplitMergeKernel{}, grids, BlockSize(), 0, kargs);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t lane = threadIdx.x % kWaveSize;
        // (query, head) row of this wave in its batch, in the order of the O rows.
        const index_t task        = blockIdx.x * kRowsPerBlock + threadIdx.x / kWaveSize;
        const index_t query       = task / kargs.nhead;
        const index_t head        = task % kargs.nhead;
        const index_t query_start = kargs.seqstart_q[blockIdx.y];
        if(query >= kargs.seqstart_q[blockIdx.y + 1] - query_start)
            return;
        const index_t row = query_start + query;

        const index_t lse_index = head * kargs.rows + row;
        const float l0          = kargs.lse[lse_index];
        const float l1          = kargs.lse[kargs.nhead * kargs.rows + lse_index];
        const float maximum     = fmaxf(l0, l1);
        // An empty split has LSE -inf: do not evaluate -inf minus -inf. Nothing is below -inf, so
        // x <= -inf is x == -inf for every x (NaN included), without -Wfloat-equal.
        constexpr float kMinusInf = -numeric<float>::infinity();
        const float w0            = l0 <= kMinusInf ? 0.0f : expf(l0 - maximum);
        const float w1            = l1 <= kMinusInf ? 0.0f : expf(l1 - maximum);
        const float denominator   = w0 + w1;
        const float inverse       = denominator == 0.0f ? 0.0f : 1.0f / denominator;

        const long_index_t o_index =
            (static_cast<long_index_t>(row) * kargs.nhead + head) * kHeadDim;
        const long_index_t o_plane = static_cast<long_index_t>(kargs.rows) * kargs.nhead * kHeadDim;
        ODataType* out             = static_cast<ODataType*>(kargs.o_ptr) +
                         static_cast<long_index_t>(row) * kargs.stride_o +
                         static_cast<long_index_t>(head) * kargs.nhead_stride_o;
        static_for<0, kHeadDim / kWaveSize, 1>{}([&](auto i) {
            const index_t d = lane + decltype(i)::value * kWaveSize;
            const float value =
                (w0 * kargs.partial_o[o_index + d] + w1 * kargs.partial_o[o_plane + o_index + d]) *
                inverse;
            out[d] = type_convert<ODataType>(value);
        });
    }
};

} // namespace ck_tile

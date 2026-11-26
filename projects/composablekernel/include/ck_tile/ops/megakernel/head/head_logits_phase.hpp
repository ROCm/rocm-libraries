// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/megakernel/gemm_phase.hpp"
#include "ck_tile/ops/megakernel/head/head_common.hpp"

namespace ck_tile {

/// @brief The lm_head GEMM with softcap and argmax fused into its epilogue, as a megakernel phase.
///
/// logits[r, n] = softcap(sum_k xn[r, k] * lm_head[n, k]) over this rank's vocab shard
/// n in [0, N), token id vocab_offset + n. Each output tile reduces its logits to one argmax key
/// per row (head_common.hpp) in registers and LDS, then merges it into row_keys[r] with a 64-bit
/// global atomic max, so the logits never reach memory. A full fp32 row of softcapped logits
/// (for sampling) is written only if args.logits is set.
///
/// row_keys must be 0 before the phase runs. They are zeroed once at allocation and cleared again
/// by the phase that consumes them (HeadAcceptPhase with clear_keys), so no reset is needed
/// between launches, as with the megakernel counters.
///
/// Work items, counter slots, tile order and the A/B windows are those of GemmPhase over the
/// same GEMM kernel; only the epilogue differs. The kernel's own epilogue and e_ptr are unused.
/// k_batch must be 1 (the softcap is not linear, so split-K partials cannot be capped).
///
/// TODO: per-tile reduction uses one LDS atomic per (thread, row); a lane shuffle over the WMMA
/// C layout first would cut that by the lanes sharing a row. The optional logits store writes
/// the C distribution directly (no LDS shuffle), so it is not coalesced.
template <typename GemmKernel_, head_softcap_numerics Numerics_ = head_softcap_numerics::hf_bf16>
struct HeadLogitsPhase
{
    using Gemm             = GemmPhase<GemmKernel_>;
    using Kernel           = typename Gemm::Kernel;
    using GemmArgs         = typename Gemm::Args;
    using TilePartitioner  = typename Gemm::TilePartitioner;
    using GemmPipeline     = typename Gemm::GemmPipeline;
    using ADataType        = typename Gemm::ADataType;
    using BDataType        = typename Gemm::BDataType;
    using barrier_pipeline = typename Gemm::barrier_pipeline;

    static constexpr head_softcap_numerics kNumerics = Numerics_;
    static constexpr index_t kBlockSize              = Gemm::kBlockSize;
    static constexpr index_t kSlotRows               = Gemm::kSlotRows;
    static constexpr index_t kMPerBlock              = TilePartitioner::MPerBlock;
    static constexpr index_t kNPerBlock              = TilePartitioner::NPerBlock;

    static_assert(!is_wavelet_pipeline<GemmPipeline>,
                  "HeadLogitsPhase: wavelet pipelines (load waves without C) are not supported");
    static_assert(Kernel::NumATensor == 1 && Kernel::NumBTensor == 1 && Kernel::NumDTensor == 0,
                  "HeadLogitsPhase: one A, one B, no D");

    struct Args
    {
        GemmArgs gemm;        // A = normed x [M, K], B = lm_head shard [N, K]; e_ptr unused
        uint64_t* row_keys;   // [M] argmax keys, 0 on entry
        float* logits;        // optional [M, ld_logits] softcapped fp32 logits, nullptr = none
        index_t ld_logits;    //
        index_t vocab_offset; // token id of shard column 0 (rank * shard size under TP)
        float multiplier;     // output_multiplier
        float softcap;        // final_logit_softcapping
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(Gemm::GetSmemSize(), static_cast<index_t>(kMPerBlock * sizeof(uint64_t)));
    }

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return Gemm::NumWork(a.gemm); }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return Gemm::NumSlots(a.gemm); }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t slot)
    {
        return Gemm::ItemsPerSlot(a.gemm, slot);
    }
    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return Gemm::SlotOf(a.gemm, work);
    }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        return Gemm::RowRange(a.gemm, work);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.gemm.k_batch == 1 && a.row_keys != nullptr &&
               (a.logits == nullptr || a.ld_logits >= a.gemm.N) && a.vocab_offset >= 0 &&
               Gemm::IsSupportedArgument(a.gemm);
    }

    CK_TILE_DEVICE static auto InitBarriers() { return Gemm::InitBarriers(); }

    template <typename BarrierToken>
    CK_TILE_DEVICE static void
    Run(const Args& args, index_t work, void* smem, const BarrierToken& barriers)
    {
        const GemmArgs& kargs = args.gemm;
        const auto [iM, iN]   = Gemm::TileIndex(kargs, work);
        const index_t i_m     = amd_wave_read_first_lane(iM * kMPerBlock);
        const index_t i_n     = amd_wave_read_first_lane(iN * kNPerBlock);
        const auto split      = typename Kernel::SplitKBatchOffset(kargs, 0);

        const std::array<const ADataType*, 1> as_ptr{
            static_cast<const ADataType*>(kargs.as_ptr[0])};
        const std::array<const BDataType*, 1> bs_ptr{
            static_cast<const BDataType*>(kargs.bs_ptr[0])};

        const auto& as_window = Kernel::MakeABlockWindows(as_ptr, kargs, split.splitted_k, i_m);
        const auto& bs_window = Kernel::MakeBBlockWindows(bs_ptr, kargs, split.splitted_k, i_n);
        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(split.splitted_k));

        auto c = RunGemmPipeline<GemmPipeline>(as_window,
                                               typename Kernel::AElementWise{},
                                               bs_window,
                                               typename Kernel::BElementWise{},
                                               num_loop,
                                               smem,
                                               barriers);

        Epilogue(args, c, i_m, i_n, smem);
    }

    /// Softcaps @p c in place, merges each row's argmax key into args.row_keys and optionally
    /// stores the softcapped tile. Collective.
    template <typename CTile>
    CK_TILE_DEVICE static void
    Epilogue(const Args& args, CTile& c, index_t i_m, index_t i_n, void* smem)
    {
        const index_t M      = args.gemm.M;
        const index_t N      = args.gemm.N;
        const index_t tid    = static_cast<index_t>(threadIdx.x);
        uint64_t* s_keys     = static_cast<uint64_t*>(smem);
        const float mult     = args.multiplier;
        const float cap      = args.softcap;
        const index_t token0 = args.vocab_offset + i_n;

        // The pipeline's last LDS reads are done before the keys overwrite its buffers.
        block_sync_lds();
        for(index_t r = tid; r < kMPerBlock; r += kBlockSize)
            s_keys[r] = 0;
        block_sync_lds();

        constexpr auto spans = CTile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            uint64_t best = 0;
            index_t row   = 0;
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto ij = make_tuple(idx0, idx1);
                const auto xi =
                    get_x_indices_from_distributed_indices(c.get_tile_distribution(), ij);
                row               = xi.at(number<0>{});
                const index_t col = xi.at(number<1>{});
                const float v     = head_softcap<kNumerics>(c[ij], mult, cap);
                c(ij)             = v;
                // Columns past the shard (B padding) must not win.
                if(i_n + col < N)
                    best = max(best, head_argmax_key(v, token0 + col));
            });
            if(best != 0)
                __scoped_atomic_fetch_max(
                    &s_keys[row], best, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
        });
        block_sync_lds();

        for(index_t r = tid; r < kMPerBlock && i_m + r < M; r += kBlockSize)
        {
            const uint64_t k = s_keys[r];
            if(k != 0)
                __scoped_atomic_fetch_max(
                    &args.row_keys[i_m + r], k, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        }

        if(args.logits != nullptr)
        {
            const auto view =
                make_naive_tensor_view<address_space_enum::global>(args.logits,
                                                                   make_tuple(M, N),
                                                                   make_tuple(args.ld_logits, 1),
                                                                   number<1>{},
                                                                   number<1>{});
            const auto padded =
                pad_tensor_view(view,
                                make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                sequence<true, true>{});
            auto window = make_tile_window(
                padded, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {i_m, i_n});
            store_tile(window, c);
        }
    }
};

} // namespace ck_tile

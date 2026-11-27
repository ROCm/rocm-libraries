// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"

namespace ck_tile {

// FP8 sched softmax streams: row max trees, then exps interleaved with output rescale and row sums.
template <typename Geometry, typename OutputFragments>
struct FmhaTdmSchedFp8Softmax
{
    using ScoreSoftmax = FmhaTdmSchedSplitSoftmaxFor<Geometry::kQkStages>;

    // Marks a softmax whose QK stages share one scale.
    struct NoColumnScale
    {
    };

    // Max of N values by max3 levels; a two-value remainder carries to the next level.
    template <index_t N>
    CK_TILE_DEVICE static float TreeMax(const array<float, N>& v)
    {
        if constexpr(N == 1)
            return v[number<0>{}];
        else if constexpr(N == 2)
            return max(v[number<0>{}], v[number<1>{}]);
        else
        {
            constexpr index_t kTriples = N / 3;
            constexpr index_t kRest    = N % 3;
            array<float, kTriples + kRest> next{};
            static_for<0, kTriples, 1>{}([&](auto i) {
                constexpr index_t k = 3 * decltype(i)::value;
                next(i) = max(max(v[number<k>{}], v[number<k + 1>{}]), v[number<k + 2>{}]);
            });
            static_for<0, kRest, 1>{}([&](auto i) {
                constexpr index_t k          = decltype(i)::value;
                next(number<kTriples + k>{}) = v[number<3 * kTriples + k>{}];
            });
            return TreeMax<kTriples + kRest>(next);
        }
    }

    // Max of N values by Chains independent max3 chains, then a tree over the chain results.
    template <index_t Chains, index_t N>
    CK_TILE_DEVICE static float ChainedMax(const array<float, N>& v)
    {
        constexpr index_t kPerChain = (N + Chains - 1) / Chains;
        array<float, Chains> acc{};
        static_for<0, Chains, 1>{}([&](auto c) {
            constexpr index_t kBegin = decltype(c)::value * kPerChain;
            constexpr index_t kEnd   = min(kBegin + kPerChain, N);
            acc(c)                   = v[number<kBegin>{}];
            static_for<0, (kEnd - kBegin) / 2, 1>{}([&](auto i) {
                constexpr index_t k = kBegin + 1 + 2 * decltype(i)::value;
                if constexpr(k + 1 < kEnd)
                    acc(c) = max(max(acc[c], v[number<k>{}]), v[number<k + 1>{}]);
                else
                    acc(c) = max(acc[c], v[number<k>{}]);
            });
        });
        return TreeMax(acc);
    }

    // Copies the score pairs of row half Half, QK stages [SuBegin, SuEnd), into values from First.
    template <index_t Half,
              index_t SuBegin,
              index_t SuEnd,
              index_t First,
              typename ScoreTensor,
              index_t N>
    CK_TILE_DEVICE static void GatherRowPairs(const ScoreTensor& score, array<float, N>& values)
    {
        using Mapping = typename ScoreSoftmax::Mapping;
        static_assert(First + (SuEnd - SuBegin) * 4 * Mapping::kPairsPerFragment <= N);
        static_for<SuBegin, SuEnd, 1>{}([&](auto su) {
            static_for<0, 2, 1>{}([&](auto h) {
                static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto p) {
                    constexpr index_t at =
                        First + (((su - SuBegin) * 2 + h) * Mapping::kPairsPerFragment + p) * 2;
                    const auto x =
                        Mapping::template LoadPair<Half * 2 + h,
                                                   su * Mapping::kPairsPerFragment + p>(score);
                    values(number<at>{})     = x[0];
                    values(number<at + 1>{}) = x[1];
                });
            });
        });
    }

    // Log2 softmax scale of row band Half: one per workgroup, or one per lane row band.
    template <index_t Half, typename LogScale>
    CK_TILE_DEVICE static float BandScale(const LogScale& log2e_scale)
    {
        if constexpr(std::is_same_v<LogScale, float>)
            return log2e_scale;
        else
            return log2e_scale[number<Half>{}];
    }

    template <typename Problem,
              typename ScoreTensor,
              typename RowTensor,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(ScoreTensor& score,
                                                     RowTensor& row_max,
                                                     LogScale log2e_scale,
                                                     float& delta_m0,
                                                     float& delta_m1,
                                                     const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
        // One max3 tree per row over both MSBs and the old max, then a single lane exchange.
        auto row_part = [&](auto half, float old_max) {
            array<float, 2 * Mapping::kElementsPerMsb + 1> values{};
            values(number<0>{}) = old_max;
            if constexpr(std::is_same_v<ColScale, NoColumnScale>)
                GatherRowPairs<decltype(half)::value, 0, Mapping::kNumSu, 1>(score, values);
            // D64 blockscale sink tiles stay at three workgroups per CU with fewer max partials.
            constexpr bool kChainedRowMax =
                Problem::BlockFmhaShape::kQKHeaddim == 64 && Problem::kHasSink &&
                !Problem::FmhaMask::IsMasking &&
                Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                Problem::QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
            const float lane_max = [&]() {
                if constexpr(!std::is_same_v<ColScale, NoColumnScale>)
                {
                    // Positive column scales commute with each stage's max.
                    array<float, Mapping::kNumSu + 1> scaled{};
                    scaled(number<0>{}) = old_max;
                    static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
                        array<float, 4 * Mapping::kPairsPerFragment> stage{};
                        GatherRowPairs<decltype(half)::value, su, su + 1, 0>(score, stage);
                        scaled(number<su + 1>{}) = TreeMax(stage) * col_scale[su];
                    });
                    return TreeMax(scaled);
                }
                else if constexpr(kChainedRowMax)
                    return ChainedMax<2>(values);
                else
                    return TreeMax(values);
            }();
            const float scale = BandScale<decltype(half)::value>(log2e_scale);
            return Softmax::template RunPart1<kValidateMax>(
                lane_max, Softmax::PermuteLaneX16(lane_max), old_max * scale, scale);
        };
        const auto p1_m0 = row_part(number<0>{}, old_max_m0);
        const auto p1_m1 = row_part(number<1>{}, old_max_m1);

        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    // Lane max of QK stage Su's columns per row half, folded into part when Accumulate.
    template <index_t Su, bool Accumulate, typename ScoreTensor>
    CK_TILE_DEVICE static void StageLaneMax(const ScoreTensor& score, array<float, 2>& part)
    {
        using Mapping = typename ScoreSoftmax::Mapping;
        static_for<0, 2, 1>{}([&](auto half) {
            constexpr index_t kFirst = Accumulate ? 1 : 0;
            array<float, kFirst + 4 * Mapping::kPairsPerFragment> values{};
            if constexpr(Accumulate)
                values(number<0>{}) = part[half];
            GatherRowPairs<decltype(half)::value, Su, Su + 1, kFirst>(score, values);
            part(half) = TreeMax(values);
        });
    }

    // RunSplitSoftmaxPart01 for scores whose earlier QK stages are already in part.
    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01FromPartial(ScoreTensor& score,
                                                                RowTensor& row_max,
                                                                float log2e_scale,
                                                                float& delta_m0,
                                                                float& delta_m1,
                                                                const array<float, 2>& part)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;
        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                      !Problem::FmhaMask::IsMasking && !Problem::kHasLogitsSoftCap);
        static_assert(RowTensor::get_thread_buffer_size() == 2);
        constexpr index_t kSu = Mapping::kNumSu - 1;
        auto row_part         = [&](auto half, float old_max) {
            array<float, 2 + 4 * Mapping::kPairsPerFragment> values{};
            values(number<0>{}) = old_max;
            values(number<1>{}) = part[half];
            GatherRowPairs<decltype(half)::value, kSu, kSu + 1, 2>(score, values);
            const float lane_max = TreeMax(values);
            return Softmax::template RunPart1<false>(
                lane_max, Softmax::PermuteLaneX16(lane_max), old_max * log2e_scale, log2e_scale);
        };
        const auto p1_m0 = row_part(number<0>{}, row_max.get_thread_buffer()[number<0>{}]);
        const auto p1_m1 = row_part(number<1>{}, row_max.get_thread_buffer()[number<1>{}]);
        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    // Row-sum add tree per MSB: nodes 0-7 add leaf pairs, 8-14 add nodes, 15 folds the pair.
    struct ExpInterleavePlan
    {
        static constexpr index_t kPairsPerMsb = 16;
        static constexpr index_t kNodesPerMsb = 16;
        static constexpr index_t kNumMsb      = 4;
        static constexpr index_t kPairs       = kPairsPerMsb * kNumMsb;
        static constexpr index_t kNodes       = kNodesPerMsb * kNumMsb;
        static constexpr index_t kLag         = 2;
        // Tree levels of one MSB: 8 leaf adds, then 4, 2 and the root, then the fold node.
        static constexpr index_t kLevel1   = kPairsPerMsb / 2;
        static constexpr index_t kLevel2   = kLevel1 + kLevel1 / 2;
        static constexpr index_t kRootNode = kLevel2 + kLevel1 / 4;
        static constexpr index_t kFoldNode = kRootNode + 1;
        static_assert(kFoldNode + 1 == kNodesPerMsb);

        index_t slot_node[kPairs] = {};
        index_t tail[kNodes]      = {};
        index_t num_tail          = 0;

        // First input of node k; adds read inputs a and a + 1, the fold reads node 14.
        static constexpr index_t InputA(index_t k)
        {
            if(k < kLevel1)
                return 2 * k;
            if(k < kLevel2)
                return 2 * (k - kLevel1);
            if(k < kRootNode)
                return kLevel1 + 2 * (k - kLevel2);
            return k == kRootNode ? kLevel2 : kRootNode;
        }
        static constexpr bool InputIsLeaf(index_t k) { return k < kLevel1; }

        // Greedily give each slot the first add whose inputs are kLag slots old.
        static constexpr ExpInterleavePlan Make()
        {
            ExpInterleavePlan plan{};
            index_t done[kNodes] = {};
            for(index_t n = 0; n < kNodes; ++n)
                done[n] = -1;
            auto input_slot = [&](index_t n, index_t which) {
                const index_t msb = n / kNodesPerMsb, k = n % kNodesPerMsb;
                const index_t a = InputA(k) + (k < kFoldNode ? which : 0);
                return InputIsLeaf(k) ? msb * kPairsPerMsb + a : done[msb * kNodesPerMsb + a];
            };
            auto ready_by = [&](index_t n, index_t slot) {
                const index_t x = input_slot(n, 0), y = input_slot(n, 1);
                return done[n] < 0 && x >= 0 && y >= 0 && x + kLag <= slot && y + kLag <= slot;
            };
            for(index_t q = 0; q < kPairs; ++q)
            {
                plan.slot_node[q] = -1;
                for(index_t n = 0; n < kNodes; ++n)
                    if(ready_by(n, q))
                    {
                        plan.slot_node[q] = n;
                        done[n]           = q;
                        break;
                    }
            }
            for(index_t n = 0; n < kNodes; ++n)
                if(done[n] < 0)
                {
                    plan.tail[plan.num_tail++] = n;
                    done[n]                    = kPairs;
                }
            return plan;
        }
    };

    // Fused output rescale: each slot issues exp, fma, exp, in-place mul, add, so no exp pairs up.
    template <bool SumRows,
              typename ScoreTensor,
              typename Fragments,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static auto RunInterleavedLocalSums(ScoreTensor& score,
                                                       Fragments& output,
                                                       float exponent_max_m0,
                                                       float exponent_max_m1,
                                                       float delta_m0,
                                                       float delta_m1,
                                                       LogScale log2e_scale,
                                                       float& output_scale_m0,
                                                       float& output_scale_m1,
                                                       const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;
        using Pair    = typename Softmax::Pair;
        using Plan    = ExpInterleavePlan;
        static_assert(Mapping::kNumMsb == Plan::kNumMsb &&
                      Mapping::kPairsPerMsb == Plan::kPairsPerMsb);
        constexpr index_t kOutPairsPerFragment = OutputFragments::kElementsPerFragment / 2;
        // Output vs score pairs: D64 rescales one per two slots, D256 two per slot.
        constexpr index_t kOutPairs = OutputFragments::kNumFragments * kOutPairsPerFragment;
        constexpr index_t kSlotsPerOutPair =
            kOutPairs < Plan::kPairs ? Plan::kPairs / kOutPairs : 1;
        constexpr index_t kOutPairsPerSlot =
            kOutPairs > Plan::kPairs ? kOutPairs / Plan::kPairs : 1;
        static_assert(kOutPairs * kSlotsPerOutPair == Plan::kPairs * kOutPairsPerSlot);
        constexpr Plan plan = Plan::Make();
        // SALU, WMMA, VMEM and LDS may cross; VALU and TRANS keep the source order.
        constexpr int32_t kValuFence = LLVMSchedGroupMask::SALU | LLVMSchedGroupMask::MFMA |
                                       LLVMSchedGroupMask::VMEM | LLVMSchedGroupMask::VMEM_READ |
                                       LLVMSchedGroupMask::VMEM_WRITE | LLVMSchedGroupMask::DS |
                                       LLVMSchedGroupMask::DS_READ | LLVMSchedGroupMask::DS_WRITE;

        array<Pair, Plan::kPairs> leaf{};
        array<Pair, Plan::kNodes> node{};
        array<float, Plan::kNumMsb> sums{};

        auto make_delta = [&](auto q) {
            constexpr index_t msb  = decltype(q)::value / Plan::kPairsPerMsb;
            constexpr index_t pair = decltype(q)::value % Plan::kPairsPerMsb;
            const float row_max    = msb < 2 ? exponent_max_m0 : exponent_max_m1;
            const float scale      = BandScale<(msb < 2 ? 0 : 1)>(log2e_scale);
            const float col        = [&]() {
                if constexpr(std::is_same_v<ColScale, NoColumnScale>)
                    return scale;
                else
                    return scale * col_scale[number<pair / Mapping::kPairsPerFragment>{}];
            }();
            leaf(q) = __builtin_elementwise_fma(Mapping::template LoadPair<msb, pair>(score),
                                                Softmax::Broadcast(col),
                                                -Softmax::Broadcast(row_max * scale));
        };
        auto exp_element = [&](auto q, auto e) {
            leaf(q)[decltype(e)::value] = Softmax::Exp2(leaf[q][decltype(e)::value]);
            if constexpr(decltype(e)::value == 1)
                Mapping::template StorePair<decltype(q)::value / Plan::kPairsPerMsb,
                                            decltype(q)::value % Plan::kPairsPerMsb>(score,
                                                                                     leaf[q]);
        };
        auto rescale_pair = [&](auto q) {
            constexpr index_t ordinal = decltype(q)::value / kOutPairsPerFragment;
            constexpr index_t e       = decltype(q)::value % kOutPairsPerFragment * 2;
            const float scale =
                ordinal / OutputFragments::kNumN < 2 ? output_scale_m0 : output_scale_m1;
            auto& fragment  = output.at(number<ordinal>{});
            const Pair v    = Pair{fragment[e], fragment[e + 1]} * Softmax::Broadcast(scale);
            fragment[e]     = v[0];
            fragment[e + 1] = v[1];
        };
        auto add_node = [&](auto n) {
            constexpr index_t msb  = decltype(n)::value / Plan::kNodesPerMsb;
            constexpr index_t k    = decltype(n)::value % Plan::kNodesPerMsb;
            constexpr index_t a    = Plan::InputA(k);
            constexpr index_t base = msb * Plan::kNodesPerMsb;
            if constexpr(k == Plan::kFoldNode)
                sums(number<msb>{}) =
                    Softmax::HorizontalAdd(node[number<base + Plan::kRootNode>{}]);
            else if constexpr(Plan::InputIsLeaf(k))
                node(n) = Softmax::AddPair(leaf[number<msb * Plan::kPairsPerMsb + a>{}],
                                           leaf[number<msb * Plan::kPairsPerMsb + a + 1>{}]);
            else
                node(n) = Softmax::AddPair(node[number<base + a>{}], node[number<base + a + 1>{}]);
        };

        output_scale_m0 = Softmax::Exp2(delta_m0);
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<0>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        output_scale_m1 = Softmax::Exp2(delta_m1);
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<1>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        static_for<0, Plan::kPairs, 1>{}([&](auto q) {
            exp_element(q, number<0>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value + 2 < Plan::kPairs)
            {
                make_delta(number<decltype(q)::value + 2>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
            exp_element(q, number<1>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value % kSlotsPerOutPair == 0)
            {
                static_for<0, kOutPairsPerSlot, 1>{}([&](auto j) {
                    rescale_pair(number<decltype(q)::value / kSlotsPerOutPair * kOutPairsPerSlot +
                                        decltype(j)::value>{});
                });
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
            if constexpr(SumRows && plan.slot_node[decltype(q)::value] >= 0)
            {
                add_node(number<plan.slot_node[decltype(q)::value]>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
        });
        if constexpr(SumRows)
            static_for<0, plan.num_tail, 1>{}([&](auto i) { add_node(number<plan.tail[i]>{}); });
        return sums;
    }

    // SumRows false leaves row_sum to the caller, which sums the P it actually consumes.
    template <typename Problem,
              bool SumRows = true,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               LogScale log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1,
                                                               Fragments& output,
                                                               const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;

        const auto validated_max = [](float value) {
            if constexpr(kValidateMax)
                return value <= -numeric<float>::infinity() ? 0.0f : value;
            else
                return value;
        };

        const float row_max_m0                  = row_max.get_thread_buffer()[number<0>{}];
        const float row_max_m1                  = row_max.get_thread_buffer()[number<1>{}];
        [[maybe_unused]] const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
        [[maybe_unused]] const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

        const auto sums = RunInterleavedLocalSums<SumRows>(score,
                                                           output,
                                                           validated_max(row_max_m0),
                                                           validated_max(row_max_m1),
                                                           delta_m0,
                                                           delta_m1,
                                                           log2e_scale,
                                                           output_scale_m0,
                                                           output_scale_m1,
                                                           col_scale);

        if constexpr(SumRows)
        {
            row_sum.get_thread_buffer()[number<0>{}] =
                __builtin_fmaf(output_scale_m0,
                               old_sum_m0,
                               Softmax::MergeRowSum(sums[number<0>{}], sums[number<1>{}]));
            row_sum.get_thread_buffer()[number<1>{}] =
                __builtin_fmaf(output_scale_m1,
                               old_sum_m1,
                               Softmax::MergeRowSum(sums[number<2>{}], sums[number<3>{}]));
        }
    }
};

} // namespace ck_tile

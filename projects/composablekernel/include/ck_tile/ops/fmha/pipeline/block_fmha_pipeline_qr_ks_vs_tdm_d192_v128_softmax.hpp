// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_state.hpp"

namespace ck_tile {

struct FmhaD192SplitSoftmax
{
    using Mapping = FmhaD192ScoreFragmentMapping;
    using Pair    = typename Mapping::Pair;

    static constexpr index_t kPart0OperationCount       = 22;
    static constexpr index_t kPart1OperationCount       = 8;
    static constexpr index_t kPart2OperationCount       = 89;
    static constexpr index_t kCurrentPart2OperationEnd  = 32;
    static constexpr index_t kPreviousPart2OperationBeg = kCurrentPart2OperationEnd;

    struct Part0State
    {
        float local_max;
        float old_max_log2e;
    };

    struct Part1State
    {
        float row_max;
        float delta;
    };

    CK_TILE_DEVICE static float Max3(float x, float y, float z)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        float result;
        asm volatile("v_max3_num_f32 %0, %1, %2, %3" : "=v"(result) : "v"(x), "v"(y), "v"(z));
        return result;
#else
        return max(max(x, y), z);
#endif
    }

    CK_TILE_DEVICE static float PermuteLaneX16(float value)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        const auto bits = bit_cast<std::uint32_t>(value);
        return bit_cast<float>(
            __builtin_amdgcn_permlanex16(bits, bits, 0x76543210, 0xfedcba98, false, false));
#else
        return value;
#endif
    }

    template <index_t Msb, typename ScoreTensor>
    CK_TILE_DEVICE static Part0State
    RunPart0(const ScoreTensor& score, float old_max, float log2e_scale)
    {
        array<float, Mapping::kNumSu> group_max{};

        static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
            constexpr index_t pair_base = su * Mapping::kPairsPerFragment;
            const auto pair0            = Mapping::LoadPair<Msb, pair_base + 0>(score);
            const auto pair1            = Mapping::LoadPair<Msb, pair_base + 1>(score);
            const auto pair2            = Mapping::LoadPair<Msb, pair_base + 2>(score);
            const auto pair3            = Mapping::LoadPair<Msb, pair_base + 3>(score);

            float value   = Max3(pair0[0], pair0[1], pair1[0]);
            value         = Max3(pair1[1], pair2[0], value);
            value         = Max3(pair2[1], pair3[0], value);
            group_max(su) = Max3(pair3[1], value, pair0[0]);
        });

        float lane_max =
            Max3(group_max[number<0>{}], group_max[number<1>{}], group_max[number<2>{}]);
        lane_max = Max3(lane_max, group_max[number<3>{}], group_max[number<1>{}]);
        lane_max = Max3(lane_max, PermuteLaneX16(lane_max), old_max);

        return Part0State{lane_max, old_max * log2e_scale};
    }

    template <bool ValidateMax = false>
    CK_TILE_DEVICE static Part1State
    RunPart1(float lhs_local_max, float rhs_local_max, float old_max_log2e, float log2e_scale)
    {
        const float row_max = max(lhs_local_max, rhs_local_max);
        const float exponent_max =
            ValidateMax && row_max == -numeric<float>::infinity() ? 0.0f : row_max;
        return Part1State{row_max, __builtin_fmaf(-exponent_max, log2e_scale, old_max_log2e)};
    }

    CK_TILE_DEVICE static Pair Broadcast(float value) { return Pair{value, value}; }

    CK_TILE_DEVICE static float Exp2(float value)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        return __builtin_amdgcn_exp2f(value);
#else
        return exp2(value);
#endif
    }

    CK_TILE_DEVICE static Pair MakeExpPair(Pair score, float row_max, float log2e_scale)
    {
        const Pair delta = __builtin_elementwise_fma(
            score, Broadcast(log2e_scale), -Broadcast(row_max * log2e_scale));
        return Pair{Exp2(delta[0]), Exp2(delta[1])};
    }

    CK_TILE_DEVICE static Pair AddPair(Pair x, Pair y) { return x + y; }

    CK_TILE_DEVICE static Pair MulPair(Pair lhs, Pair rhs) { return lhs * rhs; }

    CK_TILE_DEVICE static float HorizontalAdd(Pair value) { return value[0] + value[1]; }

    template <index_t Msb, typename ScoreTensor>
    CK_TILE_DEVICE static float
    RunPart2LocalSum(ScoreTensor& score, float exponent_max, float log2e_scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        array<Pair, Mapping::kPairsPerMsb> exp_pairs{};

        static_for<0, Mapping::kPairsPerMsb, 1>{}([&](auto pair) {
            exp_pairs(pair) =
                MakeExpPair(Mapping::LoadPair<Msb, pair>(score), exponent_max, log2e_scale);
            Mapping::StorePair<Msb, pair>(score, exp_pairs[pair]);
        });

        array<Pair, Mapping::kPairsPerMsb / 2> sum_l0{};
        static_for<0, Mapping::kPairsPerMsb / 2, 1>{}(
            [&](auto i) { sum_l0(i) = AddPair(exp_pairs[2 * i], exp_pairs[2 * i + 1]); });

        array<Pair, Mapping::kPairsPerMsb / 4> sum_l1{};
        static_for<0, Mapping::kPairsPerMsb / 4, 1>{}(
            [&](auto i) { sum_l1(i) = AddPair(sum_l0[2 * i], sum_l0[2 * i + 1]); });

        array<Pair, Mapping::kPairsPerMsb / 8> sum_l2{};
        static_for<0, Mapping::kPairsPerMsb / 8, 1>{}(
            [&](auto i) { sum_l2(i) = AddPair(sum_l1[2 * i], sum_l1[2 * i + 1]); });

        const Pair sum = AddPair(sum_l2[number<0>{}], sum_l2[number<1>{}]);
        return HorizontalAdd(sum);
    }

    CK_TILE_DEVICE static float MergeRowSum(float lhs, float rhs)
    {
        const float lane_sum = lhs + rhs;
        return lane_sum + PermuteLaneX16(lane_sum);
    }

    CK_TILE_DEVICE static float UpdateRowSum(float old_row_sum, float local_row_sum, float delta)
    {
        return __builtin_fmaf(Exp2(delta), old_row_sum, local_row_sum);
    }

    template <index_t Msb, typename OutputTensor>
    CK_TILE_DEVICE static void RescaleOutput(OutputTensor& output, float scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_for<0, Mapping::kNumSu, 1>{}([&](auto output_tile) {
            RescaleOutputTile<Msb, decltype(output_tile)::value>(output, scale);
        });
    }

    template <index_t Msb, index_t OutputTile, typename OutputTensor>
    CK_TILE_DEVICE static void RescaleOutputTile(OutputTensor& output, float scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(OutputTile >= 0 && OutputTile < Mapping::kNumSu);
        constexpr index_t pair_begin = OutputTile * Mapping::kPairsPerFragment;
        const Pair scale_pair        = Broadcast(scale);
        static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
            constexpr index_t pair_index = pair_begin + decltype(pair)::value;
            Mapping::StorePair<Msb, pair_index>(
                output, MulPair(Mapping::LoadPair<Msb, pair_index>(output), scale_pair));
        });
    }

    CK_TILE_DEVICE static bf16x2_t ConvertPairToBf16(Pair value)
    {
        return cvt_pk_bf16_f32(value[0], value[1]);
    }

    template <index_t Msb, index_t Op>
    CK_TILE_DEVICE static void
    EmitPart0Op(FmhaD192SoftmaxState& state, FmhaD192SoftmaxClosureState& closure, float scale_log2)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(Op >= 0 && Op < kPart0OperationCount);
        auto& local = closure.msb[number<Msb>{}];

        if constexpr(Op < 4)
        {
            constexpr index_t pair_base   = Op * Mapping::kPairsPerFragment;
            const auto pair0              = state.score.template Get<Msb, pair_base>();
            const auto pair1              = state.score.template Get<Msb, pair_base + 1>();
            local.group_max[number<Op>{}] = Max3(pair0[0], pair0[1], pair1[0]);
        }
        else if constexpr(Op < 8)
        {
            constexpr index_t group     = Op - 4;
            constexpr index_t pair_base = group * Mapping::kPairsPerFragment;
            const auto pair1            = state.score.template Get<Msb, pair_base + 1>();
            const auto pair2            = state.score.template Get<Msb, pair_base + 2>();
            local.group_max[number<group>{}] =
                Max3(pair1[1], pair2[0], local.group_max[number<group>{}]);
        }
        else if constexpr(Op < 12)
        {
            constexpr index_t group     = Op - 8;
            constexpr index_t pair_base = group * Mapping::kPairsPerFragment;
            const auto pair2            = state.score.template Get<Msb, pair_base + 2>();
            const auto pair3            = state.score.template Get<Msb, pair_base + 3>();
            local.group_max[number<group>{}] =
                Max3(pair2[1], pair3[0], local.group_max[number<group>{}]);
        }
        else if constexpr(Op < 16)
        {
            constexpr index_t group     = Op - 12;
            constexpr index_t pair_base = group * Mapping::kPairsPerFragment;
            const auto pair0            = state.score.template Get<Msb, pair_base>();
            const auto pair3            = state.score.template Get<Msb, pair_base + 3>();
            local.group_max[number<group>{}] =
                Max3(pair3[1], local.group_max[number<group>{}], pair0[0]);
        }
        else if constexpr(Op == 16)
        {
            local.group_max[number<0>{}] = Max3(local.group_max[number<0>{}],
                                                local.group_max[number<1>{}],
                                                local.group_max[number<2>{}]);
        }
        else if constexpr(Op == 17)
        {
            local.group_max[number<0>{}] = Max3(local.group_max[number<0>{}],
                                                local.group_max[number<3>{}],
                                                local.group_max[number<1>{}]);
        }
        else if constexpr(Op == 18)
        {
            local.permute_input = local.group_max[number<0>{}] + 0.0f;
        }
        else if constexpr(Op == 19)
        {
            local.permuted_max = PermuteLaneX16(local.permute_input);
        }
        else if constexpr(Op == 20)
        {
            local.pre_max_log2e_scl = state.old_max[number<Msb>{}] * scale_log2;
        }
        else
        {
            state.local_max[number<Msb>{}] = Max3(
                local.group_max[number<0>{}], local.permuted_max, state.old_max[number<Msb>{}]);
        }
    }

    template <bool ValidateMax, index_t Op>
    CK_TILE_DEVICE static void
    EmitPart1Op(FmhaD192SoftmaxState& state, FmhaD192SoftmaxClosureState& closure, float scale_log2)
    {
        static_assert(Op >= 0 && Op < kPart1OperationCount);

        if constexpr(Op < 2)
        {
            constexpr index_t even_msb = 2 * Op;
            constexpr index_t odd_msb  = even_msb + 1;
            const float logical_max =
                max(state.local_max[number<even_msb>{}], state.local_max[number<odd_msb>{}]);
            state.local_max[number<even_msb>{}]       = logical_max;
            state.logical_row_max[number<even_msb>{}] = logical_max;
            closure.msb[number<even_msb>{}].exponent_max =
                ValidateMax && __builtin_isinf_sign(logical_max) < 0 ? 0.0f : logical_max;
        }
        else if constexpr(Op < 4)
        {
            constexpr index_t odd_msb                = 2 * (Op - 2) + 1;
            constexpr index_t even_msb               = odd_msb - 1;
            state.local_max[number<odd_msb>{}]       = state.local_max[number<even_msb>{}];
            state.logical_row_max[number<odd_msb>{}] = state.logical_row_max[number<even_msb>{}];
            closure.msb[number<odd_msb>{}].exponent_max =
                closure.msb[number<even_msb>{}].exponent_max;
        }
        else
        {
            constexpr index_t kDeltaOrder[4] = {0, 2, 1, 3};
            constexpr index_t msb            = kDeltaOrder[Op - 4];
            state.delta[number<msb>{}] =
                __builtin_fmaf(-closure.msb[number<msb>{}].exponent_max,
                               scale_log2,
                               closure.msb[number<msb>{}].pre_max_log2e_scl);
        }
    }

    template <index_t Msb, index_t Op, typename ProbabilityFragments>
    CK_TILE_DEVICE static void EmitPart2Op(FmhaD192SoftmaxState& state,
                                           ProbabilityFragments& probability,
                                           FmhaD192SoftmaxClosureState& closure,
                                           float scale_log2)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(Op >= 0 && Op < kPart2OperationCount);
        auto& local = closure.msb[number<Msb>{}];

        if constexpr(Op == 0)
        {
            state.old_max[number<Msb>{}] = state.logical_row_max[number<Msb>{}];
            local.scale_log2_pair        = Broadcast(scale_log2);
        }
        else if constexpr(Op == 1)
        {
            local.scaled_max_0 = local.exponent_max * scale_log2;
        }
        else if constexpr(Op == 2)
        {
            state.exp_delta[number<Msb>{}] = Exp2(state.delta[number<Msb>{}]);
        }
        else if constexpr(Op == 3)
        {
            local.scaled_max_1 = local.exponent_max * scale_log2;
        }
        else if constexpr(Op == 4)
        {
            local.scaled_max_scalar = local.exponent_max * scale_log2;
        }
        else if constexpr(Op == 5)
        {
            local.scaled_max_pair = Broadcast(local.scaled_max_scalar);
        }
        else if constexpr(Op == 6)
        {
            local.exp_delta_copy = state.exp_delta[number<Msb>{}];
        }
        else if constexpr(Op == 7)
        {
            state.row_sum[number<Msb>{}] *= state.exp_delta[number<Msb>{}];
        }
        else if constexpr(Op < 24)
        {
            constexpr index_t pair = Op - 8;
            state.score.template Get<Msb, pair>() =
                __builtin_elementwise_fma(state.score.template Get<Msb, pair>(),
                                          local.scale_log2_pair,
                                          -local.scaled_max_pair);
        }
        else if constexpr(Op < 56)
        {
            constexpr index_t scalar              = Op - 24;
            constexpr index_t pair                = scalar / 2;
            constexpr index_t element             = scalar % 2;
            auto value                            = state.score.template Get<Msb, pair>();
            value[element]                        = Exp2(value[element]);
            state.score.template Get<Msb, pair>() = value;
        }
        else if constexpr(Op < 72)
        {
            constexpr index_t pair = Op - 56;
            probability.template Get<Msb, pair>() =
                ConvertPairToBf16(state.score.template Get<Msb, pair>());
        }
        else if constexpr(Op < 80)
        {
            constexpr index_t pair       = Op - 72;
            local.sum_l0[number<pair>{}] = AddPair(state.score.template Get<Msb, 2 * pair>(),
                                                   state.score.template Get<Msb, 2 * pair + 1>());
        }
        else if constexpr(Op < 84)
        {
            constexpr index_t pair = Op - 80;
            local.sum_l1[number<pair>{}] =
                AddPair(local.sum_l0[number<2 * pair>{}], local.sum_l0[number<2 * pair + 1>{}]);
        }
        else if constexpr(Op < 86)
        {
            constexpr index_t pair = Op - 84;
            local.sum_l2[number<pair>{}] =
                AddPair(local.sum_l1[number<2 * pair>{}], local.sum_l1[number<2 * pair + 1>{}]);
        }
        else if constexpr(Op == 86)
        {
            local.final_pair = AddPair(local.sum_l2[number<0>{}], local.sum_l2[number<1>{}]);
        }
        else if constexpr(Op == 87)
        {
            local.final_sum = HorizontalAdd(local.final_pair);
        }
        else
        {
            state.row_sum[number<Msb>{}] += local.final_sum;
        }
    }

    template <index_t Msb, index_t Op>
    CK_TILE_DEVICE static void EmitCurrentPart2Op(FmhaD192SoftmaxState& state,
                                                  FmhaD192SoftmaxClosureState& closure,
                                                  float scale_log2)
    {
        static_assert(Op < kCurrentPart2OperationEnd);
        struct NoProbabilityFragments
        {
        } unused_probability;
        EmitPart2Op<Msb, Op>(state, unused_probability, closure, scale_log2);
    }

    template <index_t Msb, index_t Op>
    CK_TILE_DEVICE static void EmitPreviousPart2Op(FmhaD192SoftmaxState& state,
                                                   FmhaD192ProbabilityFragments& probability,
                                                   FmhaD192SoftmaxClosureState& closure,
                                                   float scale_log2)
    {
        static_assert(Op >= kPreviousPart2OperationBeg && Op < kPart2OperationCount);
        EmitPart2Op<Msb, Op>(state, probability, closure, scale_log2);
    }
};

struct FmhaD192CrossTilePrologue
{
    using Mapping = FmhaD192ScoreFragmentMapping;
    using Softmax = FmhaD192SplitSoftmax;

    static constexpr index_t kBackEdgeOperation = Softmax::kCurrentPart2OperationEnd - 1;

    template <bool ValidateMax,
              bool HasNextTile,
              typename OneTileConsumer,
              typename MultiTileConsumer>
    CK_TILE_DEVICE static void Run(FmhaD192SoftmaxState& state,
                                   FmhaD192SoftmaxClosureState& closure,
                                   float scale_log2,
                                   OneTileConsumer& consume_one_tile,
                                   MultiTileConsumer& consume_multi_tile)
    {
        static_for<0, Softmax::kPart0OperationCount, 1>{}([&](auto op) {
            static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
                Softmax::template EmitPart0Op<decltype(msb)::value, decltype(op)::value>(
                    state, closure, scale_log2);
            });
        });
        static_for<0, Softmax::kPart1OperationCount, 1>{}([&](auto op) {
            Softmax::template EmitPart1Op<ValidateMax, decltype(op)::value>(
                state, closure, scale_log2);
        });
        static_for<0, Softmax::kCurrentPart2OperationEnd, 1>{}([&](auto op) {
            static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
                Softmax::template EmitCurrentPart2Op<decltype(msb)::value, decltype(op)::value>(
                    state, closure, scale_log2);
            });
        });

        FmhaD192CrossTileBackEdge::template Transfer<HasNextTile>(
            state, closure, consume_one_tile, consume_multi_tile);
    }
};

static_assert(FmhaD192CrossTilePrologue::kBackEdgeOperation == 31);

static_assert(FmhaD192SplitSoftmax::kCurrentPart2OperationEnd == 8 + 16 + 8);
static_assert(FmhaD192SplitSoftmax::kPart2OperationCount == 8 + 16 + 32 + 16 + 8 + 4 + 2 + 3);

struct FmhaD192SoftmaxTokenContract
{
    static constexpr index_t kPart0Merge01       = 16;
    static constexpr index_t kPart0Merge3        = 17;
    static constexpr index_t kPart0PermuteInput  = 18;
    static constexpr index_t kPart0Permute       = 19;
    static constexpr index_t kPart0ScalePriorMax = 20;
    static constexpr index_t kPart0FinalMax      = 21;

    static constexpr index_t kPart2PackedFmaBegin = 8;
    static constexpr index_t kPart2ExpBegin       = 24;
    static constexpr index_t kPart2ConvertBegin   = 56;
    static constexpr index_t kPart2SumL0Begin     = 72;
    static constexpr index_t kPart2SumL1Begin     = 80;
    static constexpr index_t kPart2SumL2Begin     = 84;
    static constexpr index_t kPart2FinalPair      = 86;
    static constexpr index_t kPart2FinalSum       = 87;
    static constexpr index_t kPart2Accumulate     = 88;

    CK_TILE_HOST_DEVICE static constexpr bool Validate()
    {
        bool valid = kPart0Merge01 == 16 && kPart0Merge3 > kPart0Merge01 &&
                     kPart0PermuteInput > kPart0Merge3 && kPart0Permute > kPart0PermuteInput &&
                     kPart0ScalePriorMax < kPart0FinalMax &&
                     kPart0FinalMax + 1 == FmhaD192SplitSoftmax::kPart0OperationCount;

        valid &= kPart2PackedFmaBegin == 8 && kPart2ExpBegin == 24 &&
                 FmhaD192SplitSoftmax::kCurrentPart2OperationEnd == kPart2ExpBegin + 8 &&
                 kPart2ConvertBegin == 56 && kPart2SumL0Begin == 72 && kPart2SumL1Begin == 80 &&
                 kPart2SumL2Begin == 84 && kPart2FinalPair == 86 && kPart2FinalSum == 87 &&
                 kPart2Accumulate == 88 &&
                 kPart2Accumulate + 1 == FmhaD192SplitSoftmax::kPart2OperationCount;

        for(index_t pair = 0; pair < FmhaD192ScoreFragmentMapping::kPairsPerMsb; ++pair)
        {
            const index_t packed_fma_producer = kPart2PackedFmaBegin + pair;
            const index_t exp_low_producer    = kPart2ExpBegin + pair * 2;
            const index_t exp_high_producer   = exp_low_producer + 1;
            const index_t p_producer          = kPart2ConvertBegin + pair;
            valid &= packed_fma_producer < exp_low_producer &&
                     exp_low_producer < exp_high_producer && exp_high_producer < p_producer &&
                     p_producer < kPart2SumL0Begin;
        }

        for(index_t i = 0; i < 8; ++i)
        {
            valid &= kPart2SumL0Begin + i < kPart2SumL1Begin + i / 2;
        }
        for(index_t i = 0; i < 4; ++i)
        {
            valid &= kPart2SumL1Begin + i < kPart2SumL2Begin + i / 2;
        }
        return valid && kPart2SumL2Begin + 1 < kPart2FinalPair &&
               kPart2FinalPair < kPart2FinalSum && kPart2FinalSum < kPart2Accumulate;
    }
};

static_assert(FmhaD192SoftmaxTokenContract::Validate());

} // namespace ck_tile

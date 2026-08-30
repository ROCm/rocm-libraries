// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"

namespace {

using Mapping     = ck_tile::FmhaD192ScoreFragmentMapping;
using ScoreState  = ck_tile::FmhaD192SoftmaxState;
using Probability = ck_tile::FmhaD192ProbabilityFragments;
using Closure     = ck_tile::FmhaD192SoftmaxClosureState;
using Softmax     = ck_tile::FmhaD192SplitSoftmax;
using Prologue    = ck_tile::FmhaD192CrossTilePrologue;

enum class InputPattern
{
    Finite,
    MixedFiniteAndMasked,
    AllMaskedEmpty,
    AllMaskedFiniteHistory,
};

using D192BlockShape = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, 192>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

struct D192Problem
{
    struct AttentionVariant
    {
    };
    struct FmhaMask
    {
    };

    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = std::uint8_t;
    using LSEDataType           = float;
    using PDataType             = ck_tile::bf16_t;
    using OaccDataType          = float;
    using ODataType             = ck_tile::bf16_t;
    using BlockFmhaShape        = D192BlockShape;

    static constexpr ck_tile::index_t kBlockSize                   = 128;
    [[maybe_unused]] static constexpr ck_tile::index_t kBlockPerCu = 1;
    [[maybe_unused]] static constexpr bool kIsGroupMode            = false;
    [[maybe_unused]] static constexpr bool kPadSeqLenQ             = false;
    [[maybe_unused]] static constexpr bool kPadSeqLenK             = false;
    [[maybe_unused]] static constexpr bool kPadHeadDimQ            = false;
    [[maybe_unused]] static constexpr bool kPadHeadDimV            = false;
    static constexpr bool kHasLogitsSoftCap                        = false;
    static constexpr bool kHasDropout                              = false;
    [[maybe_unused]] static constexpr bool kStoreLSE               = false;
    [[maybe_unused]] static constexpr bool kHasSink                = false;
    static constexpr auto BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
};

using Policy      = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;
using Pipeline    = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128<D192Problem>;
using PvBlockGemm = ck_tile::remove_cvref_t<decltype(Policy::GetPVBlockGemm<D192Problem>())>;
using D192GemmProblem =
    ck_tile::BlockGemmProblem<ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              float,
                              128,
                              ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 32>,
                                                     ck_tile::sequence<4, 1, 1>,
                                                     ck_tile::sequence<16, 16, 32>>>;
using D192WarpGemm   = ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true>;
using D192GemmPolicy = ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<ck_tile::bf16_t,
                                                                    ck_tile::bf16_t,
                                                                    float,
                                                                    ck_tile::sequence<4, 1, 1>,
                                                                    D192WarpGemm,
                                                                    ck_tile::GemmLoopOrder::MNK>;
using D192BlockGemm  = ck_tile::BlockGemmARegBRegCRegV2<D192GemmProblem, D192GemmPolicy>;
using ScoreTile      = decltype(D192BlockGemm::MakeCBlockTile());

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
using ProductionQkBlockGemm =
    ck_tile::remove_cvref_t<decltype(Policy::GetQKBlockGemm<D192Problem>())>;
static_assert(std::is_same_v<D192BlockGemm, ProductionQkBlockGemm>);
#endif

constexpr ck_tile::index_t kThreads              = 128;
constexpr ck_tile::index_t kInputScoreOffset     = 0;
constexpr ck_tile::index_t kInputOldMaxOffset    = Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kInputRowSumOffset    = kInputOldMaxOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kInputScaleOffset     = kInputRowSumOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kInputStride          = kInputScaleOffset + 1;
constexpr ck_tile::index_t kLogicalMaxOffset     = 0;
constexpr ck_tile::index_t kExponentMaxOffset    = kLogicalMaxOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kDeltaOffset          = kExponentMaxOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kExpDeltaOffset       = kDeltaOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kOldMaxOffset         = kExpDeltaOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kPartialRowSumOffset  = kOldMaxOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kPartialScoreOffset   = kPartialRowSumOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kFinalRowSumOffset    = kPartialScoreOffset + Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kFinalScoreOffset     = kFinalRowSumOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kProbabilityOffset    = kFinalScoreOffset + Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kPvOperandOffset      = kProbabilityOffset + Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kPvOperandCount       = Mapping::kNumSu * 2 * 16;
constexpr ck_tile::index_t kCkPvOperandCount     = Mapping::kNumSu * 2 * 16;
constexpr ck_tile::index_t kCkPvOperandOffset    = kPvOperandOffset + kPvOperandCount;
constexpr ck_tile::index_t kMergedRowSumOffset   = kCkPvOperandOffset + kCkPvOperandCount;
constexpr ck_tile::index_t kScaleOutputOffset    = kMergedRowSumOffset + 2;
constexpr ck_tile::index_t kLocalMaxOutputOffset = kScaleOutputOffset + 2;
constexpr ck_tile::index_t kBackEdgePhaseOffset  = kLocalMaxOutputOffset + Mapping::kNumMsb;
constexpr ck_tile::index_t kBackEdgeAliasOffset  = kBackEdgePhaseOffset + 1;
constexpr ck_tile::index_t kOutputStride         = kBackEdgeAliasOffset + 1;

static_assert(Softmax::kPart0OperationCount == 22);
static_assert(Softmax::kPart1OperationCount == 8);
static_assert(Softmax::kCurrentPart2OperationEnd == 32);
static_assert(Softmax::kPart2OperationCount == 89);
static_assert(Prologue::kBackEdgeOperation == 31);
static_assert(sizeof(ck_tile::FmhaD192ScoreFragments) == 128 * sizeof(float));
static_assert(sizeof(Probability) == 128 * sizeof(ck_tile::bf16_t));

template <InputPattern Pattern, bool HasNextTile>
__global__ __launch_bounds__(kThreads, 1) void RunCrossTilePrimitives(const float* input,
                                                                      float* output)
{
    const auto* lane_input = input + threadIdx.x * kInputStride;
    auto* lane_output      = output + threadIdx.x * kOutputStride;
    ScoreState state{};
    Probability probability{};
    Closure closure{};

    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}(
        [&](auto indices) {
            constexpr ck_tile::index_t msb        = indices[ck_tile::number<0>{}];
            constexpr ck_tile::index_t pair       = indices[ck_tile::number<1>{}];
            constexpr ck_tile::index_t scalar     = msb * Mapping::kElementsPerMsb + pair * 2;
            state.score.template Get<msb, pair>() = ck_tile::fp32x2_t{
                lane_input[kInputScoreOffset + scalar], lane_input[kInputScoreOffset + scalar + 1]};
        });

    ck_tile::static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
        state.old_max[msb] = lane_input[kInputOldMaxOffset + decltype(msb)::value];
        state.row_sum[msb] = lane_input[kInputRowSumOffset + decltype(msb)::value];
    });
    const auto scales      = ck_tile::FmhaD192Scale::FromKernelScale(lane_input[kInputScaleOffset]);
    const float scale_log2 = scales.scale_log2;

    if constexpr(Pattern == InputPattern::AllMaskedFiniteHistory)
    {
        auto ignore_one_tile   = [](ck_tile::FmhaD192OneTileFlush, ScoreState&, Closure&) {};
        auto ignore_multi_tile = [](ck_tile::FmhaD192MultiTileFirstSteady, ScoreState&, Closure&) {
        };
        Prologue::template Run<true, true>(
            state, closure, scale_log2, ignore_one_tile, ignore_multi_tile);
        ck_tile::static_for<Softmax::kPreviousPart2OperationBeg,
                            Softmax::kPart2OperationCount,
                            1>{}([&](auto op) {
            ck_tile::static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
                Softmax::template EmitPreviousPart2Op<decltype(msb)::value, decltype(op)::value>(
                    state, probability, closure, scale_log2);
            });
        });
        ck_tile::static_ford<ck_tile::sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}(
            [&](auto indices) {
                constexpr ck_tile::index_t msb        = indices[ck_tile::number<0>{}];
                constexpr ck_tile::index_t pair       = indices[ck_tile::number<1>{}];
                state.score.template Get<msb, pair>() = ck_tile::fp32x2_t{
                    -ck_tile::numeric<float>::infinity(), -ck_tile::numeric<float>::infinity()};
            });
        closure = Closure{};
    }

    constexpr bool kValidateMax = Pattern != InputPattern::Finite;
    float back_edge_phase       = -1.0f;
    float back_edge_alias       = 0.0f;
    auto consume_one_tile       = [&](ck_tile::FmhaD192OneTileFlush,
                                ScoreState& transferred_state,
                                Closure& transferred_closure) {
        back_edge_phase = 0.0f;
        back_edge_alias =
            &transferred_state == &state && &transferred_closure == &closure ? 1.0f : 0.0f;
    };
    auto consume_multi_tile = [&](ck_tile::FmhaD192MultiTileFirstSteady,
                                  ScoreState& transferred_state,
                                  Closure& transferred_closure) {
        back_edge_phase = 1.0f;
        back_edge_alias =
            &transferred_state == &state && &transferred_closure == &closure ? 1.0f : 0.0f;
    };
    Prologue::template Run<kValidateMax, HasNextTile>(
        state, closure, scale_log2, consume_one_tile, consume_multi_tile);

    lane_output[kBackEdgePhaseOffset] = back_edge_phase;
    lane_output[kBackEdgeAliasOffset] = back_edge_alias;

    ck_tile::static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
        constexpr ck_tile::index_t m           = decltype(msb)::value;
        lane_output[kLogicalMaxOffset + m]     = state.logical_row_max[msb];
        lane_output[kExponentMaxOffset + m]    = closure.msb[msb].exponent_max;
        lane_output[kDeltaOffset + m]          = state.delta[msb];
        lane_output[kExpDeltaOffset + m]       = state.exp_delta[msb];
        lane_output[kOldMaxOffset + m]         = state.old_max[msb];
        lane_output[kPartialRowSumOffset + m]  = state.row_sum[msb];
        lane_output[kLocalMaxOutputOffset + m] = state.local_max[msb];
    });
    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}(
        [&](auto indices) {
            constexpr ck_tile::index_t msb            = indices[ck_tile::number<0>{}];
            constexpr ck_tile::index_t pair           = indices[ck_tile::number<1>{}];
            const auto value                          = state.score.template Get<msb, pair>();
            constexpr ck_tile::index_t offset         = msb * Mapping::kElementsPerMsb + pair * 2;
            lane_output[kPartialScoreOffset + offset] = value[0];
            lane_output[kPartialScoreOffset + offset + 1] = value[1];
        });

    ck_tile::static_for<Softmax::kPreviousPart2OperationBeg, Softmax::kPart2OperationCount, 1>{}(
        [&](auto op) {
            ck_tile::static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
                Softmax::template EmitPreviousPart2Op<decltype(msb)::value, decltype(op)::value>(
                    state, probability, closure, scale_log2);
            });
        });

    ck_tile::static_for<0, Mapping::kNumMsb, 1>{}([&](auto msb) {
        lane_output[kFinalRowSumOffset + decltype(msb)::value] = state.row_sum[msb];
    });
    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}(
        [&](auto indices) {
            constexpr ck_tile::index_t msb              = indices[ck_tile::number<0>{}];
            constexpr ck_tile::index_t pair             = indices[ck_tile::number<1>{}];
            constexpr ck_tile::index_t offset           = msb * Mapping::kElementsPerMsb + pair * 2;
            const auto score                            = state.score.template Get<msb, pair>();
            const auto packed                           = probability.template Get<msb, pair>();
            lane_output[kFinalScoreOffset + offset]     = score[0];
            lane_output[kFinalScoreOffset + offset + 1] = score[1];
            lane_output[kProbabilityOffset + offset]    = ck_tile::type_convert<float>(packed[0]);
            lane_output[kProbabilityOffset + offset + 1] = ck_tile::type_convert<float>(packed[1]);
        });

    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumSu, 2>>{}([&](auto indices) {
        constexpr ck_tile::index_t su     = indices[ck_tile::number<0>{}];
        constexpr ck_tile::index_t m_half = indices[ck_tile::number<1>{}];
        constexpr ck_tile::index_t base   = (su * 2 + m_half) * 16;
        const auto operand                = probability.template MakePvOperand<su, m_half>();
        ck_tile::static_ford<ck_tile::sequence<2 * Mapping::kPairsPerFragment, 2>>{}(
            [&](auto pair_element) {
                constexpr ck_tile::index_t pair    = pair_element[ck_tile::number<0>{}];
                constexpr ck_tile::index_t element = pair_element[ck_tile::number<1>{}];
                lane_output[kPvOperandOffset + base + pair * 2 + element] =
                    ck_tile::type_convert<float>(operand[ck_tile::number<pair>{}][element]);
            });
    });

    auto score_tensor = ScoreTile{};
    state.score.Store(score_tensor);
    const auto p_tile = Pipeline::template MakePForGemm1<PvBlockGemm>(score_tensor);
    using WarpGemm    = typename PvBlockGemm::WarpGemm;
    using AWarpDstr   = typename WarpGemm::AWarpDstr;
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    constexpr auto a_warp_y_lengths =
        ck_tile::to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto a_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumSu, 2>>{}([&](auto indices) {
        constexpr ck_tile::index_t su     = indices[ck_tile::number<0>{}];
        constexpr ck_tile::index_t m_half = indices[ck_tile::number<1>{}];
        constexpr ck_tile::index_t base   = (su * 2 + m_half) * 16;
        const auto p_stage                = ck_tile::get_slice_tile(
            p_tile, ck_tile::sequence<0, su * 32>{}, ck_tile::sequence<128, (su + 1) * 32>{});
        AWarpTensor operand;
        operand.get_thread_buffer() = p_stage.get_y_sliced_thread_data(
            ck_tile::merge_sequences(ck_tile::sequence<0, m_half>{}, a_warp_y_index_zeros),
            ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, a_warp_y_lengths));
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        static_assert(AWarpTensor::get_thread_buffer_size() == 16);
#endif
        ck_tile::static_for<0, AWarpTensor::get_thread_buffer_size(), 1>{}([&](auto i) {
            lane_output[kCkPvOperandOffset + base + decltype(i)::value] =
                ck_tile::type_convert<float>(operand.get_thread_buffer()[i]);
        });
    });
    lane_output[kMergedRowSumOffset]     = Softmax::MergeRowSum(state.row_sum[ck_tile::number<0>{}],
                                                            state.row_sum[ck_tile::number<1>{}]);
    lane_output[kMergedRowSumOffset + 1] = Softmax::MergeRowSum(
        state.row_sum[ck_tile::number<2>{}], state.row_sum[ck_tile::number<3>{}]);
    lane_output[kScaleOutputOffset]     = scales.attention_scale;
    lane_output[kScaleOutputOffset + 1] = scales.scale_log2;
}

void CheckHip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

bool Near(float actual, float expected, float tolerance = 2.0e-5f)
{
    if(std::isinf(actual) || std::isinf(expected))
    {
        return std::isinf(actual) && std::isinf(expected) &&
               std::signbit(actual) == std::signbit(expected);
    }
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= tolerance * std::max(1.0f, std::abs(expected));
}

template <InputPattern Pattern>
constexpr const char* PatternName()
{
    if constexpr(Pattern == InputPattern::Finite)
        return "finite";
    else if constexpr(Pattern == InputPattern::MixedFiniteAndMasked)
        return "mixed_finite_masked";
    else if constexpr(Pattern == InputPattern::AllMaskedEmpty)
        return "masked_all_inf_empty";
    else
        return "masked_all_inf_finite_history";
}

template <InputPattern Pattern, bool HasNextTile>
bool RunCase(float attention_scale)
{
    std::vector<float> input(kThreads * kInputStride);
    std::vector<float> output(kThreads * kOutputStride);

    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        auto* lane         = input.data() + thread * kInputStride;
        const auto lane_id = thread % 32;
        for(ck_tile::index_t msb = 0; msb < Mapping::kNumMsb; ++msb)
        {
            for(ck_tile::index_t scalar = 0; scalar < Mapping::kElementsPerMsb; ++scalar)
            {
                const float finite_value = -10.0f + 2.0f * static_cast<float>(msb) +
                                           0.125f * static_cast<float>(scalar) +
                                           0.01f * static_cast<float>(lane_id);
                if constexpr(Pattern == InputPattern::AllMaskedEmpty)
                {
                    lane[kInputScoreOffset + msb * Mapping::kElementsPerMsb + scalar] =
                        -std::numeric_limits<float>::infinity();
                }
                else if constexpr(Pattern == InputPattern::MixedFiniteAndMasked)
                {
                    lane[kInputScoreOffset + msb * Mapping::kElementsPerMsb + scalar] =
                        (scalar + msb + lane_id) % 5 == 0 ? -std::numeric_limits<float>::infinity()
                                                          : finite_value;
                }
                else
                {
                    lane[kInputScoreOffset + msb * Mapping::kElementsPerMsb + scalar] =
                        finite_value;
                }
            }
            if constexpr(Pattern == InputPattern::AllMaskedEmpty ||
                         Pattern == InputPattern::AllMaskedFiniteHistory)
            {
                lane[kInputOldMaxOffset + msb] = -std::numeric_limits<float>::infinity();
                lane[kInputRowSumOffset + msb] = 0.0f;
            }
            else
            {
                lane[kInputOldMaxOffset + msb] = msb < 2 ? -0.4f : -0.2f;
                lane[kInputRowSumOffset + msb] = 0.25f + 0.1f * msb;
            }
        }
#if CK_TILE_FMHA_FWD_FAST_EXP2
        lane[kInputScaleOffset] = attention_scale * ck_tile::log2e_v<float>;
#else
        lane[kInputScaleOffset] = attention_scale;
#endif
    }

    float* device_input  = nullptr;
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_input, input.size() * sizeof(float)), "hipMalloc input");
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)), "hipMalloc output");
    CheckHip(
        hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy input");
    hipLaunchKernelGGL((RunCrossTilePrimitives<Pattern, HasNextTile>),
                       dim3(1),
                       dim3(kThreads),
                       0,
                       0,
                       device_input,
                       device_output);
    CheckHip(hipGetLastError(), "cross-tile primitive launch");
    CheckHip(hipDeviceSynchronize(), "cross-tile primitive synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy output");
    CheckHip(hipFree(device_input), "hipFree input");
    CheckHip(hipFree(device_output), "hipFree output");

    bool valid             = true;
    bool ck_mapping_valid  = true;
    const float scale_log2 = attention_scale * ck_tile::log2e_v<float>;
    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        const auto* lane_in  = input.data() + thread * kInputStride;
        const auto* lane_out = output.data() + thread * kOutputStride;
        const auto lane      = thread % 32;
        const auto wave      = thread / 32;
        const auto peer      = wave * 32 + (lane ^ 16);
        const auto* peer_in  = input.data() + peer * kInputStride;
        valid &= Near(lane_out[kScaleOutputOffset], attention_scale);
        valid &= Near(lane_out[kScaleOutputOffset + 1], scale_log2);
        valid &= Near(lane_out[kBackEdgePhaseOffset], HasNextTile ? 1.0f : 0.0f);
        valid &= Near(lane_out[kBackEdgeAliasOffset], 1.0f);
        float local_row_sums[4]{};
        for(ck_tile::index_t row = 0; row < 2; ++row)
        {
            float logical_max = lane_in[kInputOldMaxOffset + 2 * row];
            for(ck_tile::index_t msb = 2 * row; msb < 2 * row + 2; ++msb)
            {
                for(ck_tile::index_t scalar = 0; scalar < Mapping::kElementsPerMsb; ++scalar)
                {
                    logical_max = std::max(
                        logical_max,
                        lane_in[kInputScoreOffset + msb * Mapping::kElementsPerMsb + scalar]);
                    logical_max = std::max(
                        logical_max,
                        peer_in[kInputScoreOffset + msb * Mapping::kElementsPerMsb + scalar]);
                }
            }
            constexpr bool kValidateMax = Pattern != InputPattern::Finite;
            const float exponent_max =
                kValidateMax && std::isinf(logical_max) && std::signbit(logical_max) ? 0.0f
                                                                                     : logical_max;

            for(ck_tile::index_t msb = 2 * row; msb < 2 * row + 2; ++msb)
            {
                const float old_max = Pattern == InputPattern::AllMaskedFiniteHistory
                                          ? logical_max
                                          : lane_in[kInputOldMaxOffset + msb];
                const float delta   = std::fma(-exponent_max, scale_log2, old_max * scale_log2);
                const float alpha   = std::exp2(delta);
                float local_sum     = 0.0f;

                if constexpr(Pattern == InputPattern::AllMaskedFiniteHistory)
                {
                    for(ck_tile::index_t scalar = 0; scalar < Mapping::kElementsPerMsb; ++scalar)
                    {
                        const auto offset = msb * Mapping::kElementsPerMsb + scalar;
                        local_sum += std::exp2(std::fma(lane_in[kInputScoreOffset + offset],
                                                        scale_log2,
                                                        -exponent_max * scale_log2));
                    }
                }

                const float old_row_sum = Pattern == InputPattern::AllMaskedFiniteHistory
                                              ? local_sum
                                              : lane_in[kInputRowSumOffset + msb];
                float current_local_sum =
                    Pattern == InputPattern::AllMaskedFiniteHistory ? 0.0f : local_sum;

                valid &= Near(lane_out[kLogicalMaxOffset + msb], logical_max);
                valid &= Near(lane_out[kLocalMaxOutputOffset + msb], logical_max);
                valid &= Near(lane_out[kExponentMaxOffset + msb], exponent_max);
                valid &= Near(lane_out[kDeltaOffset + msb], delta);
                valid &= Near(lane_out[kExpDeltaOffset + msb], alpha);
                valid &= Near(lane_out[kOldMaxOffset + msb], logical_max);
                valid &= Near(lane_out[kPartialRowSumOffset + msb], old_row_sum * alpha);

                for(ck_tile::index_t scalar = 0; scalar < Mapping::kElementsPerMsb; ++scalar)
                {
                    const auto offset    = msb * Mapping::kElementsPerMsb + scalar;
                    const float score    = Pattern == InputPattern::AllMaskedFiniteHistory
                                               ? -std::numeric_limits<float>::infinity()
                                               : lane_in[kInputScoreOffset + offset];
                    const float shifted  = std::fma(score, scale_log2, -exponent_max * scale_log2);
                    const float expected = std::exp2(shifted);
                    current_local_sum += expected;
                    valid &= Near(lane_out[kPartialScoreOffset + offset],
                                  scalar < 8 ? expected : shifted);
                    valid &= Near(lane_out[kFinalScoreOffset + offset], expected);
                    valid &= Near(lane_out[kProbabilityOffset + offset], expected, 5.0e-3f);
                }
                valid &= Near(lane_out[kFinalRowSumOffset + msb],
                              old_row_sum * alpha + current_local_sum);
                local_row_sums[msb] = old_row_sum * alpha + current_local_sum;
            }

            const auto* peer_out = output.data() + peer * kOutputStride;
            const float expected_merged_sum =
                local_row_sums[2 * row] + local_row_sums[2 * row + 1] +
                peer_out[kFinalRowSumOffset + 2 * row] + peer_out[kFinalRowSumOffset + 2 * row + 1];
            valid &= Near(lane_out[kMergedRowSumOffset + row], expected_merged_sum);
        }

        for(ck_tile::index_t su = 0; su < Mapping::kNumSu; ++su)
        {
            for(ck_tile::index_t m_half = 0; m_half < 2; ++m_half)
            {
                const auto pv_base = kPvOperandOffset + (su * 2 + m_half) * 16;
                for(ck_tile::index_t i = 0; i < 8; ++i)
                {
                    const auto low_offset =
                        kProbabilityOffset + (2 * m_half) * Mapping::kElementsPerMsb + su * 8 + i;
                    const auto high_offset = low_offset + Mapping::kElementsPerMsb;
                    valid &= Near(lane_out[pv_base + i], lane_out[low_offset]);
                    valid &= Near(lane_out[pv_base + 8 + i], lane_out[high_offset]);
                }
                for(ck_tile::index_t i = 0; i < 16; ++i)
                {
                    ck_mapping_valid &=
                        Near(lane_out[pv_base + i],
                             lane_out[kCkPvOperandOffset + (su * 2 + m_half) * 16 + i]);
                }
            }
        }
    }

    valid &= ck_mapping_valid;
    std::cout << PatternName<Pattern>() << " path=" << (HasNextTile ? "multi" : "one")
              << " attention_scale=" << attention_scale << ": " << (valid ? "pass" : "fail")
              << " ck_mapping=" << (ck_mapping_valid ? "pass" : "fail") << '\n';
    return valid;
}

} // namespace

int main()
{
    try
    {
        int device_count = 0;
        CheckHip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if(device_count == 0)
        {
            return 77;
        }

        int device = 0;
        CheckHip(hipGetDevice(&device), "hipGetDevice");
        hipDeviceProp_t properties{};
        CheckHip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
        if(std::string(properties.gcnArchName).find("gfx125") == std::string::npos)
        {
            return 77;
        }

        constexpr float kDefaultAttentionScale = 0.07216878f;
        return RunCase<InputPattern::Finite, false>(kDefaultAttentionScale) &&
                       RunCase<InputPattern::Finite, true>(kDefaultAttentionScale) &&
                       RunCase<InputPattern::Finite, true>(0.25f) &&
                       RunCase<InputPattern::MixedFiniteAndMasked, true>(kDefaultAttentionScale) &&
                       RunCase<InputPattern::AllMaskedEmpty, false>(kDefaultAttentionScale) &&
                       RunCase<InputPattern::AllMaskedFiniteHistory, true>(kDefaultAttentionScale)
                   ? 0
                   : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

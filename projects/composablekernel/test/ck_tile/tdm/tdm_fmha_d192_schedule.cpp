// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace {

using Schedule = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Schedule;
using Token    = ck_tile::FmhaD192ScheduleToken;
using Mapping  = ck_tile::FmhaD192ScoreFragmentMapping;
using Softmax  = ck_tile::FmhaD192SplitSoftmax;
using Executor = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;

using D192BlockShape = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, 192>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

struct D192MappingProblem
{
    static constexpr bool kIsGroupMode = false;
    struct FmhaMask
    {
        static constexpr bool IsMasking = false;
    };

    using QDataType      = ck_tile::bf16_t;
    using KDataType      = ck_tile::bf16_t;
    using VDataType      = ck_tile::bf16_t;
    using SaccDataType   = float;
    using BlockFmhaShape = D192BlockShape;

    static constexpr ck_tile::index_t kBlockSize = 128;
    static constexpr auto BiasEnum               = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
    static constexpr bool kHasLogitsSoftCap      = false;
};

using D192Policy = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;
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
using D192ScoreTile  = decltype(D192BlockGemm::MakeCBlockTile());

struct MaxOp
{
    CK_TILE_HOST_DEVICE float operator()(float x, float y) const { return ck_tile::max(x, y); }
};

using D192RowTile = decltype(ck_tile::block_tile_reduce<float>(
    D192ScoreTile{}, ck_tile::sequence<1>{}, MaxOp{}, -ck_tile::numeric<float>::infinity()));
static_assert(D192RowTile::get_thread_buffer_size() == 2);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
using D192ProductionBlockGemm =
    ck_tile::remove_cvref_t<decltype(D192Policy::GetQKBlockGemm<D192MappingProblem>())>;
static_assert(std::is_same_v<D192BlockGemm, D192ProductionBlockGemm>);
#endif

constexpr bool ValidateMappingAgainstCkDistribution()
{
    constexpr auto descriptor = D192ScoreTile::ThreadTensorDesc{};
    constexpr auto lengths    = descriptor.get_lengths();

    static_assert(lengths[ck_tile::number<0>{}] == Mapping::kNumMIter);
    static_assert(lengths[ck_tile::number<1>{}] == Mapping::kNumNIter);
    static_assert(lengths[ck_tile::number<2>{}] == 1);
    static_assert(lengths[ck_tile::number<3>{}] == Mapping::kElementsPerWmma);
    static_assert(D192ScoreTile::get_thread_buffer_size() == Mapping::kThreadBufferSize);

    bool valid = true;
    ck_tile::static_ford<
        ck_tile::sequence<Mapping::kNumMIter, Mapping::kNumNIter, Mapping::kElementsPerWmma>>{}(
        [&](auto indices) {
            constexpr auto m_iter      = indices[ck_tile::number<0>{}];
            constexpr auto full_n_iter = indices[ck_tile::number<1>{}];
            constexpr auto element     = indices[ck_tile::number<2>{}];
            constexpr auto su          = full_n_iter / Mapping::kNIterPerSu;
            constexpr auto msb         = m_iter * 2 + full_n_iter % Mapping::kNIterPerSu;
            constexpr auto pair =
                su * Mapping::kPairsPerFragment + element / Mapping::kElementsPerPair;

            constexpr auto ck_offset = descriptor.calculate_offset(
                ck_tile::make_tuple(m_iter, full_n_iter, ck_tile::number<0>{}, element));
            constexpr auto mapped_offset =
                Mapping::GetThreadBufferOffset(msb, pair, element % Mapping::kElementsPerPair);

            valid = valid && ck_offset == mapped_offset;
        });

    return valid;
}

bool ValidatePairAccessors()
{
    auto score = D192ScoreTile{};
    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score.get_thread_buffer()[i] = static_cast<float>(i);
    }

    bool valid = true;
    ck_tile::static_ford<ck_tile::sequence<Mapping::kNumMsb, Mapping::kPairsPerMsb>>{}(
        [&](auto indices) {
            constexpr auto msb  = indices[ck_tile::number<0>{}];
            constexpr auto pair = indices[ck_tile::number<1>{}];
            const auto value    = Mapping::LoadPair<msb, pair>(score);
            const auto offset   = Mapping::GetThreadBufferOffset(msb, pair, 0);

            valid = valid && value[0] == static_cast<float>(offset) &&
                    value[1] == static_cast<float>(offset + 1);

            Mapping::Pair replacement{static_cast<float>(offset + 1000),
                                      static_cast<float>(offset + 1001)};
            Mapping::StorePair<msb, pair>(score, replacement);
        });

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        valid = valid && score.get_thread_buffer()[i] == static_cast<float>(i + 1000);
    }

    return valid;
}

struct TokenVisitor
{
    ck_tile::index_t* counts;
    bool* valid;

    template <typename TokenConstant, typename Ordinal>
    CK_TILE_HOST_DEVICE constexpr void operator()(TokenConstant, Ordinal) const
    {
        constexpr auto token = static_cast<ck_tile::index_t>(TokenConstant::value);
        *valid               = *valid && counts[token] == Ordinal::value;
        ++counts[token];
    }
};

bool ValidateRowVisitors()
{
    ck_tile::index_t qk_counts[23] = {};
    ck_tile::index_t pv_counts[23] = {};
    bool valid                     = true;

    ck_tile::static_ford<ck_tile::sequence<Schedule::kNumStages, Schedule::kQkWmmasPerStage>>{}(
        [&](auto indices) {
            constexpr auto stage = indices[ck_tile::number<0>{}];
            constexpr auto wmma  = indices[ck_tile::number<1>{}];
            Schedule::VisitQkRowHalf<stage, wmma, false>(TokenVisitor{qk_counts, &valid});
            Schedule::VisitQkRowHalf<stage, wmma, true>(TokenVisitor{qk_counts, &valid});
        });
    ck_tile::static_ford<ck_tile::sequence<Schedule::kNumStages, Schedule::kPvWmmasPerStage>>{}(
        [&](auto indices) {
            constexpr auto stage = indices[ck_tile::number<0>{}];
            constexpr auto wmma  = indices[ck_tile::number<1>{}];
            Schedule::VisitPvRowHalf<stage, wmma, false>(TokenVisitor{pv_counts, &valid});
            Schedule::VisitPvRowHalf<stage, wmma, true>(TokenVisitor{pv_counts, &valid});
        });

    for(ck_tile::index_t token = 0; token < 23; ++token)
    {
        valid &= qk_counts[token] ==
                 Schedule::CountToken(Schedule::kQkRows, static_cast<Schedule::Token>(token));
        valid &= pv_counts[token] ==
                 Schedule::CountToken(Schedule::kPvRows, static_cast<Schedule::Token>(token));
    }
    return valid;
}

bool ValidateRowExecutor()
{
    ck_tile::index_t trace[16] = {};
    ck_tile::index_t cursor    = 0;
    auto emit_wmma             = [&](auto stage, auto wmma) {
        trace[cursor++] = 1000 + decltype(stage)::value * 100 + decltype(wmma)::value;
    };
    auto emit_token = [&](auto token, auto ordinal) {
        trace[cursor++] =
            static_cast<ck_tile::index_t>(decltype(token)::value) * 100 + decltype(ordinal)::value;
    };
    auto emit_point = [&](auto, auto, auto point) {
        trace[cursor++] = 2000 + static_cast<ck_tile::index_t>(decltype(point)::value);
    };

    Executor::ExecuteQkRow<0, 0>(emit_wmma, emit_token, emit_point);
    const ck_tile::index_t expected[] = {
        1000,
        2000 + static_cast<ck_tile::index_t>(Executor::Point::AfterWmma),
        static_cast<ck_tile::index_t>(Token::Tdm) * 100,
        2000 + static_cast<ck_tile::index_t>(Executor::Point::BetweenTokenHalves),
        static_cast<ck_tile::index_t>(Token::Tdm) * 100 + 1,
        static_cast<ck_tile::index_t>(Token::P2M0) * 100,
        2000 + static_cast<ck_tile::index_t>(Executor::Point::AfterTokens),
    };

    bool valid = cursor == static_cast<ck_tile::index_t>(sizeof(expected) / sizeof(expected[0]));
    for(ck_tile::index_t i = 0; i < cursor; ++i)
    {
        valid &= trace[i] == expected[i];
    }
    return valid;
}

constexpr bool ValidateOutputRescaleOrdinals()
{
    bool seen[Mapping::kNumMsb][Mapping::kNumSu] = {};
    ck_tile::index_t ordinal                     = 0;

    for(ck_tile::index_t row = 0; row < Schedule::kNumQkRows; ++row)
    {
        for(ck_tile::index_t slot = 0; slot < Schedule::kQkRows[row].size; ++slot)
        {
            if(Schedule::kQkRows[row][slot] == Token::ORescale)
            {
                const ck_tile::index_t output_tile = ordinal / Mapping::kNumMsb;
                const ck_tile::index_t msb         = ordinal % Mapping::kNumMsb;
                if(output_tile >= Mapping::kNumSu || seen[msb][output_tile])
                {
                    return false;
                }
                seen[msb][output_tile] = true;
                ++ordinal;
            }
        }
    }

    if(ordinal != Mapping::kNumMsb * Mapping::kNumSu)
    {
        return false;
    }
    for(ck_tile::index_t msb = 0; msb < Mapping::kNumMsb; ++msb)
    {
        for(ck_tile::index_t output_tile = 0; output_tile < Mapping::kNumSu; ++output_tile)
        {
            if(!seen[msb][output_tile])
            {
                return false;
            }
        }
    }
    return true;
}

__global__ void
D192SoftmaxPrimitiveProbe(const float* input, float* output, ck_tile::bf16x2_t* packed_output)
{
    auto score = D192ScoreTile{};
    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score.get_thread_buffer()[i] = input[i];
    }

    const auto p0_m0 = Softmax::RunPart0<0>(score, input[128], input[129]);
    const auto p0_m1 = Softmax::RunPart0<1>(score, input[130], input[129]);
    const auto p1 =
        Softmax::RunPart1(p0_m0.local_max, p0_m1.local_max, p0_m0.old_max_log2e, input[129]);
    const auto local_sum = Softmax::RunPart2LocalSum<0>(score, p1.row_max, input[129]);
    const auto row_sum   = Softmax::UpdateRowSum(input[131], local_sum, p1.delta);
    Softmax::RescaleOutput<0>(score, Softmax::Exp2(p1.delta));
    const auto exp0 = Mapping::LoadPair<0, 0>(score);

    output[0]        = p1.delta;
    output[1]        = row_sum;
    packed_output[0] = Softmax::ConvertPairToBf16(exp0);
}

__global__ void D192SplitSoftmaxPolicyProbe(const float* input, float* output)
{
    auto score      = D192ScoreTile{};
    auto row_max    = D192RowTile{};
    auto row_sum    = D192RowTile{};
    auto output_acc = D192ScoreTile{};

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score.get_thread_buffer()[i]      = input[i];
        output_acc.get_thread_buffer()[i] = input[i + Mapping::kThreadBufferSize];
    }
    row_max.get_thread_buffer()[0] = input[2 * Mapping::kThreadBufferSize + 0];
    row_max.get_thread_buffer()[1] = input[2 * Mapping::kThreadBufferSize + 1];
    row_sum.get_thread_buffer()[0] = input[2 * Mapping::kThreadBufferSize + 2];
    row_sum.get_thread_buffer()[1] = input[2 * Mapping::kThreadBufferSize + 3];

    D192Policy::RunSplitSoftmax<D192MappingProblem>(
        score, row_max, row_sum, output_acc, input[2 * Mapping::kThreadBufferSize + 4]);

    output[0] = row_max.get_thread_buffer()[0];
    output[1] = row_max.get_thread_buffer()[1];
    output[2] = row_sum.get_thread_buffer()[0];
    output[3] = row_sum.get_thread_buffer()[1];
    output[4] = score.get_thread_buffer()[0] + output_acc.get_thread_buffer()[0];
}

static_assert(Schedule::ValidateContract());
static_assert(Mapping::ValidateBijection());
static_assert(ValidateMappingAgainstCkDistribution());
static_assert(Schedule::kNumQkRows == 96);
static_assert(Schedule::kNumPvRows == 64);
static_assert(Schedule::kNumWmmaRows == 160);
static_assert(ValidateOutputRescaleOrdinals());

constexpr auto kQkStage0Row0 = Schedule::GetQkRow<0>(0);
static_assert(kQkStage0Row0.size == 3);
static_assert(kQkStage0Row0[0] == Token::Tdm);
static_assert(kQkStage0Row0[1] == Token::Tdm);
static_assert(kQkStage0Row0[2] == Token::P2M0);
static_assert(Schedule::GetQkTokenOrdinal<0, 0, 0>() == 0);
static_assert(Schedule::GetQkTokenOrdinal<0, 0, 1>() == 1);
static_assert(Schedule::GetQkTokenOrdinal<0, 0, 2>() == 0);

constexpr auto kQkStage0Row1 = Schedule::GetQkRow<0>(1);
static_assert(kQkStage0Row1.size == 4);
static_assert(kQkStage0Row1[0] == Token::KM0);
static_assert(kQkStage0Row1[1] == Token::KM0);
static_assert(kQkStage0Row1[2] == Token::KM0);
static_assert(kQkStage0Row1[3] == Token::ORescale);
static_assert(Schedule::GetQkTokenOrdinal<0, 1, 0>() == 0);
static_assert(Schedule::GetQkTokenOrdinal<0, 1, 2>() == 2);
static_assert(Schedule::GetQkTokenOrdinal<0, 1, 3>() == 0);

constexpr auto kQkStage3LastRow = Schedule::GetQkRow<3>(23);
static_assert(kQkStage3LastRow.size == 0);

constexpr auto kPvStage1P1Row = Schedule::GetPvRow<1>(9);
static_assert(kPvStage1P1Row.size == 4);
static_assert(kPvStage1P1Row[0] == Token::P1);
static_assert(kPvStage1P1Row[1] == Token::P1);
static_assert(kPvStage1P1Row[2] == Token::P1);
static_assert(kPvStage1P1Row[3] == Token::P1);
static_assert(Schedule::GetPvTokenOrdinal<1, 9, 0>() == 0);
static_assert(Schedule::GetPvTokenOrdinal<1, 9, 3>() == 3);

constexpr auto kPvStage1P1RowNext = Schedule::GetPvRow<1>(10);
static_assert(kPvStage1P1RowNext.size == 4);
static_assert(Schedule::GetPvTokenOrdinal<1, 10, 0>() == 4);
static_assert(Schedule::GetPvTokenOrdinal<1, 10, 3>() == 7);

constexpr auto kPvStage3Row0 = Schedule::GetPvRow<3>(0);
static_assert(kPvStage3Row0.size == 5);
static_assert(kPvStage3Row0[0] == Token::KM0);
static_assert(kPvStage3Row0[1] == Token::KM0);
static_assert(kPvStage3Row0[2] == Token::KM0);
static_assert(kPvStage3Row0[3] == Token::ExpM0);
static_assert(kPvStage3Row0[4] == Token::ExpM0);

constexpr auto kPvStage3LastRow = Schedule::GetPvRow<3>(15);
static_assert(kPvStage3LastRow.size == 0);

} // namespace

int main()
{
    return Schedule::ValidateContract() && ValidateMappingAgainstCkDistribution() &&
                   ValidatePairAccessors() && ValidateRowVisitors() && ValidateRowExecutor()
               ? 0
               : 1;
}

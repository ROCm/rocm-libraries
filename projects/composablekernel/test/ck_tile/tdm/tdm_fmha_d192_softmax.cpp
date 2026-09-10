// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace {

using Mapping = ck_tile::FmhaD192ScoreFragmentMapping;
using Policy  = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;

using D192BlockShape = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, 192>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

template <bool Masking>
struct D192SoftmaxProblem
{
    static constexpr bool kIsGroupMode = false;
    struct FmhaMask
    {
        static constexpr bool IsMasking = Masking;
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

struct MaxOp
{
    CK_TILE_HOST_DEVICE float operator()(float x, float y) const { return ck_tile::max(x, y); }
};

using RowTile = decltype(ck_tile::block_tile_reduce<float>(
    ScoreTile{}, ck_tile::sequence<1>{}, MaxOp{}, -ck_tile::numeric<float>::infinity()));

constexpr ck_tile::index_t kThreads            = 128;
constexpr ck_tile::index_t kInputStride        = 2 * Mapping::kThreadBufferSize + 5;
constexpr ck_tile::index_t kOutputStride       = 4 + 2 * Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kTwoTileCount       = 2;
constexpr ck_tile::index_t kTwoTileStateOffset = (kTwoTileCount + 1) * Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kTwoTileInputStride = kTwoTileStateOffset + 5;
constexpr ck_tile::index_t kTwoTileScoreOutputOffset = 6;
constexpr ck_tile::index_t kTwoTileOutputAccumulatorOffset =
    kTwoTileScoreOutputOffset + kTwoTileCount * Mapping::kThreadBufferSize;
constexpr ck_tile::index_t kTwoTileOutputStride =
    kTwoTileOutputAccumulatorOffset + Mapping::kThreadBufferSize;

template <bool Masking>
__global__ void RunSplitSoftmax(const float* input, float* output)
{
    const auto* lane_input = input + threadIdx.x * kInputStride;
    auto* lane_output      = output + threadIdx.x * kOutputStride;
    auto score             = ScoreTile{};
    auto row_max           = RowTile{};
    auto row_sum           = RowTile{};
    auto output_acc        = ScoreTile{};

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score.get_thread_buffer()[i]      = lane_input[i];
        output_acc.get_thread_buffer()[i] = lane_input[Mapping::kThreadBufferSize + i];
    }
    row_max.get_thread_buffer()[0] = lane_input[2 * Mapping::kThreadBufferSize + 0];
    row_max.get_thread_buffer()[1] = lane_input[2 * Mapping::kThreadBufferSize + 1];
    row_sum.get_thread_buffer()[0] = lane_input[2 * Mapping::kThreadBufferSize + 2];
    row_sum.get_thread_buffer()[1] = lane_input[2 * Mapping::kThreadBufferSize + 3];

    Policy::RunSplitSoftmax<D192SoftmaxProblem<Masking>>(
        score, row_max, row_sum, output_acc, lane_input[2 * Mapping::kThreadBufferSize + 4]);

    lane_output[0] = row_max.get_thread_buffer()[0];
    lane_output[1] = row_max.get_thread_buffer()[1];
    lane_output[2] = row_sum.get_thread_buffer()[0];
    lane_output[3] = row_sum.get_thread_buffer()[1];
    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        lane_output[4 + i]                              = score.get_thread_buffer()[i];
        lane_output[4 + Mapping::kThreadBufferSize + i] = output_acc.get_thread_buffer()[i];
    }
}

template <bool Masking>
__global__ void RunSplitSoftmaxTwoTiles(const float* input, float* output)
{
    const auto* lane_input = input + threadIdx.x * kTwoTileInputStride;
    auto* lane_output      = output + threadIdx.x * kTwoTileOutputStride;
    auto row_max           = RowTile{};
    auto row_sum           = RowTile{};
    auto output_acc        = ScoreTile{};

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        output_acc.get_thread_buffer()[i] =
            lane_input[kTwoTileCount * Mapping::kThreadBufferSize + i];
    }
    row_max.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 0];
    row_max.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 1];
    row_sum.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 2];
    row_sum.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 3];

    ck_tile::static_for<0, kTwoTileCount, 1>{}([&](auto tile) {
        auto score = ScoreTile{};
        for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
        {
            score.get_thread_buffer()[i] = lane_input[tile * Mapping::kThreadBufferSize + i];
        }

        Policy::RunSplitSoftmax<D192SoftmaxProblem<Masking>>(
            score, row_max, row_sum, output_acc, lane_input[kTwoTileStateOffset + 4]);

        for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
        {
            lane_output[kTwoTileScoreOutputOffset + tile * Mapping::kThreadBufferSize + i] =
                score.get_thread_buffer()[i];
        }
    });

    lane_output[0] = row_max.get_thread_buffer()[0];
    lane_output[1] = row_max.get_thread_buffer()[1];
    lane_output[2] = row_sum.get_thread_buffer()[0];
    lane_output[3] = row_sum.get_thread_buffer()[1];
    lane_output[4] =
        row_max.get_thread_buffer()[0] * lane_input[kTwoTileStateOffset + 4] / ck_tile::log2e_v<> +
        ck_tile::log(row_sum.get_thread_buffer()[0]);
    lane_output[5] =
        row_max.get_thread_buffer()[1] * lane_input[kTwoTileStateOffset + 4] / ck_tile::log2e_v<> +
        ck_tile::log(row_sum.get_thread_buffer()[1]);

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        lane_output[kTwoTileOutputAccumulatorOffset + i] = output_acc.get_thread_buffer()[i];
    }
}

template <bool Masking>
__global__ void RunSplitSoftmaxTwoTilesPipelined(const float* input, float* output)
{
    const auto* lane_input = input + threadIdx.x * kTwoTileInputStride;
    auto* lane_output      = output + threadIdx.x * kTwoTileOutputStride;
    auto score_previous    = ScoreTile{};
    auto score_current     = ScoreTile{};
    auto row_max           = RowTile{};
    auto row_sum           = RowTile{};
    auto output_acc        = ScoreTile{};

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score_previous.get_thread_buffer()[i] = lane_input[i];
        output_acc.get_thread_buffer()[i] =
            lane_input[kTwoTileCount * Mapping::kThreadBufferSize + i];
    }
    row_max.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 0];
    row_max.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 1];
    row_sum.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 2];
    row_sum.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 3];
    const float scale              = lane_input[kTwoTileStateOffset + 4];

    float previous_delta_m0;
    float previous_delta_m1;
    Policy::RunSplitSoftmaxPart01<D192SoftmaxProblem<Masking>>(
        score_previous, row_max, scale, previous_delta_m0, previous_delta_m1);

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score_current.get_thread_buffer()[i] = lane_input[Mapping::kThreadBufferSize + i];
    }

    float previous_output_scale_m0;
    float previous_output_scale_m1;
    Policy::RunSplitSoftmaxPart2AndGetScale<D192SoftmaxProblem<Masking>>(score_previous,
                                                                         row_max,
                                                                         row_sum,
                                                                         scale,
                                                                         previous_delta_m0,
                                                                         previous_delta_m1,
                                                                         previous_output_scale_m0,
                                                                         previous_output_scale_m1);
    ck_tile::static_for<0, 16, 1>{}([&](auto ordinal) {
        Policy::RunOutputRescaleToken<decltype(ordinal)::value>(
            output_acc, previous_output_scale_m0, previous_output_scale_m1);
    });
    float current_delta_m0;
    float current_delta_m1;
    Policy::RunSplitSoftmaxPart01<D192SoftmaxProblem<Masking>>(
        score_current, row_max, scale, current_delta_m0, current_delta_m1);

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        lane_output[kTwoTileScoreOutputOffset + i] = score_previous.get_thread_buffer()[i];
    }

    float current_output_scale_m0;
    float current_output_scale_m1;
    Policy::RunSplitSoftmaxPart2AndGetScale<D192SoftmaxProblem<Masking>>(score_current,
                                                                         row_max,
                                                                         row_sum,
                                                                         scale,
                                                                         current_delta_m0,
                                                                         current_delta_m1,
                                                                         current_output_scale_m0,
                                                                         current_output_scale_m1);
    ck_tile::static_for<0, 16, 1>{}([&](auto ordinal) {
        Policy::RunOutputRescaleToken<decltype(ordinal)::value>(
            output_acc, current_output_scale_m0, current_output_scale_m1);
    });

    lane_output[0] = row_max.get_thread_buffer()[0];
    lane_output[1] = row_max.get_thread_buffer()[1];
    lane_output[2] = row_sum.get_thread_buffer()[0];
    lane_output[3] = row_sum.get_thread_buffer()[1];
    lane_output[4] = row_max.get_thread_buffer()[0] * scale / ck_tile::log2e_v<> +
                     ck_tile::log(row_sum.get_thread_buffer()[0]);
    lane_output[5] = row_max.get_thread_buffer()[1] * scale / ck_tile::log2e_v<> +
                     ck_tile::log(row_sum.get_thread_buffer()[1]);

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        lane_output[kTwoTileScoreOutputOffset + Mapping::kThreadBufferSize + i] =
            score_current.get_thread_buffer()[i];
        lane_output[kTwoTileOutputAccumulatorOffset + i] = output_acc.get_thread_buffer()[i];
    }
}

void CheckHip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

float ScoreValue(ck_tile::index_t lane, ck_tile::index_t msb, ck_tile::index_t scalar)
{
    return -4.0f + 0.01f * static_cast<float>(lane & 15) + 0.02f * static_cast<float>(lane >> 4) +
           0.1f * static_cast<float>(msb) + 0.001f * static_cast<float>(scalar);
}

bool Near(float actual, float expected)
{
    if(std::isinf(expected))
    {
        return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    }
    return std::abs(actual - expected) <= 2.0e-4f * std::max(1.0f, std::abs(expected));
}

template <bool Masking>
bool RunCase()
{
    std::vector<float> input(kThreads * kInputStride);
    std::vector<float> output(kThreads * kOutputStride);
    constexpr float scale = 0.125f;

    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        auto* lane_input        = input.data() + thread * kInputStride;
        const auto lane         = thread % 32;
        const float initial_max = Masking ? -std::numeric_limits<float>::infinity()
                                          : -5.0f + 0.01f * static_cast<float>(lane & 15);

        for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
        {
            const auto coordinate = Mapping::DecodeThreadBufferOffset(offset);
            lane_input[offset] =
                Masking
                    ? -std::numeric_limits<float>::infinity()
                    : ScoreValue(lane, coordinate.msb, coordinate.pair * 2 + coordinate.element);
            lane_input[Mapping::kThreadBufferSize + offset] =
                1.0f + 0.001f * static_cast<float>(offset);
        }
        lane_input[2 * Mapping::kThreadBufferSize + 0] = initial_max;
        lane_input[2 * Mapping::kThreadBufferSize + 1] = initial_max + (Masking ? 0.0f : 0.25f);
        lane_input[2 * Mapping::kThreadBufferSize + 2] = Masking ? 0.0f : 0.75f;
        lane_input[2 * Mapping::kThreadBufferSize + 3] = Masking ? 0.0f : 1.25f;
        lane_input[2 * Mapping::kThreadBufferSize + 4] = scale;
    }

    float* device_input  = nullptr;
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_input, input.size() * sizeof(float)), "hipMalloc input");
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)), "hipMalloc output");
    CheckHip(
        hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy input");
    hipLaunchKernelGGL(
        (RunSplitSoftmax<Masking>), dim3(1), dim3(kThreads), 0, 0, device_input, device_output);
    CheckHip(hipGetLastError(), "softmax probe launch");
    CheckHip(hipDeviceSynchronize(), "softmax probe synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy output");
    CheckHip(hipFree(device_input), "hipFree input");
    CheckHip(hipFree(device_output), "hipFree output");

    bool valid = true;
    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        const auto lane      = thread % 32;
        const auto wave      = thread / 32;
        const auto peer      = wave * 32 + (lane ^ 16);
        const auto* lane_in  = input.data() + thread * kInputStride;
        const auto* peer_in  = input.data() + peer * kInputStride;
        const auto* lane_out = output.data() + thread * kOutputStride;

        for(ck_tile::index_t row = 0; row < 2; ++row)
        {
            const float old_max = lane_in[2 * Mapping::kThreadBufferSize + row];
            const float old_sum = lane_in[2 * Mapping::kThreadBufferSize + 2 + row];
            float new_max       = old_max;
            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                        new_max           = std::max(new_max, lane_in[offset]);
                        new_max           = std::max(new_max, peer_in[offset]);
                    }
                }
            }

            const float exponent_max = Masking && std::isinf(new_max) ? 0.0f : new_max;
            const float delta        = std::fma(-exponent_max, scale, old_max * scale);
            float tile_sum           = 0.0f;
            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                        const float expected =
                            std::exp2(std::fma(lane_in[offset], scale, -exponent_max * scale));
                        const float actual = lane_out[4 + offset];
                        valid &= Near(actual, expected);
                        tile_sum +=
                            expected +
                            std::exp2(std::fma(peer_in[offset], scale, -exponent_max * scale));
                    }
                }
            }

            valid &= Near(lane_out[row], new_max);
            valid &= Near(lane_out[2 + row], std::fma(std::exp2(delta), old_sum, tile_sum));

            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                        const float initial_output = lane_in[Mapping::kThreadBufferSize + offset];
                        valid &= Near(lane_out[4 + Mapping::kThreadBufferSize + offset],
                                      initial_output * std::exp2(delta));
                    }
                }
            }
        }
    }

    std::cout << (Masking ? "masked_all_inf" : "lane_distinct") << ": " << (valid ? "pass" : "fail")
              << '\n';
    return valid;
}

float TwoTileScoreValue(ck_tile::index_t tile,
                        ck_tile::index_t lane,
                        ck_tile::index_t msb,
                        ck_tile::index_t scalar)
{
    return ScoreValue(lane, msb, scalar) + 0.2f * static_cast<float>(tile);
}

template <bool Masking, bool Pipelined = false>
bool RunTwoTileCase()
{
    std::vector<float> input(kThreads * kTwoTileInputStride);
    std::vector<float> output(kThreads * kTwoTileOutputStride);
    constexpr float scale = 0.125f;

    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        auto* lane_input        = input.data() + thread * kTwoTileInputStride;
        const auto lane         = thread % 32;
        const float initial_max = Masking ? -std::numeric_limits<float>::infinity()
                                          : -5.0f + 0.01f * static_cast<float>(lane & 15);

        for(ck_tile::index_t tile = 0; tile < kTwoTileCount; ++tile)
        {
            for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
            {
                const auto coordinate = Mapping::DecodeThreadBufferOffset(offset);
                lane_input[tile * Mapping::kThreadBufferSize + offset] =
                    Masking
                        ? -std::numeric_limits<float>::infinity()
                        : TwoTileScoreValue(
                              tile, lane, coordinate.msb, coordinate.pair * 2 + coordinate.element);
            }
        }
        for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
        {
            lane_input[kTwoTileCount * Mapping::kThreadBufferSize + offset] =
                1.0f + 0.001f * static_cast<float>(offset);
        }
        lane_input[kTwoTileStateOffset + 0] = initial_max;
        lane_input[kTwoTileStateOffset + 1] = initial_max + (Masking ? 0.0f : 0.25f);
        lane_input[kTwoTileStateOffset + 2] = Masking ? 0.0f : 0.75f;
        lane_input[kTwoTileStateOffset + 3] = Masking ? 0.0f : 1.25f;
        lane_input[kTwoTileStateOffset + 4] = scale;
    }

    float* device_input  = nullptr;
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_input, input.size() * sizeof(float)), "hipMalloc input");
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)), "hipMalloc output");
    CheckHip(
        hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy input");
    if constexpr(Pipelined)
    {
        hipLaunchKernelGGL((RunSplitSoftmaxTwoTilesPipelined<Masking>),
                           dim3(1),
                           dim3(kThreads),
                           0,
                           0,
                           device_input,
                           device_output);
    }
    else
    {
        hipLaunchKernelGGL((RunSplitSoftmaxTwoTiles<Masking>),
                           dim3(1),
                           dim3(kThreads),
                           0,
                           0,
                           device_input,
                           device_output);
    }
    CheckHip(hipGetLastError(), "two-tile softmax probe launch");
    CheckHip(hipDeviceSynchronize(), "two-tile softmax probe synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy output");
    CheckHip(hipFree(device_input), "hipFree input");
    CheckHip(hipFree(device_output), "hipFree output");

    bool valid = true;
    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        const auto lane       = thread % 32;
        const auto wave       = thread / 32;
        const auto peer       = wave * 32 + (lane ^ 16);
        const auto* lane_in   = input.data() + thread * kTwoTileInputStride;
        const auto* peer_in   = input.data() + peer * kTwoTileInputStride;
        const auto* lane_out  = output.data() + thread * kTwoTileOutputStride;
        float expected_max[2] = {lane_in[kTwoTileStateOffset + 0],
                                 lane_in[kTwoTileStateOffset + 1]};
        float expected_sum[2] = {lane_in[kTwoTileStateOffset + 2],
                                 lane_in[kTwoTileStateOffset + 3]};
        std::vector<float> expected_output(Mapping::kThreadBufferSize);

        for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
        {
            expected_output[offset] = lane_in[kTwoTileCount * Mapping::kThreadBufferSize + offset];
        }

        for(ck_tile::index_t tile = 0; tile < kTwoTileCount; ++tile)
        {
            for(ck_tile::index_t row = 0; row < 2; ++row)
            {
                const float old_max = expected_max[row];
                float new_max       = old_max;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                            const auto input_offset = tile * Mapping::kThreadBufferSize + offset;
                            new_max                 = std::max(new_max, lane_in[input_offset]);
                            new_max                 = std::max(new_max, peer_in[input_offset]);
                        }
                    }
                }

                const float exponent_max = Masking && std::isinf(new_max) ? 0.0f : new_max;
                const float delta        = std::fma(-exponent_max, scale, old_max * scale);
                float tile_sum           = 0.0f;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                            const auto input_offset = tile * Mapping::kThreadBufferSize + offset;
                            const float expected    = std::exp2(
                                std::fma(lane_in[input_offset], scale, -exponent_max * scale));
                            valid &=
                                Near(lane_out[kTwoTileScoreOutputOffset + input_offset], expected);
                            tile_sum += expected + std::exp2(std::fma(peer_in[input_offset],
                                                                      scale,
                                                                      -exponent_max * scale));
                        }
                    }
                }

                expected_sum[row] = std::fma(std::exp2(delta), expected_sum[row], tile_sum);
                expected_max[row] = new_max;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                            expected_output[offset] *= std::exp2(delta);
                        }
                    }
                }
            }
        }

        for(ck_tile::index_t row = 0; row < 2; ++row)
        {
            valid &= Near(lane_out[row], expected_max[row]);
            valid &= Near(lane_out[2 + row], expected_sum[row]);
            const float expected_lse =
                expected_max[row] * scale / ck_tile::log2e_v<> + std::log(expected_sum[row]);
            valid &= Near(lane_out[4 + row], expected_lse);
        }
        for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
        {
            valid &=
                Near(lane_out[kTwoTileOutputAccumulatorOffset + offset], expected_output[offset]);
        }
    }

    const auto* case_name =
        Pipelined ? (Masking ? "previous_tile_masked_all_inf" : "previous_tile_online")
                  : (Masking ? "two_tile_masked_all_inf" : "two_tile_online");
    std::cout << case_name << ": " << (valid ? "pass" : "fail") << '\n';
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

        return RunCase<false>() && RunCase<true>() && RunTwoTileCase<false>() &&
                       RunTwoTileCase<true>() && RunTwoTileCase<false, true>() &&
                       RunTwoTileCase<true, true>()
                   ? 0
                   : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

/**
 * @file IntegrationGpuConvNaive.cpp
 * @brief The naive convolution engines end to end through the hipDNN frontend: each is
 *        offered for its direction, pinned, built and executed against the CPU reference
 *        graph executor, and none is offered for a problem over its workload budget.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* FWD_ENGINE = "hipkernel:CONV_NAIVE_FWD";
constexpr const char* BWD_ENGINE = "hipkernel:CONV_NAIVE_BWD";
constexpr const char* WRW_ENGINE = "hipkernel:CONV_NAIVE_WRW";

enum class Direction
{
    FWD,
    BWD,
    WRW,
};

struct ConvNaiveCase
{
    std::string name;
    Direction direction;
    bool channelsLast;
};

/// A small convolution with distinct extents on every axis, padding, stride 2 on one axis and
/// two groups: N=2, C=4, H=9, W=7, K=6, R=3, S=2.
struct Problem
{
    std::vector<int64_t> x = {2, 4, 9, 7};
    std::vector<int64_t> w = {6, 2, 3, 2};
    std::vector<int64_t> y = {2, 6, 5, 8};
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {2, 1};
    std::vector<int64_t> dilation = {1, 1};
};

std::vector<int64_t> stridesFor(const std::vector<int64_t>& dims, bool channelsLast)
{
    // dims are (n, c, h, w).
    if(channelsLast)
    {
        return {dims[1] * dims[2] * dims[3], 1, dims[3] * dims[1], dims[1]};
    }
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

std::shared_ptr<TensorAttributes> tensor(int64_t uid,
                                         const std::string& name,
                                         const std::vector<int64_t>& dims,
                                         bool channelsLast)
{
    auto attributes = std::make_shared<TensorAttributes>();
    attributes->set_uid(uid)
        .set_name(name)
        .set_dim(dims)
        .set_stride(stridesFor(dims, channelsLast))
        .set_data_type(DataType::FLOAT);
    return attributes;
}

std::shared_ptr<Graph> buildGraph(Direction direction, bool channelsLast, const Problem& problem)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("conv_naive")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    switch(direction)
    {
    case Direction::FWD:
    {
        ConvFpropAttributes attributes;
        attributes.set_pre_padding(problem.prePadding)
            .set_post_padding(problem.postPadding)
            .set_stride(problem.stride)
            .set_dilation(problem.dilation)
            .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
        auto y = graph->conv_fprop(tensor(1, "X", problem.x, channelsLast),
                                   tensor(2, "W", problem.w, channelsLast),
                                   attributes);
        y->set_uid(3)
            .set_name("Y")
            .set_dim(problem.y)
            .set_stride(stridesFor(problem.y, channelsLast))
            .set_output(true)
            .set_data_type(DataType::FLOAT);
        break;
    }
    case Direction::BWD:
    {
        ConvDgradAttributes attributes;
        attributes.set_pre_padding(problem.prePadding)
            .set_post_padding(problem.postPadding)
            .set_stride(problem.stride)
            .set_dilation(problem.dilation)
            .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
        auto dx = graph->conv_dgrad(tensor(3, "DY", problem.y, channelsLast),
                                    tensor(2, "W", problem.w, channelsLast),
                                    attributes);
        dx->set_uid(1)
            .set_name("DX")
            .set_dim(problem.x)
            .set_stride(stridesFor(problem.x, channelsLast))
            .set_output(true)
            .set_data_type(DataType::FLOAT);
        break;
    }
    case Direction::WRW:
    {
        ConvWgradAttributes attributes;
        attributes.set_pre_padding(problem.prePadding)
            .set_post_padding(problem.postPadding)
            .set_stride(problem.stride)
            .set_dilation(problem.dilation)
            .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
        auto dw = graph->conv_wgrad(tensor(3, "DY", problem.y, channelsLast),
                                    tensor(1, "X", problem.x, channelsLast),
                                    attributes);
        dw->set_uid(2)
            .set_name("DW")
            .set_dim(problem.w)
            .set_stride(stridesFor(problem.w, channelsLast))
            .set_output(true)
            .set_data_type(DataType::FLOAT);
        break;
    }
    default:
        break;
    }
    return graph;
}

int64_t engineFor(Direction direction)
{
    switch(direction)
    {
    case Direction::FWD:
        return hipdnn_data_sdk::utilities::engineNameToId(FWD_ENGINE);
    case Direction::BWD:
        return hipdnn_data_sdk::utilities::engineNameToId(BWD_ENGINE);
    case Direction::WRW:
        return hipdnn_data_sdk::utilities::engineNameToId(WRW_ENGINE);
    default:
        break;
    }
    return 0;
}

/// The longest reduction an output element runs, in fp32 epsilons: forward and backward data
/// sum (C/G or K/G) * R * S terms, backward weights N * P * Q.
float toleranceEpsilons(Direction direction, const Problem& problem)
{
    const int64_t filter = problem.w[2] * problem.w[3];
    switch(direction)
    {
    case Direction::FWD:
        return static_cast<float>(problem.w[1] * filter);
    case Direction::BWD:
        return static_cast<float>((problem.w[0] / (problem.x[1] / problem.w[1])) * filter);
    case Direction::WRW:
        return static_cast<float>(problem.y[0] * problem.y[2] * problem.y[3]);
    default:
        break;
    }
    return 1.0f;
}

} // namespace

class IntegrationGpuConvNaive
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      ConvNaiveCase>
{
protected:
    std::vector<int64_t> rankedEngines(Graph& graph)
    {
        std::vector<int64_t> engines;
        auto result = graph.build_operation_graph(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.get_ranked_engine_ids(engines);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        return engines;
    }
};

TEST_P(IntegrationGpuConvNaive, IsOfferedPinnedAndMatchesTheReference)
{
    const auto& param = GetParam();
    const Problem problem;
    auto graph = buildGraph(param.direction, param.channelsLast, problem);
    const int64_t engine = engineFor(param.direction);

    const auto engines = rankedEngines(*graph);
    ASSERT_NE(std::find(engines.begin(), engines.end(), engine), engines.end())
        << "the naive engine is not offered for its own direction";
    for(const auto other : {Direction::FWD, Direction::BWD, Direction::WRW})
    {
        if(other != param.direction)
        {
            EXPECT_EQ(std::find(engines.begin(), engines.end(), engineFor(other)), engines.end())
                << "another direction's naive engine is offered";
        }
    }

    graph->set_preferred_engine_id_ext(engine);
    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t servingEngine = 0;
    result = graph->get_execution_plan_engine_id(servingEngine);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(servingEngine, engine);

    std::vector<BehaviorNote> notes;
    result = graph->get_behavior_notes_for_engine(engine, notes);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_NE(std::find(notes.begin(), notes.end(), BehaviorNote::RUNTIME_COMPILATION),
              notes.end());
    EXPECT_EQ(
        std::find(notes.begin(), notes.end(), BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
        notes.end());

    hip_kernel_provider::test_utilities::GraphVerificationContext context(*graph);
    registerValidatorsForOutputs(context, toleranceEpsilons(param.direction, problem));
    verifyBuiltGraph(context, 0);
}

TEST_P(IntegrationGpuConvNaive, IsNotOfferedOverItsWorkloadBudget)
{
    const auto& param = GetParam();
    // 3x3, C = K = 128, 56x56, padded: 462 M multiply-accumulates.
    Problem problem;
    problem.x = {1, 128, 56, 56};
    problem.w = {128, 128, 3, 3};
    problem.y = {1, 128, 56, 56};
    problem.prePadding = {1, 1};
    problem.postPadding = {1, 1};
    problem.stride = {1, 1};
    auto graph = buildGraph(param.direction, param.channelsLast, problem);

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // This binary loads only this provider, so with the naive engine declining there may be
    // no engine at all; that is the expected outcome, and so is a list without it.
    std::vector<int64_t> engines;
    result = graph->get_ranked_engine_ids(engines);
    if(result.code == ErrorCode::GRAPH_NOT_SUPPORTED)
    {
        return;
    }
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(std::find(engines.begin(), engines.end(), engineFor(param.direction)), engines.end());
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvNaive,
                         ::testing::Values(ConvNaiveCase{"FwdNchw", Direction::FWD, false},
                                           ConvNaiveCase{"FwdNhwc", Direction::FWD, true},
                                           ConvNaiveCase{"BwdNchw", Direction::BWD, false},
                                           ConvNaiveCase{"BwdNhwc", Direction::BWD, true},
                                           ConvNaiveCase{"WrwNchw", Direction::WRW, false},
                                           ConvNaiveCase{"WrwNhwc", Direction::WRW, true}),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV

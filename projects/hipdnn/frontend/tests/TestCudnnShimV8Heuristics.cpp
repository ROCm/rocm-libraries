// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x heuristics and fallback-list builders against the shim's
// backend C API and the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineFallbackList.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

bool keepAll(cudnnBackendDescriptor_t /*config*/)
{
    return false;
}

class TestCudnnShimV8Heuristics : public hipdnn_shim_test::ShimV8PlanFixture
{
};

TEST_F(TestCudnnShimV8Heuristics, ModeStringsRunEachMode)
{
    fe::OperationGraph_v8 graph = convGraph();

    fe::EngineConfigList configs;
    const std::vector<cudnnStatus_t> statuses = fe::get_heuristics_list(
        std::vector<std::string>{"heuristics_instant", "heuristics_fallback"},
        graph,
        keepAll,
        configs);

    // Both modes run the shim's single FALLBACK ranking, so the list repeats.
    EXPECT_EQ(statuses, (std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS}));
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));

    fe::EngineConfigList modeB;
    EXPECT_EQ(fe::get_heuristics_list(
                  std::vector<std::string>{"heuristics_mode_b"}, graph, keepAll, modeB),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(modeB.size(), 3U);

    fe::EngineConfigList unknown;
    EXPECT_TRUE(fe::get_heuristics_list(
                    std::vector<std::string>{"heuristics_unknown"}, graph, keepAll, unknown)
                    .empty());
    EXPECT_TRUE(unknown.empty());
}

// The std::array overloads legacy samples and Paddle call.
TEST_F(TestCudnnShimV8Heuristics, ArrayModeOverloadsRunEachMode)
{
    fe::OperationGraph_v8 graph = convGraph();

    fe::EngineConfigList configs;
    const std::vector<cudnnStatus_t> statuses = fe::get_heuristics_list<2>(
        std::array<std::string, 2>{"heuristics_instant", "heuristics_fallback"},
        graph,
        keepAll,
        configs);
    EXPECT_EQ(statuses, (std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS}));
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));

    // CUDNN_HEUR_MODES_COUNT entries are skipped.
    const fe::EngineConfigList byMode = fe::get_heuristics_list<2>(
        std::array<cudnnBackendHeurMode_t, 2>{CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODES_COUNT},
        graph,
        keepAll);
    EXPECT_EQ(indicesOf(byMode), (std::vector<std::int64_t>{0, 1, 2}));
}

TEST_F(TestCudnnShimV8Heuristics, CountThenFillInRankOrder)
{
    installHeuristicResults({2, 0, 1});
    fe::OperationGraph_v8 graph = convGraph();

    fe::EngineHeuristics_v8 heuristics = fe::EngineHeuristicsBuilder_v8()
                                             .setOperationGraph(graph)
                                             .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                             .build();
    ASSERT_EQ(heuristics.get_status(), CUDNN_STATUS_SUCCESS) << heuristics.get_error();

    const std::int64_t count = heuristics.getEngineConfigCount();
    EXPECT_EQ(count, 3);
    const fe::EngineConfigList& configs = heuristics.getEngineConfig(count);
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{2, 0, 1}));
}

TEST_F(TestCudnnShimV8Heuristics, SmCountTargetNotSupported)
{
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristicsBuilder_v8 builder;
    builder.setOperationGraph(graph).setHeurMode(CUDNN_HEUR_MODE_FALLBACK).setSMCount(8);

    const BuilderFailure failure
        = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET"), std::string::npos)
        << failure.what;

    // The default target (-1) is never sent.
    const fe::EngineHeuristics_v8 defaulted
        = fe::EngineHeuristicsBuilder_v8().setOperationGraph(graph).setSMCount(-1).build();
    EXPECT_EQ(defaulted.get_status(), CUDNN_STATUS_SUCCESS) << defaulted.get_error();
}

TEST_F(TestCudnnShimV8Heuristics, DevicePropertiesNotSupported)
{
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristicsBuilder_v8 builder;
    builder.setOperationGraph(graph)
        .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
        .setDeviceProperties(std::make_shared<const fe::DeviceProperties>());

    const BuilderFailure failure
        = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_ENGINEHEUR_DEVICEPROP"), std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8Heuristics, OpensourceModeOutOfRange)
{
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList configs;

    EXPECT_THROW(
        fe::get_heuristics_list(
            std::vector<fe::HeurMode_t>{fe::HeurMode_t::OPENSOURCE}, graph, keepAll, configs),
        std::out_of_range);
    EXPECT_TRUE(configs.empty());

    EXPECT_EQ(fe::get_heuristics_list(
                  std::vector<fe::HeurMode_t>{fe::HeurMode_t::FALLBACK}, graph, keepAll, configs),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(configs.size(), 3U);
}

TEST_F(TestCudnnShimV8Heuristics, FallbackListMatchesHeuristics)
{
    installHeuristicResults({1, 2, 0});
    fe::OperationGraph_v8 graph = convGraph();

    fe::EngineFallbackList_v8 fallback
        = fe::EngineFallbackListBuilder_v8()
              .setOperationGraph(graph)
              .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
              .build();

    ASSERT_EQ(fallback.get_status(), CUDNN_STATUS_SUCCESS) << fallback.get_error();
    EXPECT_EQ(indicesOf(fallback.getFallbackList()), indicesOf(engineConfigs(graph)));
    EXPECT_EQ(indicesOf(fallback.getFallbackList()), (std::vector<std::int64_t>{1, 2, 0}));
}

// Pinned against the NVIDIA table at CUDNN_VERSION 92200 (the minor <= 2
// branch); re-diff when the version pin moves. These numbers do not name
// shim engines.
TEST_F(TestCudnnShimV8Heuristics, FallbackEngineTablePinned)
{
    std::vector<int> convForward(50);
    std::iota(convForward.begin(), convForward.end(), 0);
    std::vector<int> convBackwardData(61);
    std::iota(convBackwardData.begin(), convBackwardData.end(), 0);

    EXPECT_EQ(fe::get_fallback_engine_list(
                  fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, ""),
              convForward);
    EXPECT_EQ(fe::get_fallback_engine_list(
                  fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, "ConvFwd_bias_"),
              (std::vector<int>{11, 0}));
    EXPECT_EQ(fe::get_fallback_engine_list(
                  fe::DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, ""),
              convBackwardData);
    EXPECT_EQ(fe::get_fallback_engine_list(
                  fe::DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, ""),
              (std::vector<int>{0, 1, 20}));
    EXPECT_TRUE(fe::get_fallback_engine_list(fe::DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR, "")
                    .empty());
}

} // namespace

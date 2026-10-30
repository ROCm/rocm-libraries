// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x heuristics and fallback-list builders against the mock hipDNN
// backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineFallbackList.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::AllowAll;
using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::failureOf;

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
        AllowAll,
        configs);

    // Both modes run the shim's single FALLBACK ranking, so the list repeats.
    EXPECT_EQ(statuses, (std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS}));
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));

    fe::EngineConfigList modeB;
    EXPECT_EQ(fe::get_heuristics_list(
                  std::vector<std::string>{"heuristics_mode_b"}, graph, AllowAll, modeB),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(modeB.size(), 3U);

    fe::EngineConfigList unknown;
    EXPECT_TRUE(fe::get_heuristics_list(
                    std::vector<std::string>{"heuristics_unknown"}, graph, AllowAll, unknown)
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
        AllowAll,
        configs);
    EXPECT_EQ(statuses, (std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS}));
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));

    // CUDNN_HEUR_MODES_COUNT entries are skipped.
    const fe::EngineConfigList byMode = fe::get_heuristics_list<2>(
        std::array<cudnnBackendHeurMode_t, 2>{CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODES_COUNT},
        graph,
        AllowAll);
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

    const BuildFailure failure
        = failureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("EngineHeuristicsBuilder: SM count target"), std::string::npos)
        << failure.what;

    // The default target (-1) and an all-SMs target (0) are never sent.
    for(const std::int32_t smCount : {-1, 0})
    {
        SCOPED_TRACE(smCount);
        const fe::EngineHeuristics_v8 defaulted
            = fe::EngineHeuristicsBuilder_v8().setOperationGraph(graph).setSMCount(smCount).build();
        EXPECT_EQ(defaulted.get_status(), CUDNN_STATUS_SUCCESS) << defaulted.get_error();
    }

    fe::EngineConfigList configs;
    EXPECT_EQ(fe::get_heuristics_list(std::vector<std::string>{"heuristics_fallback"},
                                      graph,
                                      AllowAll,
                                      configs,
                                      /*evaluate_all=*/false,
                                      /*sm_count=*/0),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(configs.size(), 3U);
}

// A failing mode stops the list unless evaluate_all is set; each run mode reports its status.
TEST_F(TestCudnnShimV8Heuristics, FailedModeStopsUnlessEvaluateAll)
{
    fe::OperationGraph_v8 graph = convGraph();
    const std::vector<std::string> modes{"heuristics_mode_a", "heuristics_fallback"};

    fe::EngineConfigList all;
    EXPECT_EQ(
        fe::get_heuristics_list(modes, graph, AllowAll, all, /*evaluate_all=*/true, /*sm_count=*/8),
        (std::vector<cudnnStatus_t>{CUDNN_STATUS_NOT_SUPPORTED, CUDNN_STATUS_NOT_SUPPORTED}));
    EXPECT_TRUE(all.empty());

    fe::EngineConfigList first;
    EXPECT_EQ(fe::get_heuristics_list(
                  modes, graph, AllowAll, first, /*evaluate_all=*/false, /*sm_count=*/8),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_NOT_SUPPORTED});
    EXPECT_TRUE(first.empty());
}

TEST_F(TestCudnnShimV8Heuristics, DevicePropertiesNotSupported)
{
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristicsBuilder_v8 builder;
    builder.setOperationGraph(graph)
        .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
        .setDeviceProperties(std::make_shared<const fe::DeviceProperties>());

    const BuildFailure failure
        = failureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("EngineHeuristicsBuilder: device properties"), std::string::npos)
        << failure.what;
}

// Upstream's mode map has no OPENSOURCE entry and throws std::out_of_range.
TEST_F(TestCudnnShimV8Heuristics, OpensourceModeRunsFallback)
{
    fe::OperationGraph_v8 graph = convGraph();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    fe::detail::warnedHeuristicModes().store(0);

    fe::EngineConfigList configs;
    EXPECT_EQ(
        fe::get_heuristics_list(
            std::vector<fe::HeurMode_t>{fe::HeurMode_t::OPENSOURCE}, graph, AllowAll, configs),
        std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2}));
    const auto logs = recorder.getRecordedLogs();
    EXPECT_EQ(std::count_if(logs.begin(),
                            logs.end(),
                            [](const auto& log) {
                                return log.severity == HIPDNN_SEV_WARN
                                       && log.message.find("heuristic mode OPENSOURCE ")
                                              != std::string::npos;
                            }),
              1);

    fe::EngineConfigList fallback;
    EXPECT_EQ(fe::get_heuristics_list(
                  std::vector<fe::HeurMode_t>{fe::HeurMode_t::FALLBACK}, graph, AllowAll, fallback),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_SUCCESS});
    EXPECT_EQ(fallback.size(), 3U);
}

TEST_F(TestCudnnShimV8Heuristics, UnknownHeurModeRejected)
{
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList configs;
    const BuildFailure failure = failureOf([&] {
        [[maybe_unused]] const std::vector<cudnnStatus_t> statuses = fe::get_heuristics_list(
            std::vector<fe::HeurMode_t>{fe::HeurMode_t::FALLBACK, static_cast<fe::HeurMode_t>(99)},
            graph,
            AllowAll,
            configs);
    });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(configs.empty());
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

// Binding a build() result by value must leave the builder able to build again.
TEST_F(TestCudnnShimV8Heuristics, FallbackListBuilderBuildsTwice)
{
    installHeuristicResults({1, 2, 0});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineFallbackListBuilder_v8 builder;
    builder.setOperationGraph(graph).setOperation(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);

    fe::EngineFallbackList_v8 first = builder.build();
    ASSERT_EQ(first.get_status(), CUDNN_STATUS_SUCCESS) << first.get_error();

    fe::EngineFallbackList_v8 second = builder.build();
    ASSERT_EQ(second.get_status(), CUDNN_STATUS_SUCCESS) << second.get_error();
    EXPECT_EQ(indicesOf(second.getFallbackList()), (std::vector<std::int64_t>{1, 2, 0}));
    EXPECT_EQ(indicesOf(first.getFallbackList()), (std::vector<std::int64_t>{1, 2, 0}));
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

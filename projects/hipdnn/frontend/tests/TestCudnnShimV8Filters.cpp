// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x engine-config filters against the shim's backend C API and
// the mock hipDNN backend, including PyTorch's quantized-op use of them.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Filters.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <cstdint>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

class TestCudnnShimV8Filters : public hipdnn_shim_test::ShimV8PlanFixture
{
};

TEST_F(TestCudnnShimV8Filters, FilterKeepsFalsePredicate)
{
    installEngines(4);
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList from = engineConfigs(graph);
    fe::EngineConfigList to = engineConfigs(graph);
    to.resize(1);

    fe::filter(from, to, [](cudnnBackendDescriptor_t config) { return indexOf(config) % 2 == 1; });

    // Kept configs are appended after what `to` held; rejected ones stay in
    // `from` in their original order.
    EXPECT_EQ(indicesOf(to), (std::vector<std::int64_t>{0, 0, 2}));
    EXPECT_EQ(indicesOf(from), (std::vector<std::int64_t>{1, 3}));
}

// PyTorch's quantized ops filter into a new list but then try the configs
// left in the source list, i.e. the rejected ones.
TEST_F(TestCudnnShimV8Filters, QuantizedLoopSeesRejectedSet)
{
    const std::int64_t deterministic = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    const std::vector<std::int64_t> ids = {9, deterministic, 10};
    installHeuristicResults(ids);
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList engineConfigsList = engineConfigs(graph);
    fe::EngineConfigList filtered;

    fe::filter(engineConfigsList, filtered, [](cudnnBackendDescriptor_t config) {
        return fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config)
               || fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(config)
               || fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(config);
    });

    EXPECT_EQ(indicesOf(filtered), std::vector<std::int64_t>{denseIndex(ids, deterministic)});
    EXPECT_EQ(indicesOf(engineConfigsList),
              (std::vector<std::int64_t>{denseIndex(ids, 9), denseIndex(ids, 10)}));
}

TEST_F(TestCudnnShimV8Filters, TensorCoreNoteNeverReported)
{
    installEngines(2);
    fe::OperationGraph_v8 graph = convGraph();

    for(const fe::ManagedOpaqueDescriptor& config : engineConfigs(graph))
    {
        cudnnBackendDescriptor_t raw = config->get_backend_descriptor();
        EXPECT_TRUE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(raw));
        EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(raw));
        EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(raw));
    }
}

TEST_F(TestCudnnShimV8Filters, BehaviorNoteMapped)
{
    installEngines(2);
    installBehaviorNotes(0,
                         {HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION,
                          HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 2U);

    EXPECT_TRUE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(
        configs[0]->get_backend_descriptor()));
    EXPECT_FALSE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(
        configs[1]->get_backend_descriptor()));
    EXPECT_FALSE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER>(
        configs[0]->get_backend_descriptor()));
}

} // namespace

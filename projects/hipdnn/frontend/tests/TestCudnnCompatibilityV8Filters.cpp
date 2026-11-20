// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x engine-config filters over the mock hipDNN backend, including
// PyTorch's quantized-op use of them.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Filters.h>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "fake_backend/CompatibilityV8PlanFixture.hpp"

#include <cstdint>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

class TestCudnnCompatibilityV8Filters : public hipdnn_shim_test::CompatibilityV8PlanFixture
{
};

TEST_F(TestCudnnCompatibilityV8Filters, FilterKeepsFalsePredicate)
{
    installEngines(4);
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList from = engineConfigs(graph);
    fe::EngineConfigList to = engineConfigs(graph);
    to.resize(1);

    fe::filter(
        from, to, [this](cudnnBackendDescriptor_t config) { return indexOf(config) % 2 == 1; });

    // Kept configs are appended after what `to` held; rejected ones stay in
    // `from` in their original order.
    EXPECT_EQ(indicesOf(to), (std::vector<std::int64_t>{0, 0, 2}));
    EXPECT_EQ(indicesOf(from), (std::vector<std::int64_t>{1, 3}));
}

// The index helpers the expectations here rely on fail on a config or engine outside the
// engine space instead of reading it as some index.
TEST_F(TestCudnnCompatibilityV8Filters, IndexHelpersRejectForeignConfigs)
{
    installEngines(2);
    ::testing::TestPartResultArray failures;
    std::int64_t foreign = 0;
    std::int64_t absent = 0;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        foreign = indexOf(newMockDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR));
        absent = denseIndex({0, 2}, 1);
    }
    EXPECT_EQ(foreign, -1);
    EXPECT_EQ(absent, -1);
    EXPECT_EQ(failures.size(), 2);
}

// PyTorch's quantized ops filter into a new list but then try the configs
// left in the source list, i.e. the rejected ones.
TEST_F(TestCudnnCompatibilityV8Filters, QuantizedLoopSeesRejectedSet)
{
    const std::int64_t deterministic = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    const std::vector<std::int64_t> ids = {9, deterministic, 10};
    installHeuristicResults(ids);
    installNumericalNotes(deterministic, {});
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

// An engine that declares no numerical notes is reported NONDETERMINISTIC by hipDNN.
TEST_F(TestCudnnCompatibilityV8Filters, UndeclaredEngineReportsNondeterministicOnly)
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

TEST_F(TestCudnnCompatibilityV8Filters, NumericalNoteMapped)
{
    installEngines(3);
    installNumericalNotes(0, {HIPDNN_NUMERICAL_NOTE_TENSOR_CORE, HIPDNN_NUMERICAL_NOTE_FFT});
    installNumericalNotes(1, {});
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);

    EXPECT_TRUE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(
        configs[0]->get_backend_descriptor()));
    EXPECT_TRUE(
        fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_FFT>(configs[0]->get_backend_descriptor()));
    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
        configs[0]->get_backend_descriptor()));
    // An explicit empty list is a claim, not the absent field.
    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
        configs[1]->get_backend_descriptor()));
    EXPECT_TRUE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
        configs[2]->get_backend_descriptor()));
}

// A value this layer has no cuDNN counterpart for is dropped, never aliased onto a note.
TEST_F(TestCudnnCompatibilityV8Filters, UnknownNumericalNoteIsDropped)
{
    installEngines(1);
    installNumericalNotes(
        0, {static_cast<hipdnnBackendNumericalNote_t>(77), HIPDNN_NUMERICAL_NOTE_WINOGRAD});
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 1U);

    EXPECT_TRUE(
        fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_WINOGRAD>(configs[0]->get_backend_descriptor()));
    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
        configs[0]->get_backend_descriptor()));
}

// A failed query answers false, as upstream's helpers do, so the engine passes a filter that
// rejects NONDETERMINISTIC.
TEST_F(TestCudnnCompatibilityV8Filters, FailedNumericalNoteQueryAnswersFalse)
{
    installEngines(1);
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 1U);
    failNumericalNoteQueries(HIPDNN_STATUS_INTERNAL_ERROR);

    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
        configs[0]->get_backend_descriptor()));
}

TEST_F(TestCudnnCompatibilityV8Filters, BehaviorNoteMapped)
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

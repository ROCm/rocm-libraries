// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

static_assert(std::is_enum_v<cudnn_frontend::KnobType_t>);
static_assert(!std::is_same_v<cudnn_frontend::Knob, hipdnn_frontend::Knob>);
static_assert(!std::is_same_v<cudnn_frontend::KnobType_t, hipdnn_frontend::KnobType_t>);
static_assert(cudnn_frontend::Knob{}.type == cudnn_frontend::KnobType_t::NOT_SET);
static_assert(std::is_same_v<cudaGraph_t, void*>);

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

void expectNoPlanError(const fe::error_t& error)
{
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_EQ(error.get_message(), "Graph has no compiled execution plan");
}

void expectGraphStubError(const fe::error_t& error, const std::string& methodName)
{
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(error.get_message().find(methodName), std::string::npos);
}

TEST(TestCudnnShimPlanSurface, EmptyGraphIntrospectionAndPlanMethodsUseNoPlanGuard)
{
    fe::graph::Graph graph;

    int64_t engineCount = -1;
    auto engineCountError = graph.get_engine_count(engineCount);
    EXPECT_TRUE(engineCountError.is_good()) << engineCountError.get_message();
    EXPECT_EQ(engineCount, 0);
    std::vector<fe::Knob> knobs = {fe::Knob{fe::KnobType_t::TILE_SIZE, 256, 1, 1}};
    expectNoPlanError(graph.get_knobs_for_engine(7, knobs));
    EXPECT_TRUE(knobs.empty());

    const std::unordered_map<fe::KnobType_t, int64_t> knobSettings
        = {{fe::KnobType_t::TILE_SIZE, 1}};
    expectNoPlanError(graph.create_execution_plan(7, knobSettings));

    std::string planName = "unchanged";
    expectNoPlanError(graph.get_plan_name(planName));
    EXPECT_EQ(planName, "unchanged");

    planName = "unchanged";
    expectNoPlanError(graph.get_plan_name_at_index(0, planName));
    EXPECT_EQ(planName, "unchanged");

    int64_t workspaceSize = -1;
    expectNoPlanError(graph.get_workspace_size_plan_at_index(0, workspaceSize));
    EXPECT_EQ(workspaceSize, -1);

    std::unordered_map<int64_t, void*> uidMap;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    expectNoPlanError(graph.execute_plan_at_index(nullptr, uidMap, nullptr, 0));
    expectNoPlanError(graph.execute_plan_at_index(nullptr, tensorMap, nullptr, 0));
    expectNoPlanError(graph.autotune(nullptr, uidMap, nullptr));
    expectNoPlanError(graph.autotune(nullptr, tensorMap, nullptr));
    expectNoPlanError(graph.warmup(nullptr, uidMap, nullptr));
    expectNoPlanError(graph.warmup(nullptr, tensorMap, nullptr));

    std::vector<fe::BehaviorNote_t> behaviorNotes = {fe::BehaviorNote_t::RUNTIME_COMPILATION};
    expectNoPlanError(graph.get_behavior_notes(behaviorNotes));
    EXPECT_TRUE(behaviorNotes.empty());

    behaviorNotes = {fe::BehaviorNote_t::RUNTIME_COMPILATION};
    expectNoPlanError(graph.get_behavior_notes_for_plan_at_index(0, behaviorNotes));
    EXPECT_TRUE(behaviorNotes.empty());
}

TEST(TestCudnnShimPlanSurface, CudaGraphStubsReturnNamedUnsupportedErrors)
{
    const fe::graph::Graph graph;
    std::unordered_map<int64_t, void*> uidMap;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    cudaGraph_t cudaGraph = nullptr;

    expectGraphStubError(graph.populate_cuda_graph(nullptr, uidMap, nullptr, cudaGraph),
                         "populate_cuda_graph");
    expectGraphStubError(graph.populate_cuda_graph(nullptr, tensorMap, nullptr, cudaGraph),
                         "populate_cuda_graph");
    expectGraphStubError(graph.update_cuda_graph(nullptr, uidMap, nullptr, cudaGraph),
                         "update_cuda_graph");
    expectGraphStubError(graph.update_cuda_graph(nullptr, tensorMap, nullptr, cudaGraph),
                         "update_cuda_graph");
}

TEST(TestCudnnShimPlanSurface, TensorKeyedExecutePlanRejectsNullTensorBeforeNoPlan)
{
    const fe::graph::Graph graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    tensorMap.emplace(nullptr, nullptr);

    auto error = graph.execute_plan_at_index(nullptr, tensorMap, nullptr, 0);

    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("requires every tensor to have a UID"), std::string::npos);
    EXPECT_EQ(error.get_message().find("no compiled execution plan"), std::string::npos);
}

TEST(TestCudnnShimPlanSurface, TensorKeyedExecutePlanRejectsTensorWithoutUidBeforeNoPlan)
{
    const fe::graph::Graph graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    tensorMap.emplace(std::make_shared<fe::graph::Tensor_attributes>(), nullptr);

    auto error = graph.execute_plan_at_index(nullptr, tensorMap, nullptr, 0);

    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("requires every tensor to have a UID"), std::string::npos);
    EXPECT_EQ(error.get_message().find("no compiled execution plan"), std::string::npos);
}

} // namespace

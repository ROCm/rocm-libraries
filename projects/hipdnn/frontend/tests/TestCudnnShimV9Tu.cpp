// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only v9 translation unit that includes only the cuDNN-compatibility
// shim umbrella header plus test/standard headers. Referencing the public
// symbols below catches source-compatibility regressions in the hipified cuDNN
// frontend surface.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Mirror the consumer hipification step: alias the namespace so in-code
// `cudnn_frontend::` symbols resolve to the shim.
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

// FE-namespace enums resolve through the shim aliases and are the *same* enum
// as the hipDNN type (aliased, not re-declared).
static_assert(std::is_same_v<cudnn_frontend::DataType_t, hipdnn_frontend::DataType_t>,
              "cudnn_frontend::DataType_t must alias the hipDNN enum");
static_assert(std::is_enum_v<cudnn_frontend::KnobType_t>,
              "cudnn_frontend::KnobType_t must be an enum");
static_assert(std::is_same_v<decltype(cudnn_frontend::Knob{}.type), cudnn_frontend::KnobType_t>,
              "cudnn_frontend::Knob must expose a type field");
static_assert(std::is_same_v<decltype(cudnn_frontend::Knob{}.maxValue), int64_t>,
              "cudnn_frontend::Knob must expose an int64_t maxValue field");
static_assert(std::is_same_v<decltype(cudnn_frontend::Knob{}.minValue), int64_t>,
              "cudnn_frontend::Knob must expose an int64_t minValue field");
static_assert(std::is_same_v<decltype(cudnn_frontend::Knob{}.stride), int64_t>,
              "cudnn_frontend::Knob must expose an int64_t stride field");
static_assert(cudnn_frontend::Knob{}.type == cudnn_frontend::KnobType_t::NOT_SET,
              "cudnn_frontend::Knob default type must be NOT_SET");
static_assert(cudnn_frontend::Knob{}.maxValue == 0,
              "cudnn_frontend::Knob default maxValue must be zero");
static_assert(cudnn_frontend::Knob{}.minValue == 0,
              "cudnn_frontend::Knob default minValue must be zero");
static_assert(cudnn_frontend::Knob{}.stride == 0,
              "cudnn_frontend::Knob default stride must be zero");

namespace
{
constexpr cudnn_frontend::DataType_t K_IO_TYPE = cudnn_frontend::DataType_t::HALF;
constexpr cudnn_frontend::DataType_t K_COMPUTE_TYPE = cudnn_frontend::DataType_t::FLOAT;
constexpr cudnn_frontend::HeurMode_t K_HEUR_MODE = cudnn_frontend::HeurMode_t::A;
constexpr cudnn_frontend::NumericalNote_t K_NUMERICAL_NONDETERMINISTIC
    = cudnn_frontend::NumericalNote_t::NONDETERMINISTIC;
constexpr cudnn_frontend::BehaviorNote_t K_BEHAVIOR_CUBLASLT_DEPENDENCY
    = cudnn_frontend::BehaviorNote_t::CUBLASLT_DEPENDENCY;
constexpr cudnn_frontend::KnobType_t K_KNOB_TILE_SIZE = cudnn_frontend::KnobType_t::TILE_SIZE;

// error_t / error_code_t resolve through the shim aliases.
cudnn_frontend::error_t makeOk()
{
    return {};
}

// §4.7 C-API types come from the stub <cudnn.h> pulled in by the umbrella.
cudnnStatus_t statusFor(cudnnHandle_t /*handle*/)
{
    return CUDNN_STATUS_SUCCESS;
}

TEST(TestCudnnShimV9Tu, V9OnlyTranslationUnitCompilesAndLinks)
{
    cudnn_frontend::graph::Graph graph;

    EXPECT_TRUE(makeOk().is_good());
    EXPECT_EQ(K_IO_TYPE, cudnn_frontend::DataType_t::HALF);
    EXPECT_EQ(K_COMPUTE_TYPE, cudnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(K_HEUR_MODE, cudnn_frontend::HeurMode_t::A);
    EXPECT_EQ(statusFor(nullptr), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(CUDNN_FRONTEND_VERSION, 12400);

    SUCCEED() << "v9-only TU compiled and linked against the shim umbrella.";
}

TEST(TestCudnnShimV9Tu, Phase4PublicSurfaceCompiles)
{
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    const std::vector<fe::NumericalNote_t> selectedNumericalNotes = {K_NUMERICAL_NONDETERMINISTIC};
    const std::vector<fe::NumericalNote_t> deselectedNumericalNotes
        = {fe::NumericalNote_t::NOT_SET};
    const std::vector<fe::BehaviorNote_t> behaviorNotes = {K_BEHAVIOR_CUBLASLT_DEPENDENCY};
    const std::vector<std::string> engineNames;
    const std::vector<int64_t> engineIndices;

    EXPECT_EQ(&graph.select_numeric_notes(selectedNumericalNotes), &graph);
    EXPECT_EQ(&graph.deselect_numeric_notes(deselectedNumericalNotes), &graph);
    EXPECT_EQ(&graph.select_behavior_notes(behaviorNotes), &graph);
    EXPECT_EQ(&graph.deselect_behavior_notes(behaviorNotes), &graph);
    EXPECT_EQ(&graph.deselect_workspace_greater_than(0), &graph);
    EXPECT_EQ(&graph.deselect_shared_mem_greater_than(0), &graph);
    EXPECT_EQ(&graph.deselect_engines(engineNames), &graph);
    EXPECT_EQ(&graph.deselect_engines(engineIndices), &graph);

    int64_t engineCount = -1;
    static_cast<void>(graph.get_engine_count(engineCount));

    std::vector<fe::Knob> knobs;
    static_cast<void>(graph.get_knobs_for_engine(0, knobs));

    const std::unordered_map<fe::KnobType_t, int64_t> knobSettings = {{K_KNOB_TILE_SIZE, 1}};
    static_cast<void>(graph.create_execution_plan(0, knobSettings));

    std::string planName;
    static_cast<void>(graph.get_plan_name(planName));
    static_cast<void>(graph.get_plan_name_at_index(0, planName));

    int64_t workspaceSize = 0;
    static_cast<void>(graph.get_workspace_size_plan_at_index(0, workspaceSize));

    std::unordered_map<int64_t, void*> uidMap;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    void* workspace = nullptr;
    cudaGraph_t cudaGraph = nullptr;

    static_cast<void>(graph.execute_plan_at_index(nullptr, uidMap, workspace, 0));
    static_cast<void>(graph.execute_plan_at_index(nullptr, tensorMap, workspace, 0));
    static_cast<void>(graph.autotune(nullptr, uidMap, workspace));
    static_cast<void>(graph.autotune(nullptr, tensorMap, workspace));
    static_cast<void>(graph.warmup(nullptr, uidMap, workspace));
    static_cast<void>(graph.warmup(nullptr, tensorMap, workspace));
    static_cast<void>(graph.populate_cuda_graph(nullptr, uidMap, workspace, cudaGraph));
    static_cast<void>(graph.populate_cuda_graph(nullptr, tensorMap, workspace, cudaGraph));
    static_cast<void>(graph.update_cuda_graph(nullptr, uidMap, workspace, cudaGraph));
    static_cast<void>(graph.update_cuda_graph(nullptr, tensorMap, workspace, cudaGraph));

    SUCCEED() << "Phase 4 public surface compiled against empty/no-op graph data.";
}

} // namespace

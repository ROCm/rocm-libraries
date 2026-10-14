// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Host-only coverage for the shim's public note-filter behavior.
#include "CudnnShimTestSupport.hpp"
#include "fake_backend/MockBackendFixture.hpp"

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using NumNote = fe::NumericalNote_t;
using BehNote = fe::BehaviorNote_t;
using hipdnn_shim_test::expectGraphNotSupported;

using ::testing::_;
using ::testing::AnyNumber;

// A plugin-style engine id with no registered name.
constexpr int64_t UNREGISTERED_ENGINE_ID = 0x0123456789ABCDEF;

void addPointwiseGraph(fe::graph::Graph& graph)
{
    const int64_t n = 4;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 1);
    auto b = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 2);
    auto c = graph.pointwise(
        a, b, fe::graph::Pointwise_attributes{}.set_mode(fe::PointwiseMode_t::ADD));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);
}

class TestCudnnShimNoteTriageBackend : public hipdnn_shim_test::ShimMockBackendFixture
{
protected:
    std::vector<hipdnnBackendDescriptor_t> _executionPlanDescs;
    std::unordered_map<hipdnnBackendDescriptor_t, int64_t> _engineIdsByDesc;
    std::unordered_map<int64_t, std::vector<hipdnnBackendBehaviorNote_t>> _behaviorNotesByEngineId;
    std::vector<int64_t> _rankedEngineIds{10, 20};
    std::array<char, 16> _engineDescs{};
    size_t _nextEngineDesc = 0;
    size_t _nextEngineIdLookup = 0;

    void installRankedEnginePlanMocks()
    {
        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault(
                [this](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* desc) {
                    *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                        &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
                    if(type == HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
                    {
                        _executionPlanDescs.push_back(*desc);
                    }
                    return HIPDNN_STATUS_SUCCESS;
                });

        // A behavior-note query builds a throwaway engine descriptor, so the id
        // the caller stamps on it is the only way back to the engine it stands for.
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor,
                                  hipdnnBackendAttributeName_t attribute,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  const void* arrayOfElements) {
                if(attribute == HIPDNN_ATTR_ENGINE_GLOBAL_INDEX && arrayOfElements != nullptr)
                {
                    _engineIdsByDesc[descriptor] = *static_cast<const int64_t*>(arrayOfElements);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

        ON_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor,
                                  hipdnnBackendAttributeName_t attribute,
                                  hipdnnBackendAttributeType_t,
                                  int64_t requestedElementCount,
                                  int64_t* elementCount,
                                  void* arrayOfElements) {
                if(attribute == HIPDNN_ATTR_ENGINEHEUR_RESULTS)
                {
                    // Each ranked-engine pass starts with this count query, so
                    // restarting here gives every pass the same id order.
                    if(requestedElementCount == 0)
                    {
                        _nextEngineIdLookup = 0;
                    }
                    if(elementCount != nullptr)
                    {
                        *elementCount = requestedElementCount == 0
                                            ? static_cast<int64_t>(_rankedEngineIds.size())
                                            : requestedElementCount;
                    }
                    return HIPDNN_STATUS_SUCCESS;
                }
                if(attribute == HIPDNN_ATTR_ENGINECFG_ENGINE)
                {
                    const int64_t engineId
                        = _rankedEngineIds[_nextEngineIdLookup++ % _rankedEngineIds.size()];
                    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                        &_engineDescs[_nextEngineDesc++ % _engineDescs.size()]);
                    _engineIdsByDesc[engineDesc] = engineId;
                    *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
                    return HIPDNN_STATUS_SUCCESS;
                }
                if(attribute == HIPDNN_ATTR_ENGINE_GLOBAL_INDEX)
                {
                    *static_cast<int64_t*>(arrayOfElements) = _engineIdsByDesc.at(descriptor);
                    return HIPDNN_STATUS_SUCCESS;
                }
                if(attribute == HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE)
                {
                    const auto* notes = behaviorNotesForDesc(descriptor);
                    const int64_t noteCount
                        = notes == nullptr ? 0 : static_cast<int64_t>(notes->size());
                    if(notes != nullptr && arrayOfElements != nullptr)
                    {
                        std::copy_n(notes->begin(),
                                    std::min(requestedElementCount, noteCount),
                                    static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements));
                    }
                    if(elementCount != nullptr)
                    {
                        *elementCount = noteCount;
                    }
                    return HIPDNN_STATUS_SUCCESS;
                }
                if(attribute == HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)
                {
                    *static_cast<int64_t*>(arrayOfElements) = 0;
                    return HIPDNN_STATUS_SUCCESS;
                }
                if(elementCount != nullptr)
                {
                    *elementCount = 0;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    // Tests that assert a plan by name need registered engines.
    void installRegisteredEnginePlanMocks()
    {
        _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                            hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID};
        installRankedEnginePlanMocks();
    }

    void createPlans(fe::graph::Graph& graph)
    {
        addPointwiseGraph(graph);
        ASSERT_TRUE(graph.validate().is_good());
        ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
        ASSERT_TRUE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    }

    const std::vector<hipdnnBackendBehaviorNote_t>*
        behaviorNotesForDesc(hipdnnBackendDescriptor_t descriptor) const
    {
        const auto engineEntry = _engineIdsByDesc.find(descriptor);
        if(engineEntry == _engineIdsByDesc.end())
        {
            return nullptr;
        }
        const auto notesEntry = _behaviorNotesByEngineId.find(engineEntry->second);
        return notesEntry == _behaviorNotesByEngineId.end() ? nullptr : &notesEntry->second;
    }
};

TEST(TestCudnnShimNoteTriage, DeselectNondeterministicDoesNotPoisonValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, DeselectReducedPrecisionReductionPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::REDUCED_PRECISION_REDUCTION});

    expectGraphNotSupported(graph.validate(), "REDUCED_PRECISION_REDUCTION");
}

TEST(TestCudnnShimNoteTriage, DeselectDownConvertInputsPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::DOWN_CONVERT_INPUTS});

    expectGraphNotSupported(graph.validate(), "DOWN_CONVERT_INPUTS");
}

TEST(TestCudnnShimNoteTriage, SelectStrictNanPropPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.select_numeric_notes({NumNote::STRICT_NAN_PROP});

    expectGraphNotSupported(graph.validate(), "STRICT_NAN_PROP");
}

TEST(TestCudnnShimNoteTriage, OutOfRangeNumericalNotePoisonsValidate)
{
    const auto outOfRange = static_cast<NumNote>(1000);

    fe::graph::Graph selectGraph;
    selectGraph.select_numeric_notes({outOfRange});
    expectGraphNotSupported(selectGraph.validate(), "unknown (1000)");

    fe::graph::Graph deselectGraph;
    deselectGraph.deselect_numeric_notes({outOfRange});
    expectGraphNotSupported(deselectGraph.validate(), "unknown (1000)");
}

TEST(TestCudnnShimNoteTriage, SelectDeterminismAndPrecisionNotesLeavesGraphUsable)
{
    fe::graph::Graph graph;
    graph.select_numeric_notes({NumNote::NONDETERMINISTIC, NumNote::REDUCED_PRECISION_REDUCTION});

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, DeselectAdvisoryNumericNotesLeavesGraphUsable)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::TENSOR_CORE, NumNote::WINOGRAD, NumNote::FFT});

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, KnownBehaviorNoteFiltersLeaveEmptyGraphValid)
{
    const std::vector<BehNote> notes = {BehNote::RUNTIME_COMPILATION,
                                        BehNote::REQUIRES_LAYOUT_TRANSFORM,
                                        BehNote::SUPPORTS_GRAPH_CAPTURE,
                                        BehNote::EXTERNAL_LIBRARY_DEPENDENCY,
                                        BehNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION};

    fe::graph::Graph selectGraph;
    EXPECT_EQ(&selectGraph.select_behavior_notes(notes), &selectGraph);
    EXPECT_TRUE(selectGraph.validate().is_good());

    fe::graph::Graph deselectGraph;
    EXPECT_EQ(&deselectGraph.deselect_behavior_notes(notes), &deselectGraph);
    EXPECT_TRUE(deselectGraph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, SelectCudnnOnlyBehaviorNotePoisonsValidate)
{
    const std::vector<BehNote> notes = {BehNote::REQUIRES_FILTER_INT8x32_REORDER,
                                        BehNote::REQUIRES_BIAS_INT8x32_REORDER,
                                        BehNote::SUPPORTS_CUDA_GRAPH_NATIVE_API,
                                        BehNote::CUBLASLT_DEPENDENCY};

    for(const auto note : notes)
    {
        fe::graph::Graph graph;
        EXPECT_EQ(&graph.select_behavior_notes({note}), &graph);
        expectGraphNotSupported(graph.validate(), hipdnn_frontend::to_string(note));
    }
}

TEST(TestCudnnShimNoteTriage, DeselectCudnnOnlyBehaviorNoteWarnsAndLeavesGraphUsable)
{
    const std::vector<BehNote> notes = {BehNote::REQUIRES_FILTER_INT8x32_REORDER,
                                        BehNote::REQUIRES_BIAS_INT8x32_REORDER,
                                        BehNote::SUPPORTS_CUDA_GRAPH_NATIVE_API,
                                        BehNote::CUBLASLT_DEPENDENCY};

    fe::graph::Graph graph;
    EXPECT_EQ(&graph.deselect_behavior_notes(notes), &graph);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, ResourceFiltersChainWithoutPoisoningEmptyGraph)
{
    fe::graph::Graph graph;

    EXPECT_EQ(&graph.deselect_workspace_greater_than(1024), &graph);
    EXPECT_EQ(&graph.deselect_engines(std::vector<std::string>{"MIOPEN_ENGINE"}), &graph);
    EXPECT_EQ(&graph.deselect_engines(std::vector<int64_t>{1, 2}), &graph);

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, ZeroSharedMemoryFilterIsNoOp)
{
    fe::graph::Graph graph;

    EXPECT_EQ(&graph.deselect_shared_mem_greater_than(0), &graph);

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, NonzeroSharedMemoryFilterPoisonsValidate)
{
    fe::graph::Graph graph;

    EXPECT_EQ(&graph.deselect_shared_mem_greater_than(1), &graph);
    expectGraphNotSupported(graph.validate(), "shared-memory metadata");
}

TEST(TestCudnnShimNoteTriage, EmptyAndNotSetNoteVectorsAreNoOps)
{
    fe::graph::Graph graph;
    graph.select_numeric_notes({})
        .deselect_numeric_notes({})
        .select_behavior_notes({})
        .deselect_behavior_notes({})
        .select_numeric_notes({NumNote::NOT_SET})
        .deselect_numeric_notes({NumNote::NOT_SET})
        .select_behavior_notes({BehNote::NOT_SET})
        .deselect_behavior_notes({BehNote::NOT_SET});

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, FiltersReturnSameGraphForChaining)
{
    fe::graph::Graph graph;

    EXPECT_EQ(&graph.select_numeric_notes({}), &graph);
    EXPECT_EQ(&graph.deselect_numeric_notes({}), &graph);
    EXPECT_EQ(&graph.select_behavior_notes({}), &graph);
    EXPECT_EQ(&graph.deselect_behavior_notes({}), &graph);

    graph.set_name("chain")
        .select_numeric_notes({NumNote::TENSOR_CORE})
        .deselect_numeric_notes({NumNote::WINOGRAD})
        .select_behavior_notes({BehNote::RUNTIME_COMPILATION})
        .deselect_behavior_notes({BehNote::CUBLASLT_DEPENDENCY});

    EXPECT_EQ(graph.get_name(), "chain");
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimNoteTriage, ErrorNoteAfterAdvisoryNoteStillPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::TENSOR_CORE, NumNote::REDUCED_PRECISION_REDUCTION});

    expectGraphNotSupported(graph.validate(), "REDUCED_PRECISION_REDUCTION");
}

TEST(TestCudnnShimNoteTriage, FirstRecordedNoteErrorWins)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::REDUCED_PRECISION_REDUCTION});
    graph.deselect_numeric_notes({NumNote::DOWN_CONVERT_INPUTS});

    auto err = graph.validate();

    expectGraphNotSupported(err, "REDUCED_PRECISION_REDUCTION");
    EXPECT_EQ(err.get_message().find("DOWN_CONVERT_INPUTS"), std::string::npos);
}

TEST(TestCudnnShimNoteTriage, CreateExecutionPlansAcceptsUnhonoredHeurModes)
{
    const std::vector<std::vector<fe::HeurMode_t>> modeCases
        = {{fe::HeurMode_t::A},
           {fe::HeurMode_t::FALLBACK},
           {fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::OPENSOURCE},
           {}};

    for(const auto& modes : modeCases)
    {
        fe::graph::Graph graph;
        EXPECT_TRUE(graph.create_execution_plans(modes).is_good());
        EXPECT_TRUE(graph.validate().is_good());
    }
}

TEST(TestCudnnShimNoteTriage, CreateExecutionPlansStillSurfacesRecordedNoteError)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::REDUCED_PRECISION_REDUCTION});

    expectGraphNotSupported(graph.create_execution_plans({fe::HeurMode_t::A}),
                            "REDUCED_PRECISION_REDUCTION");
}

// T-U6 regression: index-based deselect_engines on a Native (unbuilt) graph must
// only store the indices and defer translation until after the plan is built.
// Pre-fix, it eagerly mapped indices via get_ranked_engine_ids on the unbuilt
// graph — there is no ranked engine list before build — which recorded an error
// that later poisoned validate(). Host-only: no build/create_execution_plans.
TEST(TestCudnnShimNoteTriage, DeselectEngineIndicesBeforeBuildDoesNotPoisonNativeGraph)
{
    const int64_t n = 16;
    const int64_t c = 128;
    const int64_t h = 64;
    const int64_t w = 64;
    const int64_t k = 256;
    const int64_t r = 1;
    const int64_t s = 1;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("image")
                              .set_dim({n, c, h, w})
                              .set_stride({c * h * w, 1, c * w, c})
                              .set_uid(1));
    auto weight = graph.tensor(fe::graph::Tensor_attributes{}
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c})
                                   .set_uid(2));

    // conv_fprop makes the graph Mode::Native.
    auto y = graph.conv_fprop(
        x,
        weight,
        fe::graph::Conv_fprop_attributes{}.set_padding({0, 0}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    ASSERT_NE(y, nullptr);
    y->set_output(true).set_uid(3);

    EXPECT_EQ(&graph.deselect_engines(std::vector<int64_t>{0, 1}), &graph);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectEngineIndicesAfterPlanCreationAppliesBeforeBuildAll)
{
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    EXPECT_EQ(&graph.deselect_engines(std::vector<int64_t>{0}), &graph);

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_executionPlanDescs[0],
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .Times(0);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_executionPlanDescs[1],
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .Times(1);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::ALL);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), 0);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectEngineIndicesAfterPlanCreationBarsBuildPlanAtIndex)
{
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    EXPECT_EQ(&graph.deselect_engines(std::vector<int64_t>{0}), &graph);

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_executionPlanDescs[0],
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .Times(0);

    auto err = graph.build_plan_at_index(0);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_NE(err.get_message().find("barred"), std::string::npos);
}

// Regression: create_execution_plans() pins the active plan before the shim's
// filters run, and native build_plans(HEURISTICS_CHOICE) inspects only that
// plan — so barring the top-ranked engine used to turn a satisfiable build into
// an INVALID_VALUE failure. cuDNN narrows the candidate set instead.
TEST_F(TestCudnnShimNoteTriageBackend, DeselectBehaviorNoteOnTopEngineNarrowsToSurvivingPlan)
{
    installRegisteredEnginePlanMocks();
    _behaviorNotesByEngineId[hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID]
        = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    EXPECT_EQ(&graph.deselect_behavior_notes({BehNote::RUNTIME_COMPILATION}), &graph);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_good()) << err.get_message();

    std::string planName;
    EXPECT_TRUE(graph.get_plan_name(planName).is_good());
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_NAME);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), 0);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectBehaviorNoteOnEveryEngineFailsBuild)
{
    installRegisteredEnginePlanMocks();
    _behaviorNotesByEngineId[hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID]
        = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};
    _behaviorNotesByEngineId[hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID]
        = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    EXPECT_EQ(&graph.deselect_behavior_notes({BehNote::RUNTIME_COMPILATION}), &graph);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("behavior notes"), std::string::npos);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), -1);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectTopEngineIndexNarrowsToSurvivingPlan)
{
    installRegisteredEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    EXPECT_EQ(&graph.deselect_engines(std::vector<int64_t>{0}), &graph);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_good()) << err.get_message();

    std::string planName;
    EXPECT_TRUE(graph.get_plan_name(planName).is_good());
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_NAME);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), 0);
}

// Control for the two narrowing cases above: with nothing barred the shim must
// leave the heuristics choice alone and build the top-ranked plan.
TEST_F(TestCudnnShimNoteTriageBackend, UnfilteredHeuristicsChoiceBuildsTopRankedPlan)
{
    installRegisteredEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 2u);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_good()) << err.get_message();

    std::string planName;
    EXPECT_TRUE(graph.get_plan_name(planName).is_good());
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::MIOPEN_ENGINE_NAME);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), 0);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), -1);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectNondeterministicNarrowsToDeterministicEngine)
{
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    EXPECT_EQ(&graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC}), &graph);

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    std::string planName;
    EXPECT_TRUE(graph.get_plan_name(planName).is_good());
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_NAME);
}

// Plugin engines resolve to a hex-id plan name, not a registered one; they must
// still count as barred rather than be picked as the surviving plan.
TEST_F(TestCudnnShimNoteTriageBackend, DeselectNondeterministicSkipsBarredUnregisteredEngine)
{
    ASSERT_EQ(hipdnn_data_sdk::utilities::getEngineIdToNameMap().count(UNREGISTERED_ENGINE_ID), 0u);
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                        UNREGISTERED_ENGINE_ID,
                        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    ASSERT_EQ(_executionPlanDescs.size(), 3u);
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    std::string planName;
    EXPECT_TRUE(graph.get_plan_name(planName).is_good());
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_NAME);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectNondeterministicBuildAllBuildsOnlyDeterministicPlan)
{
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::ALL);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(1), 0);
}

// Canonical cuDNN order: the filter lands after plan creation and check_support
// is the first call that must report it, as upstream does.
TEST_F(TestCudnnShimNoteTriageBackend,
       DeselectNondeterministicWithoutDeterministicEngineFailsCheckSupport)
{
    installRegisteredEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.check_support();

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
}

TEST_F(TestCudnnShimNoteTriageBackend,
       DeselectNondeterministicWithoutDeterministicEngineFailsPlanCreation)
{
    installRegisteredEnginePlanMocks();

    fe::graph::Graph graph;
    addPointwiseGraph(graph);
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());

    auto err = graph.create_execution_plans({fe::HeurMode_t::A});

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectedDeterministicEngineLeavesNoDeterministicSurvivor)
{
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    graph.deselect_engines(
        std::vector<std::string>{hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_NAME});
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
    EXPECT_NE(err.get_message().find("deselect_engines"), std::string::npos);
}

// create_execution_plan(index) creates one plan; a deterministic engine elsewhere
// in the ranked list must not count as a survivor for it.
TEST_F(TestCudnnShimNoteTriageBackend, DeselectNondeterministicOnSingleCreatedPlanFailsCheckSupport)
{
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(addPointwiseGraph(graph));
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    auto createErr = graph.create_execution_plan(0, {});
    ASSERT_TRUE(createErr.is_good()) << createErr.get_message();
    ASSERT_EQ(graph.get_execution_plan_count(), 1);
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    expectGraphNotSupported(graph.check_support(), "NONDETERMINISTIC");
}

TEST_F(TestCudnnShimNoteTriageBackend, DeselectEnginesBarringEveryPlanFailsCheckSupport)
{
    installRegisteredEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));
    graph.deselect_engines(
        std::vector<std::string>{hipdnn_data_sdk::utilities::MIOPEN_ENGINE_NAME,
                                 hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_NAME});

    auto err = graph.check_support();

    expectGraphNotSupported(err, "deselect_engines");
    EXPECT_EQ(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
}

TEST_F(TestCudnnShimNoteTriageBackend, BehaviorNotesForPlanAtIndexResolvesUnregisteredEngine)
{
    ASSERT_EQ(hipdnn_data_sdk::utilities::getEngineIdToNameMap().count(UNREGISTERED_ENGINE_ID), 0u);
    _rankedEngineIds = {hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID, UNREGISTERED_ENGINE_ID};
    _behaviorNotesByEngineId[UNREGISTERED_ENGINE_ID] = {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION};
    installRankedEnginePlanMocks();

    fe::graph::Graph graph;
    ASSERT_NO_FATAL_FAILURE(createPlans(graph));

    std::vector<BehNote> notes;
    auto err = graph.get_behavior_notes_for_plan_at_index(1, notes);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(notes, std::vector<BehNote>{BehNote::RUNTIME_COMPILATION});
}

} // namespace

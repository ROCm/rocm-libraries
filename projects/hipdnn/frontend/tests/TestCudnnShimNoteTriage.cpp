// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Host-only coverage for the shim's public note-filter behavior. The triage is
// intentionally inline in Graph methods; these tests assert the observable
// contracts, not helper internals.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using NumNote = fe::NumericalNote_t;
using BehNote = fe::BehaviorNote_t;

TEST(TestCudnnShimNoteTriage, DeselectNondeterministicPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.validate();

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
}

TEST(TestCudnnShimNoteTriage, DeselectReducedPrecisionReductionPoisonsValidate)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::REDUCED_PRECISION_REDUCTION});

    auto err = graph.validate();

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("REDUCED_PRECISION_REDUCTION"), std::string::npos);
}

TEST(TestCudnnShimNoteTriage, SelectCorrectnessCriticalNotesLeavesGraphUsable)
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
        auto err = graph.validate();

        EXPECT_TRUE(err.is_bad());
        EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    }
}

TEST(TestCudnnShimNoteTriage, DeselectCudnnOnlyBehaviorNoteIsSafeNoOp)
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
    auto err = graph.validate();

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("shared-memory metadata"), std::string::npos);
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
    graph.deselect_numeric_notes({NumNote::TENSOR_CORE, NumNote::NONDETERMINISTIC});

    auto err = graph.validate();

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
}

TEST(TestCudnnShimNoteTriage, FirstRecordedNoteErrorWins)
{
    fe::graph::Graph graph;
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});
    graph.deselect_numeric_notes({NumNote::REDUCED_PRECISION_REDUCTION});

    auto err = graph.validate();

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
    EXPECT_EQ(err.get_message().find("REDUCED_PRECISION_REDUCTION"), std::string::npos);
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
    graph.deselect_numeric_notes({NumNote::NONDETERMINISTIC});

    auto err = graph.create_execution_plans({fe::HeurMode_t::A});

    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(err.get_message().find("NONDETERMINISTIC"), std::string::npos);
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

} // namespace

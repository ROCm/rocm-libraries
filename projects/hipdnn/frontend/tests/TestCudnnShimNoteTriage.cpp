// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Host-only coverage for the shim's public note-filter behavior. The triage is
// intentionally inline in Graph methods; these tests assert the observable
// contracts, not helper internals.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

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

TEST(TestCudnnShimNoteTriage, BehaviorNoteFiltersNeverPoisonTheGraph)
{
    const std::vector<BehNote> notes = {BehNote::RUNTIME_COMPILATION,
                                        BehNote::REQUIRES_LAYOUT_TRANSFORM,
                                        BehNote::SUPPORTS_GRAPH_CAPTURE,
                                        BehNote::EXTERNAL_LIBRARY_DEPENDENCY,
                                        BehNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION,
                                        BehNote::REQUIRES_FILTER_INT8x32_REORDER,
                                        BehNote::REQUIRES_BIAS_INT8x32_REORDER,
                                        BehNote::SUPPORTS_CUDA_GRAPH_NATIVE_API,
                                        BehNote::CUBLASLT_DEPENDENCY};

    fe::graph::Graph selectGraph;
    selectGraph.select_behavior_notes(notes);
    EXPECT_TRUE(selectGraph.validate().is_good());

    fe::graph::Graph deselectGraph;
    deselectGraph.deselect_behavior_notes(notes);
    EXPECT_TRUE(deselectGraph.validate().is_good());
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

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Pins note dispositions and noteName to the triage tables in docs/KNOWN_DIVERGENCES.md.
#include <hipdnn_compatibility/cudnn/detail/note_triage.h>

#include <gtest/gtest.h>

#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::detail::NoteAction;
using NumNote = fe::NumericalNote_t;
using BehNote = fe::BehaviorNote_t;

template <typename Note>
struct TriageRow
{
    Note note;
    NoteAction select;
    NoteAction deselect;
};

std::vector<TriageRow<NumNote>> numericalTriageTable()
{
    return {
        {NumNote::NOT_SET, NoteAction::NO_OP, NoteAction::NO_OP},
        {NumNote::TENSOR_CORE, NoteAction::WARN, NoteAction::WARN},
        {NumNote::DOWN_CONVERT_INPUTS, NoteAction::WARN, NoteAction::RECORD_ERROR},
        {NumNote::REDUCED_PRECISION_REDUCTION, NoteAction::WARN, NoteAction::RECORD_ERROR},
        {NumNote::FFT, NoteAction::WARN, NoteAction::WARN},
        {NumNote::NONDETERMINISTIC, NoteAction::WARN, NoteAction::MAP},
        {NumNote::WINOGRAD, NoteAction::WARN, NoteAction::WARN},
        {NumNote::WINOGRAD_TILE_4x4, NoteAction::WARN, NoteAction::WARN},
        {NumNote::WINOGRAD_TILE_6x6, NoteAction::WARN, NoteAction::WARN},
        {NumNote::WINOGRAD_TILE_13x13, NoteAction::WARN, NoteAction::WARN},
        {NumNote::STRICT_NAN_PROP, NoteAction::RECORD_ERROR, NoteAction::WARN},
        {static_cast<NumNote>(1000), NoteAction::RECORD_ERROR, NoteAction::RECORD_ERROR},
    };
}

TEST(TestCudnnShimNoteTriageTable, NumericalNoteDispositions)
{
    for(const auto& row : numericalTriageTable())
    {
        SCOPED_TRACE(fe::detail::noteName(row.note));
        EXPECT_EQ(fe::detail::triageSelect(row.note), row.select);
        EXPECT_EQ(fe::detail::triageDeselect(row.note), row.deselect);
    }
}

TEST(TestCudnnShimNoteTriageTable, OnlyDeselectNondeterministicMapsANumericalNote)
{
    for(const auto& row : numericalTriageTable())
    {
        SCOPED_TRACE(fe::detail::noteName(row.note));
        EXPECT_NE(fe::detail::triageSelect(row.note), NoteAction::MAP);
        EXPECT_EQ(fe::detail::triageDeselect(row.note) == NoteAction::MAP,
                  row.note == NumNote::NONDETERMINISTIC);
    }
}

TEST(TestCudnnShimNoteTriageTable, BehaviorNoteDispositions)
{
    const std::vector<TriageRow<BehNote>> table = {
        {BehNote::NOT_SET, NoteAction::NO_OP, NoteAction::NO_OP},
        {BehNote::RUNTIME_COMPILATION, NoteAction::MAP, NoteAction::MAP},
        {BehNote::REQUIRES_FILTER_INT8x32_REORDER, NoteAction::RECORD_ERROR, NoteAction::WARN},
        {BehNote::REQUIRES_BIAS_INT8x32_REORDER, NoteAction::RECORD_ERROR, NoteAction::WARN},
        {BehNote::SUPPORTS_CUDA_GRAPH_NATIVE_API, NoteAction::RECORD_ERROR, NoteAction::WARN},
        {BehNote::CUBLASLT_DEPENDENCY, NoteAction::RECORD_ERROR, NoteAction::WARN},
        {BehNote::REQUIRES_LAYOUT_TRANSFORM, NoteAction::MAP, NoteAction::MAP},
        {BehNote::SUPPORTS_GRAPH_CAPTURE, NoteAction::MAP, NoteAction::MAP},
        {BehNote::EXTERNAL_LIBRARY_DEPENDENCY, NoteAction::MAP, NoteAction::MAP},
        {BehNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION, NoteAction::MAP, NoteAction::MAP},
        {static_cast<BehNote>(1000), NoteAction::RECORD_ERROR, NoteAction::WARN},
    };

    for(const auto& row : table)
    {
        SCOPED_TRACE(fe::detail::noteName(row.note));
        EXPECT_EQ(fe::detail::triageSelect(row.note), row.select);
        EXPECT_EQ(fe::detail::triageDeselect(row.note), row.deselect);
    }
}

TEST(TestCudnnShimNoteTriageTable, OutOfRangeNoteNameCarriesValue)
{
    EXPECT_EQ(fe::detail::noteName(NumNote::FFT), "FFT");
    EXPECT_EQ(fe::detail::noteName(static_cast<NumNote>(1000)), "unknown (1000)");
    EXPECT_EQ(fe::detail::noteName(BehNote::RUNTIME_COMPILATION), "RUNTIME_COMPILATION");
    EXPECT_EQ(fe::detail::noteName(static_cast<BehNote>(1000)), "unknown (1000)");
}

} // namespace

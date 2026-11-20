// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Per-engine predicates behind the shim's behavior-note and numerical-note filters.
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>

#include <gtest/gtest.h>

#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using BehNote = fe::BehaviorNote_t;
using NumNote = fe::NumericalNote_t;

TEST(TestCudnnShimEngineFilters, BehaviorNotesMatchRequiresSelectedAndExcludesDeselected)
{
    const std::vector<BehNote> engineNotes
        = {BehNote::RUNTIME_COMPILATION, BehNote::SUPPORTS_GRAPH_CAPTURE};

    EXPECT_TRUE(fe::detail::behaviorNotesMatch({}, {}, engineNotes));
    EXPECT_TRUE(fe::detail::behaviorNotesMatch({BehNote::SUPPORTS_GRAPH_CAPTURE}, {}, engineNotes));
    EXPECT_FALSE(
        fe::detail::behaviorNotesMatch({BehNote::REQUIRES_LAYOUT_TRANSFORM}, {}, engineNotes));
    EXPECT_FALSE(fe::detail::behaviorNotesMatch({}, {BehNote::RUNTIME_COMPILATION}, engineNotes));
    EXPECT_TRUE(
        fe::detail::behaviorNotesMatch({}, {BehNote::EXTERNAL_LIBRARY_DEPENDENCY}, engineNotes));
}

TEST(TestCudnnShimEngineFilters, NumericalNotesMatchRequiresSelectedAndExcludesDeselected)
{
    const std::vector<NumNote> nondeterministic = {NumNote::NONDETERMINISTIC};
    const std::vector<NumNote> none;

    EXPECT_TRUE(fe::detail::numericalNotesMatch({}, {}, nondeterministic));
    EXPECT_TRUE(fe::detail::numericalNotesMatch({NumNote::NONDETERMINISTIC}, {}, nondeterministic));
    EXPECT_FALSE(fe::detail::numericalNotesMatch({NumNote::NONDETERMINISTIC}, {}, none));
    EXPECT_FALSE(
        fe::detail::numericalNotesMatch({}, {NumNote::NONDETERMINISTIC}, nondeterministic));
    EXPECT_TRUE(fe::detail::numericalNotesMatch({}, {NumNote::NONDETERMINISTIC}, none));
}

} // namespace

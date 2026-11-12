// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Per-engine predicates behind the shim's behavior-note and determinism filters.
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using BehNote = fe::BehaviorNote_t;

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

TEST(TestCudnnShimEngineFilters, OnlyTheDeterministicMiopenEngineClaimsDeterminism)
{
    EXPECT_TRUE(fe::detail::isDeterminismClaimed(
        hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID));
    EXPECT_FALSE(fe::detail::isDeterminismClaimed(hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID));
    EXPECT_FALSE(fe::detail::isDeterminismClaimed(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID));
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include "harness/bundle/LoadedEngine.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The id a case carries when it asks for the engine's normal pick rather than one catalog
/// candidate: what an engine that cannot enumerate its candidates is listed as.
inline constexpr const char* DEFAULT_CANDIDATE_ID = "default";

/// The one catalog candidate a case asks the engine to build (`metadata.engine_variant`).
///
/// `knobs` is the candidate's complete knob tuple, as the listing prints it. `id` names it in
/// the outcome record; it is not sent to the engine, which selects by the knobs.
struct EngineVariantSpec
{
    std::string id;
    std::vector<hipdnn_frontend::KnobSetting> knobs;

    /// The engine's own heuristic pick: the same plan creation a case without an
    /// engine_variant gets.
    bool isDefault() const
    {
        return id == DEFAULT_CANDIDATE_ID;
    }
};

/// Creates the execution plan a case asks for, on `graph` (a hipdnn_frontend::graph::Graph, or
/// anything with its two plan-creation calls).
///
/// No variant, or the default one, is create_execution_plans(): the heuristic path. A catalog
/// candidate is create_execution_plan_ext() with its knobs, never the heuristic path: the
/// heuristic may rank another candidate first, and add_engine_variants() feeds autotune only
/// (create_execution_plans() refuses after it), so neither of them builds exactly one chosen
/// candidate.
///
/// A template so a test can prove which of the two was called without a device. A chosen
/// candidate needs a named engine: a knob tuple selects a candidate only within one engine.
template <typename GraphT>
hipdnn_frontend::Error createCasePlans(GraphT& graph,
                                       const std::optional<LoadedEngine>& engineUnderTest,
                                       const std::optional<EngineVariantSpec>& variant)
{
    if(!variant.has_value() || variant->isDefault())
    {
        return graph.create_execution_plans();
    }
    if(!engineUnderTest.has_value())
    {
        return {hipdnn_frontend::ErrorCode::INVALID_VALUE,
                "engine_variant \"" + variant->id
                    + "\" selects a candidate of one engine, so the run must name it with "
                      "--test-engine"};
    }
    return graph.create_execution_plan_ext(engineUnderTest->id, variant->knobs);
}

} // namespace hipdnn_integration_tests::bundle

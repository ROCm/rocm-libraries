// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/LoadedEngine.hpp"

// Held by pointer so this header stays free of the frontend's Graph.hpp, which the
// harness header pulls into every test that includes it.
namespace hipdnn_frontend::graph
{
class Graph;
}

namespace hipdnn_integration_tests::bundle
{

/// Only these two codes mean the ranked list can be believed.
///
/// Anything else leaves "does this engine take the graph" unknown, so callers must
/// fail closed toward "cannot tell" rather than toward a false "declined". Future
/// codes nobody has classified land on the unknown side by default.
bool isResolved(hipdnn_frontend::ErrorCode code);

/// Will the engine this lane tests take this graph?
///
/// The one definition of applicability. The claim verdict, the executor and the
/// enforcement rungs all read the same answer, so a graph cannot be "broken claim"
/// to one of them and "engine declined" to another.
///
/// `engineUnderTest` empty means the run is unpinned (no --test-engine), where any
/// ranked engine will do.
bool enginesAccept(hipdnn_frontend::ErrorCode status,
                   const std::vector<int64_t>& rankedIds,
                   const std::optional<LoadedEngine>& engineUnderTest);

/// What one `get_ranked_engine_ids()` call said, and what it means for this lane.
///
/// Split from GraphSession on purpose: everything that has to make a decision needs
/// only this part, and this part needs no device, no handle, and no graph — so the
/// decisions are all testable on their own.
struct RankedEngines
{
    /// The frontend's own answer, kept whole rather than decomposed into a code and
    /// a message. isResolved() and enginesAccept() still take the bare ErrorCode --
    /// they are pure predicates over it -- so callers pass status.get_code().
    hipdnn_frontend::Error status;
    std::vector<int64_t> rankedIds;

    /// enginesAccept() applied to the fields above, computed once at the query.
    bool accepted = false;
};

/// The graph under test, plus the single ranked query taken against it.
///
/// Built once per test in TestBody() and handed to whatever needs it. Nothing
/// re-derives it and nothing caches it on the harness: one `from_binary`, one
/// heuristic query, one applicability answer, all visible in the call chain.
///
/// `graph` is null when the build failed (see `buildError`) and in the deviceless
/// unit harnesses, which supply a canned `engines` and never execute anything.
struct GraphSession
{
    GraphSession();
    GraphSession(GraphSession&&) noexcept;
    GraphSession& operator=(GraphSession&&) noexcept;
    ~GraphSession();

    GraphSession(const GraphSession&) = delete;
    GraphSession& operator=(const GraphSession&) = delete;

    std::unique_ptr<hipdnn_frontend::graph::Graph> graph;
    std::string buildError; ///< non-empty when from_binary failed
    RankedEngines engines;
};

} // namespace hipdnn_integration_tests::bundle

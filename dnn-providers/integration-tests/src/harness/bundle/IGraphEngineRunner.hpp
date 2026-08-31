// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "harness/VariantPack.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"

namespace hipdnn_integration_tests::bundle
{

struct IntegrationTestBundle;

/// What one frontend operation did.
///
/// Returned rather than asserted so nothing below the harness owns a GTest
/// disposition. This is what replaced reading ::testing::Test::HasFatalFailure()
/// after the fact — that could not tell an engine assertion apart from any other
/// fatal failure in the same test, and blamed the engine for both.
struct EngineOpResult
{
    bool ok = false;
    /// The engine refused the graph. Distinct from `!ok`: a decline is the engine
    /// answering "not mine", which skips, while a failure is the engine breaking.
    bool declined = false;
    std::string message;

    static EngineOpResult succeeded()
    {
        return {true, false, {}};
    }

    static EngineOpResult failed(std::string message)
    {
        return {false, false, std::move(message)};
    }

    static EngineOpResult declinedBy(std::string message)
    {
        return {false, true, std::move(message)};
    }
};

/// Everything the harness does with a hipdnn_frontend::graph::Graph.
///
/// One seam, not three, because the three things the harness needs — load the
/// graph, compile plans for it, run it — all need the same shared handle and the
/// same device. Behind this interface a unit test needs neither, which is what
/// makes enforceAtLevel() and the whole verification body testable without an
/// engine plugin loaded.
///
/// Implementations MUST return a decline rather than throw
/// EngineNotApplicableError: "this engine will not take the graph" is an answer,
/// not an error.
class IGraphEngineRunner
{
public:
    IGraphEngineRunner() = default;
    virtual ~IGraphEngineRunner() = default;

    IGraphEngineRunner(const IGraphEngineRunner&) = delete;
    IGraphEngineRunner& operator=(const IGraphEngineRunner&) = delete;
    IGraphEngineRunner(IGraphEngineRunner&&) = delete;
    IGraphEngineRunner& operator=(IGraphEngineRunner&&) = delete;

    /// from_binary, plus the single get_ranked_engine_ids() query this test makes.
    /// A build failure comes back as GraphSession::buildError, not as a throw.
    virtual GraphSession openGraph(const IntegrationTestBundle& bundle,
                                   const std::optional<LoadedEngine>& engineUnderTest)
        = 0;

    /// create_execution_plans + check_support + build_plans, pinned to
    /// `engineUnderTest` when one was named.
    virtual EngineOpResult buildPlans(GraphSession& session,
                                      const std::optional<LoadedEngine>& engineUnderTest)
        = 0;

    /// buildPlans(), then allocate the workspace and execute into `variantPack`.
    /// Non-const because hipdnn_frontend::graph::Graph::execute() takes it that way.
    virtual EngineOpResult execute(GraphSession& session,
                                   const std::optional<LoadedEngine>& engineUnderTest,
                                   VariantPack& variantPack)
        = 0;
};

} // namespace hipdnn_integration_tests::bundle

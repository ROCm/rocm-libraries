// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/bundle/IGraphEngineRunner.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The production IGraphEngineRunner: drives a real hipdnn_frontend::graph::Graph
/// on the shared handle.
///
/// Implements the steps the harness needs from an engine, in the order it needs
/// them:
///   - openGraph()  — one from_binary, then the single get_ranked_engine_ids()
///                    query whose answer becomes GraphSession::engines.
///   - buildPlans() — create_execution_plans / check_support / build_plans, with a
///                    late decline returned as an answer rather than thrown.
///   - execute()    — builds plans if needed, then runs the graph on a variant pack.
///   - benchmark()  — times the plan execute() built (--benchmark only).
///
/// This is the only place in the harness that needs a handle, a device, or a
/// loaded engine plugin. Kept in its own translation unit so the unit-test binary
/// and hipdnn_golden_data_tests can link the harness without any of them; the
/// deviceless suites inject a mock through HarnessDependencies instead.
class FrontendGraphEngineRunner : public IGraphEngineRunner
{
public:
    GraphSession openGraph(const IntegrationTestBundle& bundle,
                           const std::optional<LoadedEngine>& engineUnderTest) override;

    EngineOpResult buildPlans(GraphSession& session,
                              const std::optional<LoadedEngine>& engineUnderTest) override;

    EngineOpResult execute(GraphSession& session,
                           const std::optional<LoadedEngine>& engineUnderTest,
                           VariantPack& variantPack) override;

    hipdnn_frontend::AutotuneResult benchmark(GraphSession& session,
                                              VariantPack& variantPack,
                                              const BenchmarkSettings& settings) override;
};

} // namespace hipdnn_integration_tests::bundle

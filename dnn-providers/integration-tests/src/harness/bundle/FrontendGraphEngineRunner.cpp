// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/FrontendGraphEngineRunner.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>

#include "harness/EngineNotApplicableError.hpp"
#include "harness/GraphBenchmark.hpp"
#include "harness/PlanWorkspace.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

GraphSession
    FrontendGraphEngineRunner::openGraph(const IntegrationTestBundle& bundle,
                                         const std::optional<LoadedEngine>& engineUnderTest)
{
    GraphSession session;

    auto handle = getSharedHandle();
    session.graph = std::make_unique<hipdnn_frontend::graph::Graph>();

    const std::vector<uint8_t> graphBytes(bundle.graphBuffer.data(),
                                          bundle.graphBuffer.data() + bundle.graphBuffer.size());

    if(auto err = session.graph->from_binary(handle, graphBytes); !err.is_good())
    {
        session.buildFailed = true;
        session.buildError = err.get_message();
        return session;
    }

    // The only heuristic query this test makes. It is a pure read on the graph — the
    // executor calls create_execution_plans() on this same object afterwards, which
    // is what the unpinned path has always done.
    std::vector<int64_t> ids;
    session.engines.status = session.graph->get_ranked_engine_ids(ids);
    session.engines.rankedIds = std::move(ids);
    session.engines.accepted = enginesAccept(
        session.engines.status.get_code(), session.engines.rankedIds, engineUnderTest);

    return session;
}

EngineOpResult
    FrontendGraphEngineRunner::buildPlans(GraphSession& session,
                                          const std::optional<LoadedEngine>& engineUnderTest)
{
    if(session.graph == nullptr)
    {
        return EngineOpResult::failed("openGraph() produced no graph to compile");
    }
    auto& graph = *session.graph;

    // Applicability was settled once, in openGraph(); the callers have already
    // turned a decline into a skip, so reaching here means the engine takes it.
    if(engineUnderTest.has_value())
    {
        graph.set_preferred_engine_id_ext(engineUnderTest->id);
    }

    // Each stage returns the frontend's own Error; the first bad one is the answer
    // and carries its message out. Returning an optional rather than the result
    // keeps the early return visible at the call site instead of hiding it in a
    // macro.
    const auto firstFailure
        = [](const hipdnn_frontend::Error& result) -> std::optional<EngineOpResult> {
        if(!result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
        }
        return std::nullopt;
    };

    try
    {
        if(auto failed = firstFailure(graph.create_execution_plans()))
        {
            return *failed;
        }
        if(auto failed = firstFailure(graph.check_support()))
        {
            return *failed;
        }
        if(auto failed = firstFailure(graph.build_plans()))
        {
            return *failed;
        }
    }
    // A provider is free to decline later than the ranked list suggested. That is
    // an answer, not a break, so it is returned rather than allowed to escape.
    catch(const EngineNotApplicableError& e)
    {
        return EngineOpResult::declinedBy(e.what());
    }

    return EngineOpResult::succeeded();
}

EngineOpResult
    FrontendGraphEngineRunner::execute(GraphSession& session,
                                       const std::optional<LoadedEngine>& engineUnderTest,
                                       VariantPack& variantPack)
{
    if(auto built = buildPlans(session, engineUnderTest); !built.ok)
    {
        return built;
    }
    auto& graph = *session.graph;
    auto handle = getSharedHandle();

    // Plans are compiled by the time this runs, so every answer below it — success,
    // break, or late decline — is one the engine gave after reaching BUILDABLE.
    // Stamped once here rather than at each return, where the next one added would
    // silently under-report the depth.
    auto result = [&]() -> EngineOpResult {
        try
        {
            std::string error;
            auto workspace = allocatePlanWorkspace(graph, error);
            if(!workspace.has_value())
            {
                return EngineOpResult::failed(error);
            }

            if(auto status = graph.execute(handle, variantPack, workspace->get());
               !status.is_good())
            {
                return EngineOpResult::failed(status.get_message());
            }
        }
        catch(const EngineNotApplicableError& e)
        {
            return EngineOpResult::declinedBy(e.what());
        }

        return EngineOpResult::succeeded();
    }();

    result.plansBuilt = true;
    return result;
}

hipdnn_frontend::AutotuneResult FrontendGraphEngineRunner::benchmark(
    GraphSession& session, VariantPack& variantPack, const BenchmarkSettings& settings)
{
    if(session.graph == nullptr)
    {
        hipdnn_frontend::AutotuneResult failed;
        failed.errorMessage = "openGraph() produced no graph to benchmark";
        return failed;
    }
    return benchmarkGraph(*session.graph, getSharedHandle(), variantPack, settings);
}

} // namespace hipdnn_integration_tests::bundle

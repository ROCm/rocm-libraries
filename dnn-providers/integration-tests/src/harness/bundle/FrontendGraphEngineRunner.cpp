// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/FrontendGraphEngineRunner.hpp"

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include "harness/EngineNotApplicableError.hpp"
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
        session.buildError = err.get_message();
        return session;
    }

    // The only heuristic query this test makes. It is a pure read on the graph — the
    // executor calls create_execution_plans() on this same object afterwards, which
    // is what the unpinned path has always done.
    std::vector<int64_t> ids;
    auto status = session.graph->get_ranked_engine_ids(ids);
    session.engines.status = status.get_code();
    session.engines.statusMessage = status.get_message();
    session.engines.rankedIds = std::move(ids);
    session.engines.accepted
        = enginesAccept(session.engines.status, session.engines.rankedIds, engineUnderTest);

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

    try
    {
        if(auto result = graph.create_execution_plans(); !result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
        }
        if(auto result = graph.check_support(); !result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
        }
        if(auto result = graph.build_plans(); !result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
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

    try
    {
        int64_t workspaceSize = 0;
        if(auto result = graph.get_workspace_size(workspaceSize); !result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
        }
        if(workspaceSize < 0)
        {
            return EngineOpResult::failed("engine reported a negative workspace size");
        }
        const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        if(auto result = graph.execute(handle, variantPack, workspace.get()); !result.is_good())
        {
            return EngineOpResult::failed(result.get_message());
        }
    }
    catch(const EngineNotApplicableError& e)
    {
        return EngineOpResult::declinedBy(e.what());
    }

    return EngineOpResult::succeeded();
}

} // namespace hipdnn_integration_tests::bundle

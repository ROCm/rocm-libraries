// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/GraphSession.hpp"

#include <algorithm>

#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_integration_tests::bundle
{

bool isResolved(hipdnn_frontend::ErrorCode code)
{
    return code == hipdnn_frontend::ErrorCode::OK
           || code == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
}

bool enginesAccept(hipdnn_frontend::ErrorCode status,
                   const std::vector<int64_t>& rankedIds,
                   const std::optional<LoadedEngine>& engineUnderTest)
{
    if(!isResolved(status))
    {
        return false;
    }

    if(!engineUnderTest.has_value())
    {
        // Unpinned: the harness will let the backend pick, so any ranked engine does.
        return !rankedIds.empty();
    }

    return std::find(rankedIds.begin(), rankedIds.end(), engineUnderTest->id) != rankedIds.end();
}

// Out of line so the header can forward-declare Graph: the implicit destructor and
// moves would need the complete type wherever they were instantiated.
GraphSession::GraphSession() = default;
GraphSession::GraphSession(GraphSession&&) noexcept = default;
GraphSession& GraphSession::operator=(GraphSession&&) noexcept = default;
GraphSession::~GraphSession() = default;

} // namespace hipdnn_integration_tests::bundle

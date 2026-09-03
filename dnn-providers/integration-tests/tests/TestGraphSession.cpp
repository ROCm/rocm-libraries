// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Applicability is decided once per test, in openGraph(), and read by three places:
// the support-claim verdict, the executor, and the enforcement rungs. These pin the
// predicate they all share, so a graph cannot be "broken claim" to one of them and
// "engine declined" to another.

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "harness/bundle/GraphSession.hpp"

using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::bundle::enginesAccept;
using hipdnn_integration_tests::bundle::GraphSession;
using hipdnn_integration_tests::bundle::isResolved;
using hipdnn_integration_tests::bundle::LoadedEngine;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t UNDER_TEST_ID = 42;
constexpr int64_t OTHER_ID = 7;

const std::optional<LoadedEngine> ENGINE{LoadedEngine{UNDER_TEST_ID, "ENGINE_UNDER_TEST"}};
const std::optional<LoadedEngine> UNPINNED{};

} // namespace

// ---------------------------------------------------------------------------
// isResolved: a whitelist, so a future code nobody classified is "cannot tell"
// rather than a false "declined".
// ---------------------------------------------------------------------------

TEST(TestGraphSession, OnlyOkAndGraphNotSupportedAreResolved)
{
    EXPECT_TRUE(isResolved(ErrorCode::OK));
    EXPECT_TRUE(isResolved(ErrorCode::GRAPH_NOT_SUPPORTED));

    EXPECT_FALSE(isResolved(ErrorCode::HEURISTIC_QUERY_FAILED));
    EXPECT_FALSE(isResolved(ErrorCode::INVALID_VALUE));
}

TEST(TestGraphSession, UnknownErrorCodeIsUnresolved)
{
    EXPECT_FALSE(isResolved(static_cast<ErrorCode>(9999)));
}

// ---------------------------------------------------------------------------
// enginesAccept: pinned lane.
// ---------------------------------------------------------------------------

TEST(TestGraphSession, PinnedEngineInTheRankedListIsAccepted)
{
    EXPECT_TRUE(enginesAccept(ErrorCode::OK, {OTHER_ID, UNDER_TEST_ID}, ENGINE));
}

TEST(TestGraphSession, PinnedEngineAbsentFromTheRankedListIsNotAccepted)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::OK, {OTHER_ID}, ENGINE));
}

TEST(TestGraphSession, PinnedEngineWithAnEmptyRankedListIsNotAccepted)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::OK, {}, ENGINE));
}

// GRAPH_NOT_SUPPORTED is resolved: the backend answered, and the answer is "nobody
// takes this graph". That is a decline, not an unknown.
TEST(TestGraphSession, GraphNotSupportedIsADeclineNotAnUnknown)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::GRAPH_NOT_SUPPORTED, {}, ENGINE));
}

// An unresolved query cannot be read as acceptance even when the id happens to be
// in the list the backend half-filled.
TEST(TestGraphSession, UnresolvedQueryIsNeverAccepted)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::HEURISTIC_QUERY_FAILED, {UNDER_TEST_ID}, ENGINE));
    EXPECT_FALSE(enginesAccept(static_cast<ErrorCode>(9999), {UNDER_TEST_ID}, ENGINE));
}

// ---------------------------------------------------------------------------
// enginesAccept: unpinned lane (no --test-engine), where any ranked engine does.
// ---------------------------------------------------------------------------

TEST(TestGraphSession, UnpinnedRunAcceptsAnyRankedEngine)
{
    EXPECT_TRUE(enginesAccept(ErrorCode::OK, {OTHER_ID}, UNPINNED));
}

TEST(TestGraphSession, UnpinnedRunWithNoRankedEnginesIsNotAccepted)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::OK, {}, UNPINNED));
}

TEST(TestGraphSession, UnpinnedRunWithAnUnresolvedQueryIsNotAccepted)
{
    EXPECT_FALSE(enginesAccept(ErrorCode::HEURISTIC_QUERY_FAILED, {OTHER_ID}, UNPINNED));
}

// ---------------------------------------------------------------------------
// The session defaults to "nothing was opened", so a harness that never built a
// graph cannot look like one whose engine accepted.
// ---------------------------------------------------------------------------

TEST(TestGraphSession, DefaultSessionHasNoGraphAndAcceptsNothing)
{
    const GraphSession session;

    EXPECT_EQ(session.graph, nullptr);
    EXPECT_TRUE(session.buildError.empty());
    EXPECT_FALSE(session.engines.accepted);
    EXPECT_TRUE(session.engines.rankedIds.empty());
    EXPECT_EQ(session.engines.status.get_code(), ErrorCode::OK);
    EXPECT_TRUE(session.engines.status.get_message().empty());
}

TEST(TestGraphSession, SessionIsMovable)
{
    GraphSession session;
    session.buildError = "from_binary failed";
    session.engines.rankedIds = {UNDER_TEST_ID};
    session.engines.accepted = true;

    const GraphSession moved = std::move(session);

    EXPECT_EQ(moved.buildError, "from_binary failed");
    EXPECT_TRUE(moved.engines.accepted);
    ASSERT_EQ(moved.engines.rankedIds.size(), 1u);
    EXPECT_EQ(moved.engines.rankedIds.front(), UNDER_TEST_ID);
}

// NOLINTEND(readability-identifier-naming)

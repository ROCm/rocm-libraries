// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "harness/bundle/SupportObservationLog.hpp"

namespace hipdnn_integration_tests::bundle
{

struct WriteSummary
{
    size_t filesWritten = 0;
    size_t filesUnchanged = 0; // on-disk bytes already matched — no mtime bump
    size_t filesSkipped = 0; // left untouched: nothing to claim, or refused
    size_t observationsApplied = 0;
    std::vector<std::string> errors;
};

WriteSummary writeObservedSupportClaims(const std::vector<ObservedGraphSupport>& observations);

struct AuthoringResult
{
    WriteSummary writeSummary;
    bool shouldFail = false;
};

/// What the run reported about itself, for authorSupportClaims() to reconcile.
///
/// A struct rather than four positional std::size_t parameters: the counts are the
/// same type, they are filled in one place in main.cpp, and a transposition there
/// would compile and change the process exit code.
struct AuthoringRunSummary
{
    /// Graphs that reached the observer and got an answer from the engines.
    std::size_t graphsObserved = 0;

    /// Graphs that reached the observer and got nothing back at all. Their claims
    /// are left as they were on disk.
    std::size_t graphsUnobserved = 0;

    /// Graphs whose SetUp() returned before the observer ran, counted at the skip
    /// itself. See SupportObservationLog::recordSkipBeforeObservation().
    std::size_t graphsSkippedBeforeObservation = 0;

    /// Bundles that registered a test, counted at registration -- before any guard
    /// runs and before --gtest_filter deselects anything.
    std::size_t graphsRegistered = 0;

    /// A --gtest_filter or a shard split was in force. The graphs it removed were
    /// gone before SetUp could count them, so the shortfall below has an
    /// explanation this process cannot enumerate.
    bool selectionNarrowed = false;
};

AuthoringResult authorSupportClaims(const std::vector<ObservedGraphSupport>& observations,
                                    const AuthoringRunSummary& inputs,
                                    std::ostream& log);

/// Whether a run that selected this much of the suite can be expected to account
/// for every registered graph.
///
/// Split from selectionWasNarrowed() so the rule can be pinned against every filter
/// spelling without a test having to mutate process-wide GTest state.
bool selectionIsNarrowed(const std::string& gtestFilter, bool shardingActive);

/// selectionIsNarrowed() applied to this process.
bool selectionWasNarrowed();

} // namespace hipdnn_integration_tests::bundle

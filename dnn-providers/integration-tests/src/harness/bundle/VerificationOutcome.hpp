// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "harness/BundleMetadata.hpp"

namespace hipdnn_integration_tests::bundle
{

/// How far a test actually got with the engine under test.
///
/// Ordered on purpose: each rung includes the ones below it. "Did this test do what
/// its bundle asks?" is then a `>=` against requiredDepth(enforcement_level), and
/// "may we publish this claim as working support?" is the same comparison rather
/// than a second rule worded differently.
enum class VerificationDepth : uint8_t
{
    NOT_REACHED = 0, ///< the engine never got the graph
    APPLICABLE = 1, ///< engine in the ranked list; no plans built
    BUILDABLE = 2, ///< plans compiled
    EXECUTED = 3, ///< the graph ran, but no oracle compared its outputs
    VERIFIED = 4, ///< outputs compared against golden data or a reference
};

inline const char* toString(VerificationDepth depth)
{
    switch(depth)
    {
    case VerificationDepth::NOT_REACHED:
        return "not-reached";
    case VerificationDepth::APPLICABLE:
        return "applicable";
    case VerificationDepth::BUILDABLE:
        return "buildable";
    case VerificationDepth::EXECUTED:
        return "executed";
    case VerificationDepth::VERIFIED:
        return "verified";
    default:
        return "unknown";
    }
}

/// How far a bundle must get before it has done its job — and so before its support
/// claim may move from "advertised" to "confirmed".
inline VerificationDepth requiredDepth(EnforcementLevel level)
{
    switch(level)
    {
    case EnforcementLevel::APPLICABILITY:
        return VerificationDepth::APPLICABLE;
    case EnforcementLevel::BUILDABLE:
        return VerificationDepth::BUILDABLE;
    case EnforcementLevel::FULL:
    default:
        // Fail closed: a future level nobody has classified asks for the most, so a
        // claim cannot be confirmed by a rung nobody has thought about yet.
        return VerificationDepth::VERIFIED;
    }
}

enum class OutcomeStatus : uint8_t
{
    PASSED,
    SKIPPED,
    FAILED,
};

/// Who broke, when the status is FAILED.
///
/// Here so a support claim is only knocked back by evidence against the *engine*. A
/// reference executor that crashed, or a bundle whose golden `.bin` was never
/// pulled, turns the run red without saying anything about the engine. Marking that
/// cell "accepted but failed in use" would publish "do not use this" over somebody
/// else's bug.
enum class FailureOrigin : uint8_t
{
    NONE,
    ENGINE, ///< the engine declined, failed to build, or failed to execute
    COMPARISON, ///< the engine ran and its outputs did not match the oracle
    ORACLE, ///< the reference executor itself errored
    HARNESS, ///< missing golden data, an impossible mode, a harness bug
};

inline const char* toString(FailureOrigin origin)
{
    switch(origin)
    {
    case FailureOrigin::NONE:
        return "none";
    case FailureOrigin::ENGINE:
        return "engine";
    case FailureOrigin::COMPARISON:
        return "comparison";
    case FailureOrigin::ORACLE:
        return "reference";
    case FailureOrigin::HARNESS:
        return "harness";
    default:
        return "unknown";
    }
}

/// What one test body did, as a value.
///
/// Everything under TestBody() returns one of these instead of calling GTEST_SKIP()
/// or FAIL() on the spot. That is what lets the claim verdict and the test result
/// each be decided once, in one place, from the same facts, rather than each being
/// reconstructed afterwards from whatever state the other left behind.
struct VerificationOutcome
{
    OutcomeStatus status = OutcomeStatus::SKIPPED;
    VerificationDepth depth = VerificationDepth::NOT_REACHED;
    FailureOrigin origin = FailureOrigin::NONE;

    /// Skip reason or failure text, ready to print.
    std::string message;

    /// This failure is already on the gtest record, in more detail than `message`
    /// could add — the per-tensor diffs from the comparison.
    ///
    /// A flag rather than "message is empty" because emptiness is a property every
    /// producer has by accident and only one has on purpose. Overloading it means an
    /// error state with nothing to say reports nothing at all, which is the silent
    /// pass this harness exists to rule out. Only alreadyReported() sets it.
    bool alreadyReported = false;

    static VerificationOutcome passed(VerificationDepth depth)
    {
        return {OutcomeStatus::PASSED, depth, FailureOrigin::NONE, {}, false};
    }

    static VerificationOutcome skipped(VerificationDepth depth, std::string message)
    {
        return {OutcomeStatus::SKIPPED, depth, FailureOrigin::NONE, std::move(message), false};
    }

    static VerificationOutcome
        failed(VerificationDepth depth, FailureOrigin origin, std::string message)
    {
        return {OutcomeStatus::FAILED, depth, origin, std::move(message), false};
    }

    /// A failure whose detail is already on the record. The caller MUST have issued
    /// at least one gtest failure before building this.
    static VerificationOutcome alreadyReportedFailure(VerificationDepth depth, FailureOrigin origin)
    {
        return {OutcomeStatus::FAILED, depth, origin, {}, true};
    }
};

/// One line of "how far did it actually get", for the detail column of the support
/// claim report. This is what turns a bare CLAIM_CONFIRMED / CLAIM_ACCEPTED into
/// something a reader can act on.
inline std::string describeOutcome(const VerificationOutcome& outcome, VerificationDepth required)
{
    if(outcome.status == OutcomeStatus::FAILED)
    {
        if(outcome.origin == FailureOrigin::ENGINE || outcome.origin == FailureOrigin::COMPARISON)
        {
            return std::string("engine accepted the graph, but the test failed at ")
                   + toString(outcome.depth) + " (" + toString(outcome.origin) + ")";
        }
        return std::string("engine accepted the graph; the run failed outside the engine (")
               + toString(outcome.origin) + ") at " + toString(outcome.depth);
    }

    if(outcome.depth >= required)
    {
        return std::string("engine in ranked list; reached ") + toString(outcome.depth)
               + ", the depth this bundle requires";
    }

    return std::string("engine in ranked list; reached ") + toString(outcome.depth)
           + ", bundle requires " + toString(required);
}

} // namespace hipdnn_integration_tests::bundle

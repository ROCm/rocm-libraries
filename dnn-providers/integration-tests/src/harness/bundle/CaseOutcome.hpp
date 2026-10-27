// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/VerificationOutcome.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The one outcome a case gets on one engine (RFC 0023 §9.1).
enum class CaseOutcome
{
    PASS, ///< accepted and verified against an oracle
    DECLINED, ///< the support query resolved and the engine is absent from the ranked list
    WRONG, ///< accepted, ran, and the output did not match the oracle
    CRASH, ///< ranked, then failed to build or execute (a late NOT_SUPPORTED included)
    UNCLAIMED_PASS, ///< PASS on a graph whose sidecar does not claim this cell
    UNEXERCISED, ///< nothing verified it; never green
};

const char* toString(CaseOutcome outcome);

/// Derives the outcome from the facts the harness already decided (RFC 0023 §9.2).
/// Pure: no gtest state, no I/O.
///
///   outcome         what the test body did, with its stage
///   claimResults    the verdicts the claim phase produced, after finalizeClaims()
///   engineName      the engine under test, empty for an unpinned run
///   required        the depth the bundle's enforcement_level asks for
///   hadOtherFailure ::testing::Test::HasFailure(): a PASSED outcome that still left a
///                   failure on the gtest record (a harness complaint, a non-fatal
///                   check) is not a pass
CaseOutcome classifyCaseOutcome(const VerificationOutcome& outcome,
                                const std::vector<SupportResult>& claimResults,
                                std::string_view engineName,
                                VerificationDepth required,
                                bool hadOtherFailure);

/// Who a record is about. Everything here is known before the case runs. The `started`
/// record carries only `test` and `bundle` (enough to name the case a crash happened in);
/// the `finished` record carries all of it.
struct CaseIdentity
{
    std::string test; ///< "Suite.Test", the name --gtest_filter selects on
    std::string bundle;
    std::string engine; ///< empty when the run named no engine
    std::string arch;
    std::string platform;
    std::optional<CasePurpose> purpose;
    /// How the case asked to be graded and placed (`metadata.validator`, `metadata.placement`),
    /// spelled as the metadata spells them. Empty when the identity was not built from a bundle's
    /// metadata (a bundle that failed to load), and then left out of the record.
    std::string validator;
    std::string placement;
    std::optional<int64_t> placementUid;
    /// The fill recipes the harness parsed from `metadata.inputs`, keyed by tensor uid; null
    /// when the case sets none. Lets a passing run show that its recipe reached the harness.
    nlohmann::json inputs;
};

/// What one case ended with: the identity plus the classified outcome and the facts
/// it came from.
struct CaseOutcomeRecord
{
    CaseIdentity identity;
    CaseOutcome outcome = CaseOutcome::UNEXERCISED;
    OutcomeStage stage = OutcomeStage::NONE;
    VerificationDepth depth = VerificationDepth::NOT_REACHED;
    FailureOrigin origin = FailureOrigin::NONE;
    Verifier verifier = Verifier::NONE;
    /// Why, in the harness's own words. Empty for a PASS.
    std::string reason;
    /// The non-success status the case expected and got ("BAD_PARAM"); empty otherwise.
    std::string expectedStatus;
};

/// Builds the record for a test body that reached an outcome. `reason` is the
/// outcome's message, or a stock sentence when the detail is already on the gtest
/// record (a comparison mismatch) or the outcome is a pass that went wrong.
CaseOutcomeRecord makeCaseOutcomeRecord(CaseIdentity identity,
                                        const VerificationOutcome& outcome,
                                        CaseOutcome classified,
                                        bool hadOtherFailure);

/// A test that ended before it had a VerificationOutcome: a SetUp() skip, a throw, an
/// authoring run. Always UNEXERCISED.
CaseOutcomeRecord
    makeUnexercisedRecord(CaseIdentity identity, OutcomeStage stage, std::string reason);

/// One JSON object, no trailing newline. Invalid UTF-8 in a message is replaced, not
/// thrown on.
std::string startedRecordJson(const CaseIdentity& identity);
std::string finishedRecordJson(const CaseOutcomeRecord& record);

} // namespace hipdnn_integration_tests::bundle

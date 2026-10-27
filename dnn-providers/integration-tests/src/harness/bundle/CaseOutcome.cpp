// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/CaseOutcome.hpp"

#include <algorithm>
#include <utility>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests::bundle
{

const char* toString(CaseOutcome outcome)
{
    switch(outcome)
    {
    case CaseOutcome::PASS:
        return "PASS";
    case CaseOutcome::DECLINED:
        return "DECLINED";
    case CaseOutcome::WRONG:
        return "WRONG";
    case CaseOutcome::CRASH:
        return "CRASH";
    case CaseOutcome::UNCLAIMED_PASS:
        return "UNCLAIMED_PASS";
    case CaseOutcome::UNEXERCISED:
        return "UNEXERCISED";
    default:
        return "UNKNOWN";
    }
}

CaseOutcome classifyCaseOutcome(const VerificationOutcome& outcome,
                                const std::vector<SupportResult>& claimResults,
                                std::string_view engineName,
                                VerificationDepth required,
                                bool hadOtherFailure)
{
    switch(outcome.status)
    {
    case OutcomeStatus::PASSED:
    {
        // A pass that left a failure on the gtest record, or that stopped short of the
        // depth its bundle asks for, verified less than it says. Never green.
        if(hadOtherFailure || outcome.depth < required)
        {
            return CaseOutcome::UNEXERCISED;
        }

        const bool unclaimed = std::any_of(
            claimResults.begin(), claimResults.end(), [&](const SupportResult& result) {
                return result.verdict == SupportVerdict::UNCLAIMED_SUPPORT
                       && result.engineName == engineName;
            });
        return unclaimed ? CaseOutcome::UNCLAIMED_PASS : CaseOutcome::PASS;
    }

    case OutcomeStatus::SKIPPED:
        return outcome.declined ? CaseOutcome::DECLINED : CaseOutcome::UNEXERCISED;

    case OutcomeStatus::FAILED:
    {
        // The frontend rejected the graph: nothing about the engine was learned.
        if(outcome.stage == OutcomeStage::GRAPH)
        {
            return CaseOutcome::UNEXERCISED;
        }

        // A claimed graph the engine is absent for. The claim failure is already the
        // gate (RFC 0015); the outcome only says what the engine did.
        if(outcome.declined)
        {
            return CaseOutcome::DECLINED;
        }

        if(outcome.origin == FailureOrigin::COMPARISON)
        {
            return CaseOutcome::WRONG;
        }

        // Ranked, then failed to build or run. A late NOT_SUPPORTED lands here too:
        // the engine had accepted the graph.
        if(outcome.origin == FailureOrigin::ENGINE
           && (outcome.stage == OutcomeStage::BUILD || outcome.stage == OutcomeStage::EXECUTE))
        {
            return CaseOutcome::CRASH;
        }

        // An oracle or harness error, an unresolved query, an unclassified failure.
        return CaseOutcome::UNEXERCISED;
    }

    default:
        return CaseOutcome::UNEXERCISED;
    }
}

CaseOutcomeRecord makeCaseOutcomeRecord(CaseIdentity identity,
                                        const VerificationOutcome& outcome,
                                        CaseOutcome classified,
                                        bool hadOtherFailure)
{
    CaseOutcomeRecord record;
    record.identity = std::move(identity);
    record.outcome = classified;
    record.stage = outcome.stage;
    record.depth = outcome.depth;
    record.origin = outcome.origin;
    record.verifier = outcome.verifier;

    if(!outcome.message.empty())
    {
        record.reason = outcome.message;
    }
    else if(outcome.alreadyReported)
    {
        record.reason = "output did not match the oracle (per-tensor detail is in the test log)";
    }
    else if(outcome.status == OutcomeStatus::PASSED && hadOtherFailure)
    {
        record.reason = "the test body passed but recorded another failure (see the test log)";
    }
    else if(classified == CaseOutcome::UNEXERCISED && outcome.status == OutcomeStatus::PASSED)
    {
        record.reason = "the test body passed without reaching the depth its bundle requires";
    }
    else if(classified == CaseOutcome::UNCLAIMED_PASS)
    {
        record.reason = "engine supports this graph but has no claim in the sidecar";
    }

    return record;
}

CaseOutcomeRecord
    makeUnexercisedRecord(CaseIdentity identity, OutcomeStage stage, std::string reason)
{
    CaseOutcomeRecord record;
    record.identity = std::move(identity);
    record.outcome = CaseOutcome::UNEXERCISED;
    record.stage = stage;
    record.reason = std::move(reason);
    return record;
}

namespace
{

nlohmann::json identityJson(const char* event, const CaseIdentity& identity)
{
    nlohmann::json json = nlohmann::json::object();
    json["event"] = event;
    json["test"] = identity.test;
    json["bundle"] = identity.bundle;
    return json;
}

// Invalid UTF-8 (an engine message built from raw bytes) must not stop the line from
// being written: a lost record is a case with no outcome.
std::string dumpLine(const nlohmann::json& json)
{
    return json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

} // namespace

std::string startedRecordJson(const CaseIdentity& identity)
{
    return dumpLine(identityJson("started", identity));
}

std::string finishedRecordJson(const CaseOutcomeRecord& record)
{
    auto json = identityJson("finished", record.identity);
    json["engine"] = record.identity.engine;
    json["arch"] = record.identity.arch;
    json["platform"] = record.identity.platform;
    if(record.identity.purpose.has_value())
    {
        json["purpose"] = toString(*record.identity.purpose);
    }
    if(!record.identity.validator.empty())
    {
        json["validator"] = record.identity.validator;
    }
    if(!record.identity.placement.empty())
    {
        json["placement"] = record.identity.placement;
    }
    if(record.identity.placementUid.has_value())
    {
        json["placement_uid"] = *record.identity.placementUid;
    }
    if(record.identity.inputs.is_object() && !record.identity.inputs.empty())
    {
        json["inputs"] = record.identity.inputs;
    }
    json["outcome"] = toString(record.outcome);
    json["stage"] = toString(record.stage);
    json["depth"] = toString(record.depth);
    json["origin"] = toString(record.origin);
    json["verifier"] = toString(record.verifier);
    json["reason"] = record.reason;
    return dumpLine(json);
}

} // namespace hipdnn_integration_tests::bundle

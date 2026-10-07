// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/CatalogSweepTypes.hpp"

#include <cmath>
#include <sstream>

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

// A report is read by scripts: a number that is not finite has no JSON spelling, so it
// is written as null rather than as a value a parser would reject or misread.
nlohmann::json numberOrNull(const std::optional<double>& value)
{
    if(!value.has_value() || !std::isfinite(*value))
    {
        return nullptr;
    }
    return *value;
}

nlohmann::json knobsToJson(const std::vector<CatalogKnob>& knobs)
{
    auto array = nlohmann::json::array();
    for(const auto& knob : knobs)
    {
        nlohmann::json entry;
        entry["knob_id"] = knob.id;
        std::visit([&entry](const auto& value) { entry["value"] = value; }, knob.value);
        array.push_back(std::move(entry));
    }
    return array;
}

} // namespace

const char* toString(CatalogStatus status)
{
    switch(status)
    {
    case CatalogStatus::PASS:
        return "PASS";
    case CatalogStatus::FAIL_NUMERIC:
        return "FAIL_NUMERIC";
    case CatalogStatus::FAIL_BUILD:
        return "FAIL_BUILD";
    case CatalogStatus::FAIL_EXECUTE:
        return "FAIL_EXECUTE";
    case CatalogStatus::FILTERED:
        return "FILTERED";
    case CatalogStatus::NO_CATALOG:
        return "NO_CATALOG";
    default:
        return "UNKNOWN";
    }
}

const char* toString(GraphOutcome outcome)
{
    switch(outcome)
    {
    case GraphOutcome::SWEPT:
        return "SWEPT";
    case GraphOutcome::SKIPPED:
        return "SKIPPED";
    case GraphOutcome::FAILED:
        return "FAILED";
    default:
        return "UNKNOWN";
    }
}

bool isFailure(CatalogStatus status)
{
    return status != CatalogStatus::PASS && status != CatalogStatus::FILTERED;
}

CatalogStatus classifyEntry(const EntryOutcome& outcome)
{
    if(outcome.failedStage.has_value())
    {
        switch(*outcome.failedStage)
        {
        case EntryStage::CREATE_PLAN:
        case EntryStage::CHECK_SUPPORT:
        case EntryStage::BUILD_PLANS:
            return CatalogStatus::FAIL_BUILD;
        case EntryStage::EXECUTE:
            return CatalogStatus::FAIL_EXECUTE;
        case EntryStage::COMPARE:
            return CatalogStatus::FAIL_NUMERIC;
        default:
            return CatalogStatus::FAIL_EXECUTE;
        }
    }
    return outcome.outputMismatch || outcome.nanElements > 0 ? CatalogStatus::FAIL_NUMERIC
                                                             : CatalogStatus::PASS;
}

nlohmann::json toJson(const CatalogRecord& record, const SweepRunContext& context)
{
    nlohmann::json json;
    json["graph"] = record.graph;
    json["engine_id"] = record.engineId;
    json["engine_name"] = record.engineName;
    json["entry_index"] = record.entryIndex;
    json["knobs"] = knobsToJson(record.knobs);
    json["status"] = toString(record.status);
    json["max_abs_err"] = numberOrNull(record.maxAbsErr);
    json["mismatch_pct"] = numberOrNull(record.mismatchPct);
    json["message"] = record.message;
    json["compile_time_ms"] = numberOrNull(record.compileTimeMs);
    json["device_arch"] = context.deviceArch;
    json["hipdnn_version"] = context.hipdnnVersion;
    return json;
}

void GraphTally::noteRecord(CatalogStatus status)
{
    switch(status)
    {
    case CatalogStatus::PASS:
        ++passed;
        break;
    case CatalogStatus::FAIL_NUMERIC:
        ++failNumeric;
        break;
    case CatalogStatus::FAIL_BUILD:
        ++failBuild;
        break;
    case CatalogStatus::FAIL_EXECUTE:
        ++failExecute;
        break;
    case CatalogStatus::NO_CATALOG:
        ++noCatalog;
        break;
    case CatalogStatus::FILTERED:
        ++filtered;
        break;
    default:
        break;
    }
}

std::string formatReconcileLine(const std::string& graph, const GraphTally& tally)
{
    std::ostringstream line;
    line << "reconcile graph=" << graph << " entries=" << tally.returned
         << " = passed=" << tally.passed << " + failed=" << tally.failed()
         << " + filtered=" << tally.filtered << " [fail_numeric=" << tally.failNumeric
         << " fail_build=" << tally.failBuild << " fail_execute=" << tally.failExecute
         << " no_catalog=" << tally.noCatalog << "]";
    if(!tally.reconciles())
    {
        line << " MISMATCH";
    }
    return line.str();
}

nlohmann::json reconcileToJson(const std::string& graph,
                               const GraphTally& tally,
                               GraphOutcome outcome,
                               const std::string& message,
                               const SweepRunContext& context)
{
    nlohmann::json json;
    json["kind"] = "reconcile";
    json["graph"] = graph;
    json["entries"] = tally.returned;
    json["passed"] = tally.passed;
    json["failed"] = tally.failed();
    json["filtered"] = tally.filtered;
    json["fail_numeric"] = tally.failNumeric;
    json["fail_build"] = tally.failBuild;
    json["fail_execute"] = tally.failExecute;
    json["no_catalog"] = tally.noCatalog;
    json["reconciled"] = tally.reconciles();
    json["graph_outcome"] = toString(outcome);
    json["message"] = message;
    json["line"] = formatReconcileLine(graph, tally);
    json["device_arch"] = context.deviceArch;
    json["hipdnn_version"] = context.hipdnnVersion;
    return json;
}

} // namespace hipdnn_integration_tests::catalog_sweep

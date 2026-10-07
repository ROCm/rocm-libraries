// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

// The vocabulary of a catalog sweep and the rules that turn what happened to one entry
// into a status, a record and a graph's reconciliation. Pure values: no device, no
// frontend, no gtest.

namespace hipdnn_integration_tests::catalog_sweep
{

/// What became of one catalog entry (or, for NO_CATALOG, of one engine).
enum class CatalogStatus
{
    PASS, ///< built, executed, and every output matched the reference
    FAIL_NUMERIC, ///< executed, but an output is out of tolerance or holds a NaN
    FAIL_BUILD, ///< plan creation, support check or plan build failed
    FAIL_EXECUTE, ///< the plan built but the run (or its setup) failed
    FILTERED, ///< excluded by --catalog-filter; enumerated and recorded, never run
    NO_CATALOG, ///< the engine returned no catalog for this graph
};

const char* toString(CatalogStatus status);

/// True for the statuses that fail a graph.
bool isFailure(CatalogStatus status);

/// One knob of a catalog entry. The value keeps its type: the report must tell the
/// integer 2 from the string "2", because a knob set identifies an entry.
struct CatalogKnob
{
    std::string id;
    std::variant<std::int64_t, double, std::string> value;
};

/// Where an entry stopped.
enum class EntryStage
{
    CREATE_PLAN, ///< create_execution_plan_ext
    CHECK_SUPPORT, ///< check_support
    BUILD_PLANS, ///< build_plans
    EXECUTE, ///< workspace, input fill, output allocation, execution
    COMPARE, ///< the comparison itself could not run
};

/// Everything the harness learned about running one entry; classifyEntry() turns it into
/// a status. Separated from the run so the decision is testable without a device.
struct EntryOutcome
{
    /// The first stage that failed, or nullopt when every stage ran.
    std::optional<EntryStage> failedStage;
    std::string message;

    /// Meaningful only when every stage ran.
    bool outputMismatch = false;
    std::size_t nanElements = 0;

    /// Worst absolute difference from the reference over all outputs, and the share of
    /// elements out of tolerance (or NaN), in percent. nullopt when the output type has
    /// no element-wise measure.
    std::optional<double> maxAbsErr;
    std::optional<double> mismatchPct;

    /// Milliseconds spent in create_execution_plan_ext + check_support + build_plans.
    /// Engines that compile at plan-build time spend nearly all of their time here.
    std::optional<double> compileTimeMs;
};

/// A failed stage maps to its own failure; a clean run is PASS unless an output
/// mismatched or holds a NaN. A NaN is a failure even when the validator's tolerance
/// would let it through, because a NaN is what an unwritten output looks like.
CatalogStatus classifyEntry(const EntryOutcome& outcome);

/// Run-wide facts every record carries.
struct SweepRunContext
{
    std::string deviceArch;
    std::string hipdnnVersion;
};

/// One line of the report. Field names are the schema.
struct CatalogRecord
{
    std::string graph;
    std::int64_t engineId = 0;
    std::string engineName;
    /// Position in the engine's catalog. Informational: the knob set identifies an entry.
    /// -1 on the record of an engine that has no catalog.
    std::int64_t entryIndex = -1;
    std::vector<CatalogKnob> knobs;
    CatalogStatus status = CatalogStatus::PASS;
    std::optional<double> maxAbsErr;
    std::optional<double> mismatchPct;
    std::string message;
    std::optional<double> compileTimeMs;
};

/// The report line for one entry: graph, engine_id, engine_name, entry_index, knobs,
/// status, max_abs_err, mismatch_pct, message, compile_time_ms, device_arch,
/// hipdnn_version. `knobs` is an array of {knob_id, value}.
nlohmann::json toJson(const CatalogRecord& record, const SweepRunContext& context);

/// How a graph's entries add up. `returned` is counted when a catalog is read -- one for
/// each entry an engine returned, one for an engine that returned none -- and the
/// outcomes are counted as records are written, so the two sides are independent and
/// `reconciles()` can actually fail: it is the proof that no returned entry went
/// unreported.
struct GraphTally
{
    std::size_t returned = 0;
    std::size_t passed = 0;
    std::size_t failNumeric = 0;
    std::size_t failBuild = 0;
    std::size_t failExecute = 0;
    std::size_t noCatalog = 0;
    std::size_t filtered = 0;

    /// An engine without a catalog counts as one failed slot, so the identity covers it.
    std::size_t failed() const
    {
        return failNumeric + failBuild + failExecute + noCatalog;
    }

    void noteReturned(std::size_t count)
    {
        returned += count;
    }

    void noteRecord(CatalogStatus status);

    /// entries = passed + failed + filtered.
    bool reconciles() const
    {
        return returned == passed + failed() + filtered;
    }
};

/// The one line printed and recorded per graph:
/// `reconcile graph=<g> entries=12 = passed=10 + failed=2 + filtered=0 [fail_numeric=1
/// fail_build=0 fail_execute=0 no_catalog=1]`, with ` MISMATCH` appended when the identity
/// does not hold.
std::string formatReconcileLine(const std::string& graph, const GraphTally& tally);

/// How a graph's sweep ended, apart from its entries.
enum class GraphOutcome
{
    SWEPT, ///< every entry that was returned was run or filtered and recorded
    SKIPPED, ///< the graph cannot be swept here (no reference, no such engine); not a failure
    FAILED, ///< the sweep could not be carried out (graph or reference would not load)
};

const char* toString(GraphOutcome outcome);

/// The summary record of a graph, written beside (never inside) its entry records and
/// told apart by `"kind": "reconcile"`.
nlohmann::json reconcileToJson(const std::string& graph,
                               const GraphTally& tally,
                               GraphOutcome outcome,
                               const std::string& message,
                               const SweepRunContext& context);

} // namespace hipdnn_integration_tests::catalog_sweep

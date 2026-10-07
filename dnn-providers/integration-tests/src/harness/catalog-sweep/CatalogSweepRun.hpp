// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "harness/catalog-sweep/CatalogReportWriter.hpp"
#include "harness/catalog-sweep/CatalogSweep.hpp"
#include "harness/catalog-sweep/CatalogSweepOptions.hpp"
#include "harness/catalog-sweep/CatalogSweepTypes.hpp"

namespace hipdnn_integration_tests::catalog_sweep
{

/// What a whole sweep run adds up to, across graphs.
struct SweepTotals
{
    std::size_t graphsSwept = 0; ///< graphs whose entries were enumerated and reported
    std::size_t graphsSkipped = 0; ///< graphs that cannot be swept here
    std::size_t graphsFailed = 0; ///< graphs with a failed entry, a missing catalog or an error
    GraphTally entries; ///< the per-graph tallies, summed
};

/// The state every graph of one sweep run shares: where the report goes, which entries
/// are run, what the run adds up to.
class CatalogSweepRun
{
public:
    CatalogSweepRun(CatalogSweepOptions options,
                    SweepRunContext context,
                    std::optional<std::string> pinnedEngine)
        : _options(std::move(options))
        , _context(std::move(context))
        , _pinnedEngine(std::move(pinnedEngine))
        , _writer(_options.reportPath)
    {
    }

    const CatalogSweepOptions& options() const
    {
        return _options;
    }

    const SweepRunContext& context() const
    {
        return _context;
    }

    const std::optional<std::string>& pinnedEngine() const
    {
        return _pinnedEngine;
    }

    const SweepTotals& totals() const
    {
        return _totals;
    }

    RecordSink sink()
    {
        return [this](const nlohmann::json& record) { _writer.append(record); };
    }

    /// Adds a swept graph to the totals.
    void note(const GraphSweepResult& result)
    {
        switch(result.outcome)
        {
        case GraphOutcome::SKIPPED:
            ++_totals.graphsSkipped;
            break;
        case GraphOutcome::FAILED:
            ++_totals.graphsFailed;
            break;
        case GraphOutcome::SWEPT:
        default:
            ++_totals.graphsSwept;
            if(result.failed())
            {
                ++_totals.graphsFailed;
            }
            break;
        }
        _totals.entries.returned += result.tally.returned;
        _totals.entries.passed += result.tally.passed;
        _totals.entries.failNumeric += result.tally.failNumeric;
        _totals.entries.failBuild += result.tally.failBuild;
        _totals.entries.failExecute += result.tally.failExecute;
        _totals.entries.noCatalog += result.tally.noCatalog;
        _totals.entries.filtered += result.tally.filtered;
    }

    /// A graph this run cannot sweep for a reason known before any work: the report still
    /// says so, with an empty reconciliation, so every graph the run selected is accounted
    /// for.
    void noteSkipped(const std::string& graph, const std::string& reason, std::ostream& log)
    {
        const GraphTally empty;
        log << "catalog sweep: " << graph << " skipped: " << reason << "\n"
            << formatReconcileLine(graph, empty) << "\n";
        _writer.append(reconcileToJson(graph, empty, GraphOutcome::SKIPPED, reason, _context));
        note({empty, GraphOutcome::SKIPPED, reason});
    }

private:
    CatalogSweepOptions _options;
    SweepRunContext _context;
    std::optional<std::string> _pinnedEngine;
    CatalogReportWriter _writer;
    SweepTotals _totals;
};

} // namespace hipdnn_integration_tests::catalog_sweep

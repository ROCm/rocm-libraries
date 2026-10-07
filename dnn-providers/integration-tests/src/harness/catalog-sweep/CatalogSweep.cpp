// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/CatalogSweep.hpp"

#include <algorithm>
#include <exception>
#include <ostream>
#include <utility>

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

constexpr const char* K_FILTERED_MESSAGE = "excluded by --catalog-filter";

/// Collects what one graph's sweep writes, so the tally and the report cannot disagree:
/// a record is counted exactly when it is handed to the sink.
class GraphReporter
{
public:
    GraphReporter(const GraphSweepRequest& request, const RecordSink& sink, std::ostream& log)
        : _request(request)
        , _sink(sink)
        , _log(log)
    {
    }

    GraphTally& tally()
    {
        return _tally;
    }

    void record(CatalogRecord record)
    {
        record.graph = _request.graph;
        _tally.noteRecord(record.status);
        _log << "  [" << toString(record.status) << "] engine=" << record.engineName
             << " entry=" << record.entryIndex;
        if(!record.message.empty() && record.status != CatalogStatus::FILTERED)
        {
            _log << ": " << record.message;
        }
        _log << "\n";
        _sink(toJson(record, _request.context));
    }

    GraphSweepResult finish(GraphOutcome outcome, std::string message)
    {
        const auto line = formatReconcileLine(_request.graph, _tally);
        _log << line << "\n";
        _sink(reconcileToJson(_request.graph, _tally, outcome, message, _request.context));
        return {_tally, outcome, std::move(message)};
    }

private:
    const GraphSweepRequest& _request;
    const RecordSink& _sink;
    std::ostream& _log;
    GraphTally _tally;
};

CatalogRecord baseRecord(const CatalogEngine& engine, std::int64_t entryIndex)
{
    CatalogRecord record;
    record.engineId = engine.id;
    record.engineName = engine.name;
    record.entryIndex = entryIndex;
    return record;
}

EntryOutcome
    runGuarded(ICatalogGraphRunner& runner, const CatalogEngine& engine, std::size_t entryIndex)
{
    // An entry that throws is still an entry: losing it would break the reconciliation
    // and hide the one result that most needs a look.
    try
    {
        return runner.runEntry(engine, entryIndex);
    }
    catch(const std::exception& e)
    {
        EntryOutcome outcome;
        outcome.failedStage = EntryStage::EXECUTE;
        outcome.message = std::string("unexpected exception: ") + e.what();
        return outcome;
    }
    catch(...)
    {
        EntryOutcome outcome;
        outcome.failedStage = EntryStage::EXECUTE;
        outcome.message = "unexpected non-standard exception";
        return outcome;
    }
}

} // namespace

GraphSweepResult sweepGraph(ICatalogGraphRunner& runner,
                            const GraphSweepRequest& request,
                            const RecordSink& sink,
                            std::ostream& log)
{
    GraphReporter reporter(request, sink, log);
    const CatalogFilter unfiltered;
    const CatalogFilter& filter = request.filter != nullptr ? *request.filter : unfiltered;

    log << "catalog sweep: " << request.graph << "\n";

    // A graph no engine takes is an answer, not a break: skipped, as the verification run
    // skips it, unless --fail-on-unsupported asks for the opposite.
    const auto unsupported = [&](std::string message) {
        return reporter.finish(request.failOnUnsupported ? GraphOutcome::FAILED
                                                         : GraphOutcome::SKIPPED,
                               std::move(message));
    };

    const auto listed = runner.listEngines();
    if(!listed.ok && listed.notSupported)
    {
        return unsupported("no engine supports this graph");
    }
    if(!listed.ok)
    {
        return reporter.finish(GraphOutcome::FAILED, "listing engines failed: " + listed.error);
    }

    std::vector<CatalogEngine> engines;
    for(const auto& engine : listed.engines)
    {
        if(!request.pinnedEngine.has_value() || engine.name == *request.pinnedEngine)
        {
            engines.push_back(engine);
        }
    }
    if(engines.empty())
    {
        return unsupported(request.pinnedEngine.has_value()
                               ? "engine " + *request.pinnedEngine + " does not list this graph"
                               : "no engine supports this graph");
    }

    // The reference is the expensive part and says nothing about a graph no engine will run,
    // so it comes last: after the engines are known, and only when at least one of them has
    // a (graph, engine) pair the filter lets through. A graph nothing serves is therefore
    // never failed by a reference problem.
    const bool anyRuns = std::any_of(engines.begin(), engines.end(), [&](const CatalogEngine& e) {
        return filter.matches(request.graph, e.name);
    });
    if(anyRuns)
    {
        const auto reference = runner.prepareReference();
        if(reference.status == ReferenceResult::Status::UNAVAILABLE)
        {
            return reporter.finish(GraphOutcome::SKIPPED, reference.message);
        }
        if(reference.status == ReferenceResult::Status::FAILED)
        {
            return reporter.finish(GraphOutcome::FAILED, reference.message);
        }
    }

    for(const auto& engine : engines)
    {
        const bool run = filter.matches(request.graph, engine.name);
        const auto catalog = runner.fetchCatalog(engine);

        // An engine with no entries to offer is one reported slot: a failure when the
        // engine was meant to run, a FILTERED record when it was not (the filter must not
        // turn an engine the user excluded into a failure of the graph).
        if(!catalog.ok || catalog.entries.empty())
        {
            const std::string why
                = catalog.ok ? "the engine returned an empty catalog" : catalog.error;
            reporter.tally().noteReturned(1);

            auto record = baseRecord(engine, -1);
            if(run)
            {
                record.status = CatalogStatus::NO_CATALOG;
                record.message = why;
            }
            else
            {
                record.status = CatalogStatus::FILTERED;
                record.message = std::string(K_FILTERED_MESSAGE) + " (no catalog: " + why + ")";
            }
            reporter.record(std::move(record));
            continue;
        }

        reporter.tally().noteReturned(catalog.entries.size());
        for(std::size_t index = 0; index < catalog.entries.size(); ++index)
        {
            auto record = baseRecord(engine, static_cast<std::int64_t>(index));
            record.knobs = catalog.entries[index].knobs;

            if(!run)
            {
                record.status = CatalogStatus::FILTERED;
                record.message = K_FILTERED_MESSAGE;
            }
            else
            {
                const auto outcome = runGuarded(runner, engine, index);
                record.status = classifyEntry(outcome);
                record.message = outcome.message;
                record.maxAbsErr = outcome.maxAbsErr;
                record.mismatchPct = outcome.mismatchPct;
                record.compileTimeMs = outcome.compileTimeMs;
            }
            reporter.record(std::move(record));
        }
    }

    const bool reconciles = reporter.tally().reconciles();
    return reporter.finish(GraphOutcome::SWEPT,
                           reconciles ? std::string{}
                                      : "entries returned by the engines were not all reported");
}

} // namespace hipdnn_integration_tests::catalog_sweep

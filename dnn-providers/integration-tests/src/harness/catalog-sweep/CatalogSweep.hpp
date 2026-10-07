// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/catalog-sweep/CatalogSweepOptions.hpp"
#include "harness/catalog-sweep/CatalogSweepTypes.hpp"

namespace hipdnn_integration_tests::catalog_sweep
{

struct CatalogEngine
{
    std::int64_t id = 0;
    std::string name;
};

struct CatalogEntry
{
    std::vector<CatalogKnob> knobs;
};

/// What preparing the reference came to.
struct ReferenceResult
{
    enum class Status
    {
        READY,
        UNAVAILABLE, ///< no reference can run this graph; the graph cannot be swept here
        FAILED, ///< the graph or the reference broke; a defect, not a gap
    };
    Status status = Status::READY;
    std::string message;
};

struct EngineList
{
    bool ok = true;
    /// Set with `ok == false` when the failure is the graph not being supported by any
    /// engine (the frontend's GRAPH_NOT_SUPPORTED, the answer the verification harness
    /// also treats as "declined" rather than "broke"). Any other failure leaves it false.
    bool notSupported = false;
    std::string error;
    std::vector<CatalogEngine> engines;
};

struct CatalogFetch
{
    bool ok = true;
    std::string error;
    std::vector<CatalogEntry> entries;
};

/// Everything a sweep of one graph asks of the graph, the device and the reference.
///
/// One seam, so the loop that decides what is run, what is filtered and what is
/// reported -- the part that carries the accounting -- is exercised without a GPU. An
/// implementation owns the graph for the whole sweep: the same object serves the
/// reference, the engine listing, every catalog fetch and every entry.
class ICatalogGraphRunner
{
public:
    ICatalogGraphRunner() = default;
    virtual ~ICatalogGraphRunner() = default;

    ICatalogGraphRunner(const ICatalogGraphRunner&) = delete;
    ICatalogGraphRunner& operator=(const ICatalogGraphRunner&) = delete;
    ICatalogGraphRunner(ICatalogGraphRunner&&) = delete;
    ICatalogGraphRunner& operator=(ICatalogGraphRunner&&) = delete;

    /// Builds the inputs once, runs the reference once, and keeps its outputs on the host.
    virtual ReferenceResult prepareReference() = 0;

    /// Every engine the graph lists, in the backend's order.
    virtual EngineList listEngines() = 0;

    /// The engine's knob catalog for this graph. A failure is the engine having none.
    virtual CatalogFetch fetchCatalog(const CatalogEngine& engine) = 0;

    /// Runs entry `entryIndex` of the catalog last fetched for `engine`: creates the plan,
    /// checks support, builds it, refills the inputs, resets the outputs, executes and
    /// compares. Reports rather than throws.
    virtual EntryOutcome runEntry(const CatalogEngine& engine, std::size_t entryIndex) = 0;
};

struct GraphSweepRequest
{
    std::string graph;
    const CatalogFilter* filter = nullptr; ///< null runs everything
    /// --test-engine: sweep only this engine. Not a filter: the other engines are out of
    /// scope for the run, as they are for every other mode of this binary.
    std::optional<std::string> pinnedEngine;
    /// --fail-on-unsupported: a graph no (pinned) engine supports fails instead of being
    /// skipped, as in the verification run.
    bool failOnUnsupported = false;
    SweepRunContext context;
};

/// Called once per report line, in order.
using RecordSink = std::function<void(const nlohmann::json&)>;

struct GraphSweepResult
{
    GraphTally tally;
    GraphOutcome outcome = GraphOutcome::SWEPT;
    std::string message;

    /// Any failed entry, any engine without a catalog, a sweep that could not be carried
    /// out, or entries that were returned but not accounted for.
    bool failed() const
    {
        return outcome == GraphOutcome::FAILED || tally.failed() > 0 || !tally.reconciles();
    }
};

/// Sweeps one graph: reference, engines, catalogs, entries; writes one record per entry
/// and one reconcile record, and prints the reconcile line to `log`.
///
/// Every entry an engine returns is reported -- run and classified, or FILTERED -- and
/// never skipped; an entry whose run throws is reported FAIL_EXECUTE rather than lost.
/// The engines are listed first; the reference runs once afterwards, and only when at
/// least one listed engine has a (graph, engine) pair the filter lets through. A graph no
/// engine serves (or the pinned engine does not list) is skipped without a reference.
GraphSweepResult sweepGraph(ICatalogGraphRunner& runner,
                            const GraphSweepRequest& request,
                            const RecordSink& sink,
                            std::ostream& log);

} // namespace hipdnn_integration_tests::catalog_sweep

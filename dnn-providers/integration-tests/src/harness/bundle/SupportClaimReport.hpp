// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "harness/bundle/SupportVerdict.hpp"

namespace hipdnn_integration_tests::bundle
{

// Process-wide collector of support-claim verdicts. Mirrors UnverifiableBundleReport:
// thread-safe Meyers singleton populated during test execution, printed once after
// RUN_ALL_TESTS().
//
// Storage: every verdict lands in one vector and callers filter by SupportVerdict at
// read time. Do not partition storage by verdict — the verdict is already a field on
// the record, so a second encoding in the choice of container is redundant, and the
// two encodings drift. Keeping one vector also means a new SupportVerdict enumerator
// requires no change to this class at all.
//
// Progressive display:
//   Level 1 (always): one-line counter summary
//   Level 2 (if failures): per-bundle failure detail
//   Level 3 (if unclaimed): the bundles that are supported but unclaimed
//
// Also provides the empty-query guard (RFC 0015 §7.2): enforcement is requested
// but zero support queries were observed in the whole run.
class SupportClaimReport
{
public:
    static SupportClaimReport& get()
    {
        static SupportClaimReport s_instance;
        return s_instance;
    }

    SupportClaimReport(const SupportClaimReport&) = delete;
    SupportClaimReport& operator=(const SupportClaimReport&) = delete;
    SupportClaimReport(SupportClaimReport&&) = delete;
    SupportClaimReport& operator=(SupportClaimReport&&) = delete;

    // Record one verdict. Callers must not pass NO_SIDECAR — a graph with no
    // support.json has nothing to report, and admitting those records would both
    // bury the real verdicts and break getGraphsWithClaimsQueried() below.
    void record(const SupportResult& result)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.push_back(result);
    }

    // Both are counted at *discovery* time, before any decision to run or drop
    // the graph, and both are run-level events rather than per-record ones.
    // That placement is the whole design: a claim-bearing graph that fails to
    // load registers no GTest case and so produces no SupportResult at all, so
    // counting at discovery makes that silent drop *arm* the guard below instead
    // of disarming it. This is exactly why these two cannot be derived from
    // _records — they count graphs that may never appear there.

    // Every graph found on disk.
    void recordGraphFound()
    {
        _graphsFound.fetch_add(1, std::memory_order_relaxed);
    }

    // ...of which these have a .support.json beside them.
    void recordGraphWithClaimsFound()
    {
        _graphsWithClaimsFound.fetch_add(1, std::memory_order_relaxed);
    }

    // ...of which these we actually got to query. Counted once per graph at
    // the call site, not once per engine — multi-engine enforcement produces
    // N records per graph, so deriving from _records.size() would overcount.
    void recordGraphQueried()
    {
        _graphsQueried.fetch_add(1, std::memory_order_relaxed);
    }

    size_t getGraphsWithClaimsQueried() const
    {
        return _graphsQueried.load(std::memory_order_relaxed);
    }

    // RFC 0015 §7.2. True when the run found graphs carrying support claims but
    // never actually asked the backend about a single one of them — no GPU, the
    // plugin failed to load, no --test-engine, every graph filtered out.
    // Enforcement then "passes" having verified nothing, so main() turns this
    // into a hard failure.
    bool claimsFoundButNoneQueried() const
    {
        return getGraphsWithClaimsFound() > 0 && getGraphsWithClaimsQueried() == 0;
    }

    size_t count(SupportVerdict verdict) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return static_cast<size_t>(
            std::count_if(_records.begin(), _records.end(), [verdict](const SupportResult& r) {
                return r.verdict == verdict;
            }));
    }

    bool hasFailures() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return std::any_of(_records.begin(), _records.end(), [](const SupportResult& r) {
            return isFailure(r.verdict);
        });
    }

    size_t getTotalRecorded() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _records.size();
    }

    // Level 1: counters. Level 2: failure detail. Level 3: unclaimed bundles.
    void print(std::ostream& os = std::cerr) const
    {
        std::vector<SupportResult> records;
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            records = _records;
        }

        // One snapshot, one pass per verdict — so the Level 1 counters and the
        // Level 2/3 listings below can never disagree with each other.
        const auto tally = [&records](SupportVerdict verdict) {
            return static_cast<size_t>(
                std::count_if(records.begin(), records.end(), [verdict](const SupportResult& r) {
                    return r.verdict == verdict;
                }));
        };

        const size_t sat = tally(SupportVerdict::SATISFIED);
        const size_t broke = tally(SupportVerdict::CLAIM_BROKEN);
        const size_t err = tally(SupportVerdict::QUERY_ERRORED);
        const size_t unc = tally(SupportVerdict::UNCLAIMED_SUPPORT);
        const size_t notEnf = tally(SupportVerdict::NOT_ENFORCED);

        // Stay silent only when the run had nothing to enforce: not one sidecar
        // anywhere, so not one record either. That is the pre-adoption state, and
        // it is every run in the tree today. Once a sidecar exists we print even
        // if there is nothing to say about it, because "enforcing nothing" is a
        // result the user needs to see rather than silence that reads as success.
        if(records.empty() && getGraphsWithClaimsFound() == 0)
        {
            return;
        }

        // Level 1: one-line summary. The discovery counts come first because they
        // are what makes a run of all-zeros legible: 12 graphs carrying claims and
        // 0 queried is a broken run, while 0 and 0 is simply a run with nothing
        // to enforce.
        os << "\n==== SUPPORT CLAIM SUMMARY ====\n"
           << "  graphs: " << getGraphsFound() << " found, " << getGraphsWithClaimsFound()
           << " with claims, " << records.size() << " queried\n"
           << "  satisfied: " << sat << "  broken: " << broke << "  errored: " << err
           << "  unclaimed: " << unc << "  not-enforced: " << notEnf << "\n";

        // Level 2: failure detail (only when failures exist)
        if(broke + err > 0)
        {
            os << "\n---- CLAIM FAILURES (" << (broke + err) << ") ----\n";
            for(const auto& r : records)
            {
                if(!isFailure(r.verdict))
                {
                    continue;
                }
                os << "  " << toString(r.verdict) << "  " << r.bundlePath << "\n"
                   << "    engine=" << r.engineName << "  arch=" << r.arch
                   << "  platform=" << r.platform << "\n"
                   << "    " << r.detail << "\n";
                if(!r.queryMessage.empty())
                {
                    os << "    query: " << r.queryMessage << "\n";
                }
            }
        }

        // Level 3: unclaimed support. Names the bundles, because a bare count
        // tells the reader nothing they can act on.
        if(unc > 0)
        {
            os << "\n---- UNCLAIMED SUPPORT (" << unc << ") ----\n";
            for(const auto& r : records)
            {
                if(r.verdict != SupportVerdict::UNCLAIMED_SUPPORT)
                {
                    continue;
                }
                os << "  " << r.bundlePath << "\n"
                   << "    engine=" << r.engineName << "  arch=" << r.arch
                   << "  platform=" << r.platform << "\n";
            }
            os << "\nThese are supported but not recorded in a sidecar.\n";
        }
    }

    void reset()
    {
        _graphsFound.store(0, std::memory_order_relaxed);
        _graphsWithClaimsFound.store(0, std::memory_order_relaxed);
        _graphsQueried.store(0, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

    size_t getGraphsFound() const
    {
        return _graphsFound.load(std::memory_order_relaxed);
    }
    size_t getGraphsWithClaimsFound() const
    {
        return _graphsWithClaimsFound.load(std::memory_order_relaxed);
    }

private:
    SupportClaimReport() = default;

    mutable std::mutex _mutex;
    std::vector<SupportResult> _records; // every verdict; filter by SupportVerdict when reading

    // found ⊇ withClaimsFound ⊇ queried

    std::atomic<size_t> _graphsFound{0}; // every graph on disk
    std::atomic<size_t> _graphsWithClaimsFound{0}; // ...of which these have a sidecar
    std::atomic<size_t> _graphsQueried{0}; // ...of which these we actually queried
};

} // namespace hipdnn_integration_tests::bundle

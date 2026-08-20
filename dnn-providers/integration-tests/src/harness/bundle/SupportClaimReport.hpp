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

// Thread-safe singleton collecting support-claim verdicts. Printed once after
// RUN_ALL_TESTS(). Also provides the empty-query guard: enforcement requested
// but zero support queries observed.
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

    void record(const SupportResult& result)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.push_back(result);
    }

    // Counted at discovery time, not derivable from _records (a graph that
    // fails to load produces no SupportResult but still arms the guard).
    void recordGraphFound()
    {
        _graphsFound.fetch_add(1, std::memory_order_relaxed);
    }

    void recordGraphWithClaims()
    {
        _graphsWithClaims.fetch_add(1, std::memory_order_relaxed);
    }

    // Counted once per graph, not per engine.
    void recordGraphWithClaimsVerified()
    {
        _graphsWithClaimsVerified.fetch_add(1, std::memory_order_relaxed);
    }

    size_t getGraphsWithClaimsVerified() const
    {
        return _graphsWithClaimsVerified.load(std::memory_order_relaxed);
    }

    // RFC 0015 §7.2. True when the run found graphs carrying support claims but
    // never actually asked the backend about a single one of them — no GPU, the
    // plugin failed to load, no --test-engine, every graph filtered out.
    // Enforcement then "passes" having verified nothing, so main() turns this
    // into a hard failure.
    bool claimsFoundButNoneVerified() const
    {
        return getGraphsWithClaims() > 0 && getGraphsWithClaimsVerified() == 0;
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

        const auto tally = [&records](SupportVerdict verdict) {
            return static_cast<size_t>(
                std::count_if(records.begin(), records.end(), [verdict](const SupportResult& r) {
                    return r.verdict == verdict;
                }));
        };

        const size_t sat = tally(SupportVerdict::SATISFIED);
        const size_t broke = tally(SupportVerdict::CLAIM_BROKEN);
        const size_t err = tally(SupportVerdict::QUERY_ERRORED);
        const size_t notLoaded = tally(SupportVerdict::ENGINE_NOT_LOADED);
        const size_t unc = tally(SupportVerdict::UNCLAIMED_SUPPORT);
        const size_t notEnf = tally(SupportVerdict::NOT_ENFORCED);

        if(records.empty() && getGraphsWithClaims() == 0)
        {
            return;
        }

        os << "\n==== SUPPORT CLAIM SUMMARY ====\n"
           << "  graphs: " << getGraphsFound() << " found, " << getGraphsWithClaims()
           << " with claims, " << getGraphsWithClaimsVerified() << " queried (" << records.size()
           << " verdicts)\n"
           << "  satisfied: " << sat << "  broken: " << broke << "  errored: " << err
           << "  not-loaded: " << notLoaded << "  unclaimed: " << unc
           << "  not-enforced: " << notEnf << "\n";

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
        _graphsWithClaims.store(0, std::memory_order_relaxed);
        _graphsWithClaimsVerified.store(0, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

    size_t getGraphsFound() const
    {
        return _graphsFound.load(std::memory_order_relaxed);
    }
    size_t getGraphsWithClaims() const
    {
        return _graphsWithClaims.load(std::memory_order_relaxed);
    }

private:
    SupportClaimReport() = default;

    mutable std::mutex _mutex;
    std::vector<SupportResult> _records;

    // found ⊇ withClaims ⊇ withClaimsVerified
    std::atomic<size_t> _graphsFound{0};
    std::atomic<size_t> _graphsWithClaims{0};
    std::atomic<size_t> _graphsWithClaimsVerified{0};
};

} // namespace hipdnn_integration_tests::bundle

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Process-wide, run-level singletons for RFC 0015's §7.1 (unclaimed-support
// summary) and §7.2 (empty-query guard). Both mirror the existing
// UnverifiableBundleReport / SupportMatrixCollector pattern: populated during
// test execution, consumed once after RUN_ALL_TESTS() in main().
namespace hipdnn_integration_tests::bundle
{

// RFC 0015 §7.2: the run-level empty-query guard. If enforcement was
// requested (at least one claim-bearing bundle was registered) but zero
// support queries were observed anywhere in the whole run (no GPU, plugin
// failed to load, an over-narrow --gtest_filter), the run must FAIL loudly
// rather than report green — a claim that is never queried is silently
// unenforced, and this is the only floor that catches "nothing ran at all".
class SupportQueryGuard
{
public:
    static SupportQueryGuard& get()
    {
        static SupportQueryGuard s_instance;
        return s_instance;
    }

    SupportQueryGuard(const SupportQueryGuard&) = delete;
    SupportQueryGuard& operator=(const SupportQueryGuard&) = delete;
    SupportQueryGuard(SupportQueryGuard&&) = delete;
    SupportQueryGuard& operator=(SupportQueryGuard&&) = delete;

    // Called once per claim-bearing bundle at registration time.
    void noteClaimBearingBundleRegistered()
    {
        _claimBearingBundleCount.fetch_add(1, std::memory_order_relaxed);
    }

    // Called once per support query observed during test execution
    // (get_ranked_engine_ids at any rung: applicability, buildable, or full).
    void noteQueryObserved()
    {
        _queriesObserved.fetch_add(1, std::memory_order_relaxed);
    }

    // True iff at least one claim-bearing bundle was registered this run --
    // i.e. enforcement was expected to happen somewhere.
    bool enforcementExpected() const
    {
        return _claimBearingBundleCount.load(std::memory_order_relaxed) != 0;
    }

    std::size_t claimBearingBundleCount() const
    {
        return _claimBearingBundleCount.load(std::memory_order_relaxed);
    }

    std::size_t queriesObserved() const
    {
        return _queriesObserved.load(std::memory_order_relaxed);
    }

    // True iff the guard should fail the run: enforcement was expected but
    // literally nothing queried support.
    bool tripped() const
    {
        return enforcementExpected() && queriesObserved() == 0;
    }

    void reset()
    {
        _claimBearingBundleCount.store(0, std::memory_order_relaxed);
        _queriesObserved.store(0, std::memory_order_relaxed);
    }

private:
    SupportQueryGuard() = default;

    std::atomic<std::size_t> _claimBearingBundleCount{0};
    std::atomic<std::size_t> _queriesObserved{0};
};

// RFC 0015 §7.1: one "supported but unclaimed" observation -- a claim-bearing
// bundle whose engine turned out to support an (engine, arch, platform) combo
// the bundle's support.json does not list. Informational only: printed as an
// end-of-run summary, never a FAIL and never a SKIP change (the within-run,
// human-readable twin of the §12 CI harvest).
struct UnclaimedSupportRecord
{
    std::string bundle;
    std::string engine;
    std::string arch;
    std::string platform;
};

class UnclaimedSupportReport
{
public:
    static UnclaimedSupportReport& get()
    {
        static UnclaimedSupportReport s_instance;
        return s_instance;
    }

    UnclaimedSupportReport(const UnclaimedSupportReport&) = delete;
    UnclaimedSupportReport& operator=(const UnclaimedSupportReport&) = delete;
    UnclaimedSupportReport(UnclaimedSupportReport&&) = delete;
    UnclaimedSupportReport& operator=(UnclaimedSupportReport&&) = delete;

    void record(std::string bundle, std::string engine, std::string arch, std::string platform)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.push_back(
            {std::move(bundle), std::move(engine), std::move(arch), std::move(platform)});
    }

    std::vector<UnclaimedSupportRecord> getRecords() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _records;
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

    // Prints the summary to `os`. No-op when nothing was recorded.
    void print(std::ostream& os = std::cout) const
    {
        std::vector<UnclaimedSupportRecord> records;
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            records = _records;
        }

        if(records.empty())
        {
            return;
        }

        os << "\n==== UNCLAIMED SUPPORT (RFC 0015 \xc2\xa7"
              "7.1) ====\n"
           << "The following engines support a claim-bearing bundle's graph on an\n"
           << "(engine, arch, platform) its support.json does not list. Not a failure --\n"
           << "run --write-support-claims to record the observed coverage if it is durable.\n";
        for(const auto& record : records)
        {
            os << "  " << record.bundle << ": engine=" << record.engine << " arch=" << record.arch
               << " platform=" << record.platform << "\n";
        }
    }

private:
    UnclaimedSupportReport() = default;

    mutable std::mutex _mutex;
    std::vector<UnclaimedSupportRecord> _records;
};

} // namespace hipdnn_integration_tests::bundle

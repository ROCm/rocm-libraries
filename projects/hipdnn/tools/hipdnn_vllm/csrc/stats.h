// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Process-wide coverage counters. Counters are keyed by
// (op, route, reason, phase); phase is "capture" while the current stream is
// capturing a HIP graph, else "eager". Counter cells are never freed, so hot
// paths keep a pointer to their cell and increment it atomically with no lock.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_vllm
{

inline constexpr const char* ROUTE_HIPDNN = "hipdnn";
inline constexpr const char* ROUTE_FALLBACK = "fallback";

struct CounterRow
{
    std::string op;
    std::string route;
    std::string reason;
    std::string phase;
    uint64_t count;
};

struct RuntimeTotals
{
    uint64_t planBuilds;
    uint64_t planBuildDeclines;
    uint64_t planBuildNs;
    uint64_t planCacheHits;
    uint64_t planCacheEvictions;
    uint64_t executeCalls;
    uint64_t executeHostNs;
};

class Stats
{
public:
    static Stats& instance();

    // Returns the stable cell for a key, creating it on first use.
    std::atomic<uint64_t>* cell(const std::string& op,
                                const std::string& route,
                                const std::string& reason,
                                bool capture);

    void count(const std::string& op,
               const std::string& route,
               const std::string& reason,
               bool capture)
    {
        cell(op, route, reason, capture)->fetch_add(1, std::memory_order_relaxed);
    }

    void planBuilt(uint64_t ns, bool declined)
    {
        _planBuilds.fetch_add(1, std::memory_order_relaxed);
        _planBuildNs.fetch_add(ns, std::memory_order_relaxed);
        if(declined)
        {
            _planBuildDeclines.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void planCacheHit()
    {
        _planCacheHits.fetch_add(1, std::memory_order_relaxed);
    }

    void planCacheEviction()
    {
        _planCacheEvictions.fetch_add(1, std::memory_order_relaxed);
    }

    void executed(uint64_t hostNs)
    {
        _executeCalls.fetch_add(1, std::memory_order_relaxed);
        _executeHostNs.fetch_add(hostNs, std::memory_order_relaxed);
    }

    std::vector<CounterRow> rows() const;
    RuntimeTotals totals() const;
    void reset();

private:
    struct Cell
    {
        CounterRow key;
        std::atomic<uint64_t> value{0};
    };

    mutable std::mutex _mutex;
    std::unordered_map<std::string, std::unique_ptr<Cell>> _cells;

    std::atomic<uint64_t> _planBuilds{0};
    std::atomic<uint64_t> _planBuildDeclines{0};
    std::atomic<uint64_t> _planBuildNs{0};
    std::atomic<uint64_t> _planCacheHits{0};
    std::atomic<uint64_t> _planCacheEvictions{0};
    std::atomic<uint64_t> _executeCalls{0};
    std::atomic<uint64_t> _executeHostNs{0};
};

} // namespace hipdnn_vllm

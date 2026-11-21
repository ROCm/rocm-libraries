// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stats.h"

#include <algorithm>
#include <tuple>

namespace hipdnn_vllm
{

Stats& Stats::instance()
{
    // Never destroyed, like the plan cache and the runtimes: leaked plan
    // entries and static cell pointers (PlanEntry cells, the PARTIAL cells)
    // point into _cells, and a late call from a non-Python thread at exit must
    // not touch a destroyed map.
    static auto* s_stats = new Stats();
    return *s_stats;
}

std::atomic<uint64_t>* Stats::cell(const std::string& op,
                                   const std::string& route,
                                   const std::string& reason,
                                   bool capture)
{
    const char* phase = capture ? "capture" : "eager";
    std::string key;
    key.reserve(op.size() + route.size() + reason.size() + 12);
    key.append(op).push_back('\x1f');
    key.append(route).push_back('\x1f');
    key.append(reason).push_back('\x1f');
    key.append(phase);

    const std::lock_guard<std::mutex> lock(_mutex);
    auto it = _cells.find(key);
    if(it == _cells.end())
    {
        auto fresh = std::make_unique<Cell>();
        fresh->key = CounterRow{.op = op, .route = route, .reason = reason, .phase = phase, .count = 0};
        it = _cells.emplace(std::move(key), std::move(fresh)).first;
    }
    return &it->second->value;
}

std::vector<CounterRow> Stats::rows() const
{
    std::vector<CounterRow> out;
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        out.reserve(_cells.size());
        for(const auto& [key, cellPtr] : _cells)
        {
            const uint64_t value = cellPtr->value.load(std::memory_order_relaxed);
            if(value == 0)
            {
                continue;
            }
            CounterRow row = cellPtr->key;
            row.count = value;
            out.push_back(std::move(row));
        }
    }
    std::ranges::sort(out, [](const CounterRow& a, const CounterRow& b) {
        return std::tie(a.op, a.route, a.reason, a.phase)
               < std::tie(b.op, b.route, b.reason, b.phase);
    });
    return out;
}

RuntimeTotals Stats::totals() const
{
    return RuntimeTotals{
        .planBuilds = _planBuilds.load(std::memory_order_relaxed),
        .planBuildDeclines = _planBuildDeclines.load(std::memory_order_relaxed),
        .planBuildNs = _planBuildNs.load(std::memory_order_relaxed),
        .planCacheHits = _planCacheHits.load(std::memory_order_relaxed),
        .planCacheEvictions = _planCacheEvictions.load(std::memory_order_relaxed),
        .executeCalls = _executeCalls.load(std::memory_order_relaxed),
        .executeHostNs = _executeHostNs.load(std::memory_order_relaxed),
    };
}

void Stats::reset()
{
    {
        // Cells are zeroed, not erased: hot paths hold pointers to them.
        const std::lock_guard<std::mutex> lock(_mutex);
        for(auto& [key, cellPtr] : _cells)
        {
            cellPtr->value.store(0, std::memory_order_relaxed);
        }
    }
    _planBuilds.store(0, std::memory_order_relaxed);
    _planBuildDeclines.store(0, std::memory_order_relaxed);
    _planBuildNs.store(0, std::memory_order_relaxed);
    _planCacheHits.store(0, std::memory_order_relaxed);
    _planCacheEvictions.store(0, std::memory_order_relaxed);
    _executeCalls.store(0, std::memory_order_relaxed);
    _executeHostNs.store(0, std::memory_order_relaxed);
}

} // namespace hipdnn_vllm

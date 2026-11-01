// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// LRU plan cache. Values are a built plan or the Decline from the one build
// attempt, so a declined signature costs one build per process.
//
// Pinned entries: a plan a HIP graph captured is replayed by that graph, which
// never touches the cache, so eviction (capacity, setCapacity, clear) would
// destroy a plan whose launches are still replayed. pin() keeps such entries
// alive for the life of the process, independent of the LRU.

#pragma once

#include "decline.h"

#include <hipdnn_frontend.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hipdnn_vllm
{

struct BuiltPlan
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    int64_t workspaceBytes = 0;
    std::string engineName;
    // Slot i of an execute call binds tensor uid slotUids[i].
    std::vector<int64_t> slotUids;
    // Reused uid -> pointer map. Its nodes are allocated once at build; execute
    // only rewrites the values (under the Runtime's lock).
    std::unordered_map<int64_t, void*> variantPack;
};

struct PlanEntry
{
    std::string op;
    std::shared_ptr<BuiltPlan> plan; // null when declined
    std::optional<Decline> decline;
    // Stats cells for this entry's op and outcome, resolved once at build.
    std::atomic<uint64_t>* eagerCell = nullptr;
    std::atomic<uint64_t>* captureCell = nullptr;
};

class PlanCache
{
public:
    explicit PlanCache(size_t capacity)
        : _capacity(capacity == 0 ? 1 : capacity)
    {
    }

    std::shared_ptr<PlanEntry> get(const std::string& key);
    void put(const std::string& key, std::shared_ptr<PlanEntry> entry);
    void setCapacity(size_t capacity);
    size_t size() const;
    size_t capacity() const;
    // Drops the LRU only; pinned entries stay alive.
    void clear();
    void pin(const std::shared_ptr<PlanEntry>& entry);
    size_t pinnedCount() const;

private:
    void evictLocked();

    using Lru = std::list<std::pair<std::string, std::shared_ptr<PlanEntry>>>;

    mutable std::mutex _mutex;
    size_t _capacity;
    Lru _lru; // front = most recently used
    std::unordered_map<std::string, Lru::iterator> _index;
    std::unordered_set<std::shared_ptr<PlanEntry>> _pinned;
};

} // namespace hipdnn_vllm

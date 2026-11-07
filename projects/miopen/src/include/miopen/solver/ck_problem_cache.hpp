// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/env.hpp>

#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>

/// Set to 0 to bypass the per-problem CK caches (CKProblemCache users: the CK plugin's
/// applicability answers and the CK backward-weights default configurations), so that every
/// query computes its answer again, as before the caches existed.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_PROBLEM_CACHE)

namespace miopen {
namespace solver {

/// A per-process memo of answers that depend only on a problem, shared by all threads. It holds
/// at most `capacity` problems: storing one more first clears it, so a workload with an unbounded
/// number of distinct problems recomputes instead of growing. An answer is computed outside the
/// lock; threads racing on a new problem compute the same answer, and the first one stored stays.
template <class Map>
class CKProblemCache
{
public:
    using Key   = typename Map::key_type;
    using Value = typename Map::mapped_type;

    /// Problems remembered before starting over. An entry takes a few hundred bytes, so a full
    /// cache stays within a few hundred MB even for the largest workloads.
    static constexpr std::size_t DefaultCapacity = 500000;

    explicit CKProblemCache(std::size_t capacity_ = DefaultCapacity) : capacity(capacity_) {}

    CKProblemCache(const CKProblemCache&)            = delete;
    CKProblemCache& operator=(const CKProblemCache&) = delete;

    /// The stored answer for `key`, or `compute()`'s, which is stored unless it is empty.
    template <class Compute>
    std::optional<Value> GetOrCompute(const Key& key, Compute&& compute)
    {
        if(env::disabled(MIOPEN_DEBUG_CK_PROBLEM_CACHE))
            return compute();
        {
            std::shared_lock lock{mutex};
            if(const auto it = entries.find(key); it != entries.end())
                return it->second;
        }
        std::optional<Value> computed = compute();
        if(computed)
        {
            std::unique_lock lock{mutex};
            if(entries.size() >= capacity && entries.find(key) == entries.end())
                entries.clear();
            entries.try_emplace(key, *computed);
        }
        return computed;
    }

    std::size_t Size() const
    {
        std::shared_lock lock{mutex};
        return entries.size();
    }

private:
    std::size_t capacity;
    mutable std::shared_mutex mutex;
    Map entries;
};

} // namespace solver
} // namespace miopen

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Measure-and-cache selection store for the AOT catalog engine (Phase 2). Maps a
// canonical problem key -> the symbol of the fastest kernel measured for that
// problem, so a given (family, problem) is tuned once and every subsequent plan
// for the same problem skips straight to the winner.
//
// The cache is process-lifetime (owned by the engine) and optionally persisted
// to a JSON file so the winner survives across runs. Persistence is best-effort:
// a missing or garbage file loads as an empty cache, and a failed write is
// logged and ignored -- correctness never depends on the file.

#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "catalog/CatalogTypes.hpp"

namespace aot_catalog_engine::catalog
{

// Canonical key for a tuning decision: "<family>|k1=v1,k2=v2,...", the ordered
// ProblemShape rendered deterministically (ShapeValue by variant type). The
// family name already encodes arch+dtype so keys never collide across families.
std::string problemKey(const std::string& family, const ProblemShape& problem);

class TuneCache
{
public:
    // Default: resolve the persistence path from the HIPDNN_AOT_TUNE_CACHE
    // environment variable, else <temp_dir>/hipdnn_aot_tune_cache.json. Loads
    // the file on construction (no-throw: missing/garbage -> empty).
    TuneCache();

    // Explicit-path ctor (tests): persist to `path` (may be empty to disable
    // persistence entirely). Loads it on construction if present.
    explicit TuneCache(std::string path);

    // The winning symbol previously recorded for `key`, or nullopt on a miss.
    std::optional<std::string> lookup(const std::string& key) const;

    // Record `symbol` as the winner for `key` (measured at `ms` median), and
    // persist the whole cache if a path is configured. `ms` is stored for
    // diagnostics/persistence only.
    void store(const std::string& key, const std::string& symbol, double ms);

    // The configured persistence path (may be empty).
    const std::string& path() const
    {
        return _path;
    }

private:
    struct Entry
    {
        std::string symbol;
        double ms = 0.0;
    };

    // No-throw file load into _entries (called under _mutex from a ctor).
    void loadLocked();
    // Best-effort atomic write of _entries (called under _mutex from store()).
    void saveLocked() const;

    mutable std::mutex _mutex;
    std::map<std::string, Entry> _entries;
    std::string _path;
};

} // namespace aot_catalog_engine::catalog

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The command-line surface of --sweep-knob-catalog, as pure values and decisions: how
// a shard is spelled and which graphs it owns, what a filter expression means, which
// environments the sweep refuses to start in. Nothing here touches a device, a handle
// or gtest, so every rule is unit-testable.

namespace hipdnn_integration_tests::catalog_sweep
{

/// `--catalog-shard <i>/<n>`: this process sweeps the graphs shard `index` of `count`
/// owns.
///
/// The shard unit is the *graph*, not the entry. A graph's entries are only known after
/// its catalog has been fetched, which needs the graph built and the reference run, so
/// partitioning by entry would make every shard pay for every graph. Ownership is a
/// stable hash of the graph's gtest name, so the partition does not depend on discovery
/// order, on how many other graphs exist, or on --gtest_filter: running the same shard
/// twice, or on two hosts, selects the same graphs. Every graph belongs to exactly one
/// shard, so the union of the shards' reports covers each graph exactly once.
struct ShardSpec
{
    std::size_t index = 0;
    std::size_t count = 1;
};

/// Parses "<i>/<n>" (both decimal, 0 <= i < n, n >= 1).
/// @throws std::invalid_argument naming the offending text.
ShardSpec parseShard(std::string_view text);

/// Whether this shard owns `graphName` (the bundle's "Suite.Test" gtest name).
bool shardOwns(const ShardSpec& shard, std::string_view graphName);

/// `--catalog-filter <expr>`: which (graph, engine) pairs are *run*. Every other pair is
/// still enumerated, and each of its entries is reported FILTERED -- the filter narrows
/// the work, never the accounting.
///
/// Grammar:
/// @code
///   expr   := clause (',' clause)*      all clauses must hold
///   clause := field op values
///   field  := 'graph' | 'engine'        the gtest "Suite.Test" name / the engine name
///   op     := '=' | '!=' | '~' | '!~'   glob match / its negation / substring / its negation
///   values := pattern ('|' pattern)*    any alternative may match
/// @endcode
/// Globs match the whole name, case-sensitively: `*` is any run of characters, `?` is
/// exactly one. Whitespace around fields, operators and values is ignored. A pattern
/// cannot contain ',' or '|'. Examples: `engine=ROCKE*`, `graph~Pointwise,engine!=MIOPEN_ENGINE`,
/// `engine=A|B`.
class CatalogFilter
{
public:
    /// The filter that runs everything.
    CatalogFilter() = default;

    /// @throws std::invalid_argument describing the first malformed clause.
    static CatalogFilter parse(std::string_view expression);

    /// True when every clause holds for this pair.
    bool matches(std::string_view graph, std::string_view engine) const;

    /// True when every `graph` clause holds, whatever the engine. A graph that fails this
    /// has no run entries under any engine, so nothing needs to be executed for it.
    bool matchesGraph(std::string_view graph) const;

    bool isUnfiltered() const
    {
        return _clauses.empty();
    }

private:
    enum class Field
    {
        GRAPH,
        ENGINE,
    };
    enum class Match
    {
        GLOB,
        SUBSTRING,
    };
    struct Clause
    {
        Field field = Field::GRAPH;
        Match match = Match::GLOB;
        bool negated = false;
        std::vector<std::string> patterns;
    };

    static bool clauseHolds(const Clause& clause, std::string_view value);

    std::vector<Clause> _clauses;
};

/// Everything the sweep needs from the command line, once validated.
struct CatalogSweepOptions
{
    /// JSON Lines, appended to; never truncated.
    std::filesystem::path reportPath;
    CatalogFilter filter;
    ShardSpec shard;
};

/// The raw flag values, before any is validated.
struct CatalogSweepRequest
{
    bool sweep = false; ///< --sweep-knob-catalog
    std::optional<std::string> reportPath; ///< --catalog-report
    std::optional<std::string> filter; ///< --catalog-filter
    std::optional<std::string> shard; ///< --catalog-shard
};

/// Exactly one of the two members is set, or neither when the sweep was not requested.
struct CatalogSweepResolution
{
    std::optional<CatalogSweepOptions> options;
    std::optional<std::string> error;
};

/// Decides what the flags ask for. Refuses, rather than ignores, a --catalog-* flag
/// given without --sweep-knob-catalog, and a sweep without --catalog-report: a sweep
/// whose results go nowhere has done nothing.
CatalogSweepResolution resolveCatalogSweepOptions(const CatalogSweepRequest& request);

/// An environment variable's value, or nullopt when it is not set. Distinguishes a
/// variable that is set to the empty string from one that is absent.
using EnvLookup = std::function<std::optional<std::string>(const std::string& name)>;

/// The process environment, through hipdnn's own platform layer.
EnvLookup processEnvironment();

/// Why the sweep must not start in this environment, or nullopt when it may.
///
/// - `HIPDNN_FORCE_BENCHMARKING` must be unset. *Any* value, `0` included, counts as set:
///   it overrides the `global.benchmarking` knob process-wide, so a stray value changes
///   which plan every catalog entry builds.
/// - `HIPDNN_CACHE_DIR` must name a fresh directory: absent on disk, or an existing
///   empty directory. hipDNN persists engine rankings under it, and a ranking recorded
///   by an earlier run would change which kernels a later run is offered or built.
///   An empty value is as good as unset, because hipDNN itself treats it as unset and
///   falls back to the per-user default cache.
std::optional<std::string> checkSweepEnvironment(const EnvLookup& env);

} // namespace hipdnn_integration_tests::catalog_sweep

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

/// @file PoolAssembly.hpp
/// @brief Several source pools into one corpus: deduplicate, allocate, order, take.
///
/// Op-general: works on @ref ProblemPoint and names no operation or parameter.
namespace hipdnn_corpus_gen
{

/// Source precedence for deduplication and manifest order: a recorded model shape is the most
/// useful provenance for an audit, then a pack geometry, then a sample.
inline const std::vector<std::string>& corpusSources()
{
    static const std::vector<std::string> s_sources{"model", "kernel", "sweep"};
    return s_sources;
}

/// Default corpus share per source. Kernel geometries get the most because the pack was
/// compiled for them. Preferences, not quotas: see @ref allocate.
inline const std::map<std::string, double>& defaultShares()
{
    static const std::map<std::string, double> s_shares{
        {"model", 0.15}, {"kernel", 0.60}, {"sweep", 0.25}};
    return s_shares;
}

/// Whether @p shares gives @p source a share above zero; unnamed sources are off.
/// Check before collecting a pool, or a disabled pool still shrinks what the search is asked
/// to find.
inline bool sourceEnabled(const std::map<std::string, double>& shares, const std::string& source)
{
    const auto found = shares.find(source);
    return found != shares.end() && found->second > 0.0;
}

/// One candidate problem, with the provenance that makes the corpus auditable.
struct PoolEntry
{
    ProblemPoint point;

    /// Which pool it came from: one of @ref corpusSources.
    std::string source;

    /// Where inside that pool (pack file, model name, draw index). Free text: sources differ.
    std::string origin;

    /// The stratification label (see RegimeLabel.hpp), carried so ordering and reporting agree.
    std::string regime;

    /// What a cut is spread over: the categorical combination plus regime; empty means regime
    /// alone. Without the combination, a cut can drop whole dtypes.
    std::string stratum;
};

/// The pools, keyed by source name.
using SourcePools = std::map<std::string, std::vector<PoolEntry>>;

/// @brief How many problems each source contributes, given each source's capacity.
///
/// Shares first; any shortfall is redistributed round-robin over enabled pools with room, so
/// no single source absorbs it. A share of 0 excludes its source; it is never refilled.
inline std::map<std::string, int64_t> allocate(int64_t count,
                                               const std::map<std::string, int64_t>& capacity,
                                               const std::map<std::string, double>& shares)
{
    const auto shareOf = [&shares](const std::string& source) {
        const auto found = shares.find(source);
        return found == shares.end() ? 0.0 : found->second;
    };

    double total = 0.0;
    for(const auto& entry : capacity)
    {
        total += shareOf(entry.first);
    }
    if(total <= 0.0)
    {
        total = 1.0;
    }

    std::map<std::string, int64_t> allocation;
    int64_t assigned = 0;
    for(const auto& entry : capacity)
    {
        const auto wanted
            = static_cast<int64_t>(static_cast<double>(count) * shareOf(entry.first) / total);
        allocation[entry.first] = std::min(entry.second, wanted);
        assigned += allocation[entry.first];
    }

    auto remaining = count - assigned;
    while(remaining > 0)
    {
        std::vector<std::string> open;
        for(const auto& source : corpusSources())
        {
            const auto found = capacity.find(source);
            if(found != capacity.end() && allocation[source] < found->second
               && sourceEnabled(shares, source))
            {
                open.push_back(source);
            }
        }
        if(open.empty())
        {
            break;
        }
        for(const auto& source : open)
        {
            if(remaining == 0)
            {
                break;
            }
            ++allocation[source];
            --remaining;
        }
    }
    return allocation;
}

/// @brief One entry per distinct problem (all declared parameters), earlier sources winning.
///
/// Sources @p shares disables are dropped first, so they cannot claim an enabled source's
/// point. @p dropped receives the per-source duplicate count, for the manifest.
inline SourcePools deduplicate(const SourcePools& pools,
                               const std::map<std::string, double>& shares,
                               std::map<std::string, int64_t>& dropped)
{
    std::set<std::string> seen;
    SourcePools unique;
    dropped.clear();

    for(const auto& source : corpusSources())
    {
        std::vector<PoolEntry> kept;
        int64_t duplicates = 0;
        const auto found = pools.find(source);
        if(found != pools.end() && sourceEnabled(shares, source))
        {
            for(const auto& entry : found->second)
            {
                if(!seen.insert(detail::describe(entry.point)).second)
                {
                    ++duplicates;
                    continue;
                }
                kept.push_back(entry);
            }
        }
        unique[source] = std::move(kept);
        dropped[source] = duplicates;
    }
    return unique;
}

namespace detail
{

/// One pool reordered so any prefix holds the pool's stratum mix in proportion, keeping each
/// stratum's internal order. Pools often arrive grouped, so a raw prefix would be biased.
inline std::vector<PoolEntry> spread(const std::vector<PoolEntry>& pool)
{
    // First-appearance order, not sorted: ties between equal-size strata must not depend on
    // label collation.
    std::vector<std::string> order;
    std::map<std::string, std::vector<PoolEntry>> buckets;
    for(const auto& entry : pool)
    {
        const auto& key = entry.stratum.empty() ? entry.regime : entry.stratum;
        if(buckets.find(key) == buckets.end())
        {
            order.push_back(key);
        }
        buckets[key].push_back(entry);
    }

    struct Placed
    {
        double position;
        size_t rank;
        size_t index;
        const PoolEntry* entry;
    };

    std::vector<Placed> placed;
    placed.reserve(pool.size());
    for(size_t rank = 0; rank < order.size(); ++rank)
    {
        const auto& members = buckets.at(order[rank]);
        const auto span = static_cast<double>(members.size());
        for(size_t index = 0; index < members.size(); ++index)
        {
            placed.push_back(Placed{(static_cast<double>(index) * 2.0 + 1.0) / (span * 2.0),
                                    rank,
                                    index,
                                    &members[index]});
        }
    }

    std::stable_sort(placed.begin(), placed.end(), [](const Placed& left, const Placed& right) {
        if(left.position != right.position)
        {
            return left.position < right.position;
        }
        if(left.rank != right.rank)
        {
            return left.rank < right.rank;
        }
        return left.index < right.index;
    });

    std::vector<PoolEntry> result;
    result.reserve(placed.size());
    for(const auto& row : placed)
    {
        result.push_back(*row.entry);
    }
    return result;
}

} // namespace detail

/// What a regime quota asked for and what the pools could give it.
struct RegimeQuotaOutcome
{
    int64_t asked = 0;
    int64_t taken = 0;
};

/// @brief The corpus: regime quotas first, then the rest of @p count cut by @p shares.
///
/// Every pool is spread (see @ref detail::spread) and cut from its front, in source
/// precedence. @p count below the quotas' sum does not trim them. @p allocation receives
/// each source's total ask; @p quotaOutcome each regime's asked and taken (an unfilled
/// quota is a finding about what the engine serves).
inline std::vector<PoolEntry> select(const SourcePools& pools,
                                     int64_t count,
                                     const std::map<std::string, double>& shares,
                                     std::map<std::string, int64_t>& allocation,
                                     const std::map<std::string, int64_t>& quotas,
                                     std::map<std::string, RegimeQuotaOutcome>& quotaOutcome)
{
    SourcePools ordered;
    for(const auto& source : corpusSources())
    {
        const auto found = pools.find(source);
        // Spread every pool before cutting, so a prefix is not just whatever arrived first.
        ordered[source]
            = detail::spread(found == pools.end() ? std::vector<PoolEntry>{} : found->second);
    }

    std::map<std::string, std::vector<bool>> chosen;
    std::map<std::string, int64_t> fromQuotas;
    quotaOutcome.clear();
    for(const auto& [regime, asked] : quotas)
    {
        auto& outcome = quotaOutcome[regime];
        outcome.asked = asked;
        for(const auto& source : corpusSources())
        {
            const auto& pool = ordered[source];
            auto& marks = chosen[source];
            marks.resize(pool.size(), false);
            for(size_t i = 0; i < pool.size() && outcome.taken < asked; ++i)
            {
                if(!marks[i] && pool[i].regime == regime)
                {
                    marks[i] = true;
                    ++outcome.taken;
                    ++fromQuotas[source];
                }
            }
        }
    }

    int64_t quotaTotal = 0;
    for(const auto& entry : quotaOutcome)
    {
        quotaTotal += entry.second.taken;
    }

    // What the quotas left, still in spread order, cut by shares for the remainder of `count`.
    std::map<std::string, int64_t> capacity;
    std::map<std::string, std::vector<size_t>> remaining;
    for(const auto& source : corpusSources())
    {
        auto& marks = chosen[source];
        marks.resize(ordered[source].size(), false);
        for(size_t i = 0; i < marks.size(); ++i)
        {
            if(!marks[i])
            {
                remaining[source].push_back(i);
            }
        }
        capacity[source] = static_cast<int64_t>(remaining[source].size());
    }
    allocation = allocate(std::max<int64_t>(0, count - quotaTotal), capacity, shares);
    for(const auto& source : corpusSources())
    {
        const auto take = std::min(static_cast<size_t>(std::max<int64_t>(0, allocation[source])),
                                   remaining[source].size());
        for(size_t i = 0; i < take; ++i)
        {
            chosen[source][remaining[source][i]] = true;
        }
        allocation[source] += fromQuotas[source];
    }

    std::vector<PoolEntry> selected;
    for(const auto& source : corpusSources())
    {
        const auto& pool = ordered[source];
        for(size_t i = 0; i < pool.size(); ++i)
        {
            if(chosen[source][i])
            {
                selected.push_back(pool[i]);
            }
        }
    }
    return selected;
}

/// @ref select without regime quotas.
inline std::vector<PoolEntry> select(const SourcePools& pools,
                                     int64_t count,
                                     const std::map<std::string, double>& shares,
                                     std::map<std::string, int64_t>& allocation)
{
    std::map<std::string, RegimeQuotaOutcome> unused;
    return select(pools, count, shares, allocation, {}, unused);
}

} // namespace hipdnn_corpus_gen

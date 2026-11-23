// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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

    /// The quota key (`<axis>:<label>`, see formQuotaKey) each form axis puts it under. A form
    /// floor counts it beside its regime: forms cross regimes, so one entry serves both.
    std::vector<std::string> forms;

    /// What a cut is spread over: the categorical combination plus regime; empty means regime
    /// alone. Without the combination, a cut can drop whole dtypes.
    std::string stratum;

    /// What a weighted cut allocates between: the categorical combination. Empty means each
    /// stratum is a group of its own. Within a group the strata keep their members' mix.
    std::string group;

    /// The group's relative size in a cut (see weighGroups); 0 means its member count, so a
    /// pool that sets none is cut in proportion to what it holds.
    double weight = 0.0;

    /// Found for a regime quota, or reserved as a served edge: the run owes it by name. Kept
    /// even when its source's share is 0, because a share sizes a source's proportional cut,
    /// not what a quota or `--include-extremes` asks for.
    bool owed = false;
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

/// @brief One entry per distinct problem, earlier sources winning.
///
/// Two entries are one problem when @p identity gives them the same value; by default, when
/// every declared parameter agrees. Entries of a source @p shares disables are dropped first,
/// so they cannot claim an enabled source's point, except entries marked @ref PoolEntry::owed,
/// which are kept. @p dropped receives the per-source duplicate count, for the manifest.
inline SourcePools deduplicate(
    const SourcePools& pools,
    const std::map<std::string, double>& shares,
    std::map<std::string, int64_t>& dropped,
    const std::function<std::string(const PoolEntry&)>& identity
    = [](const PoolEntry& entry) { return detail::describe(entry.point); })
{
    std::set<std::string> seen;
    SourcePools unique;
    dropped.clear();

    for(const auto& source : corpusSources())
    {
        std::vector<PoolEntry> kept;
        int64_t duplicates = 0;
        const auto found = pools.find(source);
        if(found != pools.end())
        {
            const bool enabled = sourceEnabled(shares, source);
            for(const auto& entry : found->second)
            {
                if(!enabled && !entry.owed)
                {
                    continue;
                }
                if(!seen.insert(identity(entry)).second)
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

/// How weighGroups weighed a pool.
struct GroupWeighing
{
    /// Groups were set: the declaration weights some parameter. Otherwise the pool is cut in
    /// proportion to what it holds, as recorded shapes are.
    bool weighed = false;

    /// The unweighted enum and bool parameters left as the pool's combinations carry them, not
    /// at equal shares: the declared constraints tie their values to others' (a causal mask
    /// only with one anchor, say), so equal shares would starve whole combinations.
    std::vector<std::string> uneven;
};

namespace detail
{

/// How far @p parameter's values are from equal shares of @p sizes over @p combinations: the
/// largest gap between a value's share and an equal share, over the values that hold any.
inline double gapFromEqualShares(const Parameter& parameter,
                                 const std::vector<ProblemPoint>& combinations,
                                 const std::vector<double>& sizes)
{
    std::map<std::string, double> mass;
    double total = 0.0;
    for(size_t index = 0; index < combinations.size(); ++index)
    {
        const auto found = combinations[index].find(parameter.name);
        if(found != combinations[index].end() && sizes[index] > 0.0)
        {
            mass[describe({*found})] += sizes[index];
            total += sizes[index];
        }
    }
    double gap = 0.0;
    for(const auto& [value, held] : mass)
    {
        gap = std::max(gap, std::abs(held / total - 1.0 / static_cast<double>(mass.size())));
    }
    return gap;
}

} // namespace detail

/// Sets the group and weight of every entry of @p pool, so that a cut of the pool holds each
/// weighted categorical value at its declared share instead of at its share of what the pool
/// happens to hold: a group is a categorical combination, weighing its size under
/// combinationSizes with every unweighted enum and bool parameter's values at equal shares.
/// When equal shares are unreachable under the declared constraints (the raking approaches
/// them only by taking some combination toward 0, and does not settle), the parameter furthest
/// from them is left as the combinations carry it, one at a time until the raking settles, so
/// no combination drops out of the cut. A declaration with no weights is left unweighed. Only
/// for pools of generated problems; recorded shapes (a model's, a kernel pack's) are what they
/// are and are cut in proportion to what the pool holds.
inline GroupWeighing weighGroups(std::vector<PoolEntry>& pool, const OperationMetadata& metadata)
{
    GroupWeighing weighing;
    if(!detail::anyWeighted(metadata.parameters))
    {
        return weighing;
    }
    weighing.weighed = true;
    std::map<std::string, size_t> index;
    std::vector<ProblemPoint> combinations;
    for(auto& entry : pool)
    {
        ProblemPoint combination;
        for(const auto& parameter : metadata.parameters)
        {
            if(parameter.enumerable().empty())
            {
                continue;
            }
            const auto held = entry.point.find(parameter.name);
            if(held != entry.point.end())
            {
                combination.insert(*held);
            }
        }
        entry.group = detail::describe(combination);
        if(index.emplace(entry.group, combinations.size()).second)
        {
            combinations.push_back(std::move(combination));
        }
    }

    std::set<std::string> even;
    for(const auto& parameter : metadata.parameters)
    {
        if(parameter.weights.empty() && !parameter.enumerable().empty())
        {
            even.insert(parameter.name);
        }
    }
    bool settled = false;
    auto sizes = detail::combinationSizes(metadata, combinations, even, &settled);
    while(!settled && !even.empty())
    {
        const Parameter* furthest = nullptr;
        double gap = -1.0;
        for(const auto& parameter : metadata.parameters)
        {
            if(even.count(parameter.name) == 0)
            {
                continue;
            }
            const auto own = detail::gapFromEqualShares(parameter, combinations, sizes);
            if(own > gap)
            {
                gap = own;
                furthest = &parameter;
            }
        }
        even.erase(furthest->name);
        weighing.uneven.push_back(furthest->name);
        sizes = detail::combinationSizes(metadata, combinations, even, &settled);
    }
    for(auto& entry : pool)
    {
        entry.weight = sizes[index.at(entry.group)];
    }
    return weighing;
}

namespace detail
{

/// The bucket a spread places @p entry in when its pool has no groups: its stratum, or its
/// regime when it has none.
inline const std::string& strataKey(const PoolEntry& entry)
{
    return entry.stratum.empty() ? entry.regime : entry.stratum;
}

/// @p entries interleaved so any prefix holds each bucket (as @p bucketOf names it) in
/// proportion to the bucket's weight (@p weightOf, given its members), each bucket keeping its
/// members' order. The k-th member of a bucket of weight W sits at (k + s) / W, with s in
/// [0, 1) the start @p startOf gives the bucket (its rank by first appearance, its members):
/// a prefix holds about W times its length of members from every bucket still holding any. A
/// start of 1/2 rounds each bucket to nearest; staggered starts let the buckets whose share of
/// a short prefix is below one member each take their turn, as systematic sampling does,
/// instead of the heaviest always winning it.
template <typename BucketOf, typename WeightOf, typename StartOf>
std::vector<const PoolEntry*> interleave(const std::vector<const PoolEntry*>& entries,
                                         const BucketOf& bucketOf,
                                         const WeightOf& weightOf,
                                         const StartOf& startOf)
{
    // First-appearance order, not sorted: ties between equal-size buckets must not depend on
    // label collation.
    std::vector<std::string> order;
    std::map<std::string, std::vector<const PoolEntry*>> buckets;
    for(const auto* entry : entries)
    {
        const auto& key = bucketOf(*entry);
        auto found = buckets.find(key);
        if(found == buckets.end())
        {
            order.push_back(key);
            found = buckets.emplace(key, std::vector<const PoolEntry*>{}).first;
        }
        found->second.push_back(entry);
    }

    struct Placed
    {
        double position;
        size_t rank;
        size_t index;
        const PoolEntry* entry;
    };

    std::vector<Placed> placed;
    placed.reserve(entries.size());
    for(size_t rank = 0; rank < order.size(); ++rank)
    {
        const auto& members = buckets.at(order[rank]);
        const auto span = weightOf(members);
        const auto start = startOf(rank, members);
        for(size_t index = 0; index < members.size(); ++index)
        {
            placed.push_back(Placed{(static_cast<double>(index) * 2.0 + 2.0 * start) / (span * 2.0),
                                    rank,
                                    index,
                                    members[index]});
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

    std::vector<const PoolEntry*> result;
    result.reserve(placed.size());
    for(const auto& row : placed)
    {
        result.push_back(row.entry);
    }
    return result;
}

/// One pool reordered so any prefix holds the pool's stratum mix in proportion, keeping each
/// stratum's internal order. Pools often arrive grouped, so a raw prefix would be biased.
///
/// Entries that name a @ref PoolEntry::group are ordered in two levels: the strata of a group
/// in proportion to their members, and the groups in proportion to their weights. Without
/// groups, each stratum is a group of its own and weighs its members.
inline std::vector<PoolEntry> spread(const std::vector<PoolEntry>& pool)
{
    const auto groupKey = [](const PoolEntry& entry) -> const std::string& {
        return entry.group.empty() ? strataKey(entry) : entry.group;
    };
    const auto byMembers = [](const std::vector<const PoolEntry*>& members) {
        return static_cast<double>(members.size());
    };

    const auto nearest = [](size_t, const std::vector<const PoolEntry*>&) { return 0.5; };

    std::vector<const PoolEntry*> all;
    all.reserve(pool.size());
    for(const auto& entry : pool)
    {
        all.push_back(&entry);
    }

    // The strata inside each group, the groups in first-appearance order.
    std::vector<std::string> order;
    std::map<std::string, std::vector<const PoolEntry*>> groups;
    for(const auto* entry : all)
    {
        const auto& key = groupKey(*entry);
        auto found = groups.find(key);
        if(found == groups.end())
        {
            order.push_back(key);
            found = groups.emplace(key, std::vector<const PoolEntry*>{}).first;
        }
        found->second.push_back(entry);
    }
    std::vector<const PoolEntry*> grouped;
    grouped.reserve(pool.size());
    for(const auto& key : order)
    {
        for(const auto* entry : interleave(groups.at(key), strataKey, byMembers, nearest))
        {
            grouped.push_back(entry);
        }
    }

    const auto byWeight = [&byMembers](const std::vector<const PoolEntry*>& members) {
        return members.front()->weight > 0.0 ? members.front()->weight : byMembers(members);
    };
    // A weighted group starts at the fractional part of its rank times the golden ratio, so
    // the starts of any run of ranks are spread over [0, 1).
    const auto staggered = [](size_t rank, const std::vector<const PoolEntry*>& members) {
        if(!(members.front()->weight > 0.0))
        {
            return 0.5;
        }
        const auto phase = static_cast<double>(rank + 1) * 0.6180339887498949;
        return phase - std::floor(phase);
    };
    std::vector<PoolEntry> result;
    result.reserve(pool.size());
    for(const auto* entry : interleave(grouped, groupKey, byWeight, staggered))
    {
        result.push_back(*entry);
    }
    return result;
}

/// Whether @p entry counts toward the quota @p key: its regime, or one of its forms.
inline bool carries(const PoolEntry& entry, const std::string& key)
{
    return entry.regime == key
           || std::find(entry.forms.begin(), entry.forms.end(), key) != entry.forms.end();
}

} // namespace detail

/// What a regime quota asked for and what the pools could give it.
struct RegimeQuotaOutcome
{
    int64_t asked = 0;
    int64_t taken = 0;
};

/// @brief The corpus: quotas first, then the rest of @p count cut by @p shares.
///
/// A quota is keyed by a regime label or by a form key (see PoolEntry::forms); an entry counts
/// toward every quota whose key it carries, so one graph can fill a regime's quota and a
/// form's. Every pool is spread (see @ref detail::spread) and cut from its front, in source
/// precedence. @p count below the quotas' sum does not trim them. @p allocation receives
/// each source's total ask; @p quotaOutcome each quota's asked and taken (an unfilled
/// quota is a finding about what the engine serves). @p reserved names points (by
/// `detail::describe`) taken first, the served edges; they count against @p count and
/// towards the quotas they carry.
inline std::vector<PoolEntry> select(const SourcePools& pools,
                                     int64_t count,
                                     const std::map<std::string, double>& shares,
                                     std::map<std::string, int64_t>& allocation,
                                     const std::map<std::string, int64_t>& quotas,
                                     std::map<std::string, RegimeQuotaOutcome>& quotaOutcome,
                                     const std::set<std::string>& reserved = {})
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
    int64_t quotaTotal = 0;
    for(const auto& source : corpusSources())
    {
        const auto& pool = ordered[source];
        auto& marks = chosen[source];
        marks.resize(pool.size(), false);
        for(size_t i = 0; i < pool.size() && !reserved.empty(); ++i)
        {
            if(reserved.count(detail::describe(pool[i].point)) > 0)
            {
                marks[i] = true;
                ++fromQuotas[source];
                ++quotaTotal;
            }
        }
    }
    quotaOutcome.clear();
    for(const auto& [key, asked] : quotas)
    {
        auto& outcome = quotaOutcome[key];
        outcome.asked = asked;
        // Already marked -- a served edge, or another quota's pick -- and carrying this key.
        int64_t carried = 0;
        for(const auto& source : corpusSources())
        {
            const auto& pool = ordered[source];
            for(size_t i = 0; i < pool.size(); ++i)
            {
                carried += chosen[source][i] && detail::carries(pool[i], key) ? 1 : 0;
            }
        }
        outcome.taken = std::min(asked, carried);
        for(const auto& source : corpusSources())
        {
            const auto& pool = ordered[source];
            auto& marks = chosen[source];
            for(size_t i = 0; i < pool.size() && outcome.taken < asked; ++i)
            {
                if(!marks[i] && detail::carries(pool[i], key))
                {
                    marks[i] = true;
                    ++outcome.taken;
                    ++fromQuotas[source];
                    ++quotaTotal;
                }
            }
        }
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

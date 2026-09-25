// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

/// @file PoolAssembly.hpp
/// @brief Several source pools into one corpus: deduplicate, allocate, order, take.
///
/// The exploration in ProblemSpace.hpp answers "what does this operation admit". It does not
/// answer "which of those problems should be measured", and the two are different questions.
/// A corpus drawn only from the declared space covers what nobody wrote down and under-weights
/// what production actually runs; a corpus drawn only from recorded model shapes is a list of
/// what was already known and says nothing about the rest of the region. So the corpus is
/// assembled from named pools with declared shares, and the manifest records which pool each
/// entry came from -- a corpus that reported a count without saying where the points came from
/// could not be audited for realism.
///
/// This is a port of the retired Python assembler, and it is op-general where that was not:
/// it works on @ref ProblemPoint, which is whatever the declaration declares, rather than on an
/// SDPA shape record. Nothing here names an operation or a parameter.
namespace hipdnn_corpus_gen
{

/// Order of precedence when two sources describe the same problem, and the order the manifest
/// reports.
///
/// A recorded model shape beats a packed kernel geometry beats a sample: the duplicate is the
/// same problem either way, so what is being chosen is which *provenance* the manifest records,
/// and "this is what llama runs" is worth more to a later audit than "the sampler also drew
/// it".
inline const std::vector<std::string>& corpusSources()
{
    static const std::vector<std::string> sources{"model", "kernel", "sweep"};
    return sources;
}

/// Default share of the corpus each source is allocated.
///
/// Kernel geometries take the largest share because they are the only problems an engine's own
/// pack guarantees it was compiled for. A source that cannot fill its share hands the remainder
/// back (see @ref allocate), so these are preferences, not quotas.
inline const std::map<std::string, double>& defaultShares()
{
    static const std::map<std::string, double> shares{
        {"model", 0.15}, {"kernel", 0.60}, {"sweep", 0.25}};
    return shares;
}

/// One candidate problem, with the provenance that makes the corpus auditable.
struct PoolEntry
{
    ProblemPoint point;

    /// Which pool it came from: one of @ref corpusSources.
    std::string source;

    /// Where inside that pool -- a pack file and its kernel count, a model name, a draw index.
    /// Free text, because the sources are not alike and flattening them into one vocabulary
    /// would lose the only detail an audit needs.
    std::string origin;

    /// The stratification label (see RegimeLabel.hpp). Carried rather than recomputed so that
    /// ordering and reporting cannot disagree about which regime an entry is in.
    std::string regime;

    /// What a cut is spread over: the entry's categorical combination (dtype, layout, mode...)
    /// together with its regime. Empty falls back to the regime alone. Without the combination,
    /// a declaration with no regime label spreads over nothing, a cut keeps whichever
    /// combinations were searched first, and the last dtype disappears from the corpus.
    std::string stratum;
};

/// The pools, keyed by source name.
using SourcePools = std::map<std::string, std::vector<PoolEntry>>;

/// @brief How many problems each source contributes, given what each source has.
///
/// Shares first, then whatever a short pool could not use is redistributed one at a time over
/// the pools that still have room. Round-robin redistribution rather than a priority order,
/// because handing an entire shortfall to one source is how a corpus that asked for a mix gets
/// 90% of one population -- exactly the failure a per-regime coverage table exists to expose.
///
/// A share of exactly 0 EXCLUDES its source rather than deferring it. Redistribution used to
/// refill it: `--kernel-share 1.0 --model-share 0 --sweep-share 0` against a pack with 974
/// eligible geometries returned 974 kernel problems and then 4026 from the two sources the
/// caller had just switched off. For an engine whose kernels are compiled per exact shape that
/// is not a mixed corpus, it is 4026 declines -- the caller asked for the pack's own geometries
/// and got mostly the opposite.
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
        const auto wanted = static_cast<int64_t>(static_cast<double>(count)
                                                 * shareOf(entry.first) / total);
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
               && shareOf(source) > 0.0)
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
/// Deduplication is on the whole point -- every declared parameter, categorical and numeric --
/// because every one of them changes which kernel is fastest. Two entries differing only in
/// provenance are one problem measured twice: the same graph benchmarked twice under two names,
/// which inflates a corpus and biases whichever regime it lands in.
///
/// @p dropped receives the per-source count of entries removed, so the manifest can say what
/// the overlap between sources actually was rather than leaving a short corpus unexplained.
inline SourcePools deduplicate(const SourcePools& pools, std::map<std::string, int64_t>& dropped)
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

/// One pool reordered so that any prefix of it holds the pool's regime mix.
///
/// A pool that arrives grouped -- model shapes read file by file, so each file's rows are
/// contiguous -- is one whose front is not a sample of it. Truncating the front is then
/// truncating the alphabet: against a published shape directory that dropped 95 of 200 shapes
/// and with them every regime that happened to be written down in a late-named file.
///
/// Proportional rather than round-robin: the pool's own mix is the fact worth preserving -- it
/// is what real models run -- so a prefix should look like the pool and not like one row of
/// every regime the pool happens to mention. Each member is placed at its fractional position
/// within its regime and the positions are merged, which puts `n * share` of every regime in
/// any prefix of length `n` and keeps each regime's internal order.
inline std::vector<PoolEntry> spread(const std::vector<PoolEntry>& pool)
{
    // First-appearance order, not sorted order: `rank` breaks ties between regimes of equal
    // size deterministically, and it must not depend on how the labels happen to collate.
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

/// @brief The corpus: each pool's allocation, taken from the front of the pool.
///
/// The front, not a sample: every pool is ordered so that a prefix stays spread across the
/// space it covers -- the kernel pool by its own stratification, the sweep by its declared
/// mixture, and the model pool by @ref detail::spread here. Re-sampling at this point would
/// undo all three.
///
/// @p allocation receives what each source was asked for, which is not the same as what the
/// manifest's mix reports: a source can be allocated more than it delivers only if a pool
/// shrank between the two, and recording both is how that would be noticed.
inline std::vector<PoolEntry> select(const SourcePools& pools,
                                     int64_t count,
                                     const std::map<std::string, double>& shares,
                                     std::map<std::string, int64_t>& allocation)
{
    SourcePools ordered;
    std::map<std::string, int64_t> capacity;
    for(const auto& source : corpusSources())
    {
        const auto found = pools.find(source);
        // Every pool is spread before it is cut, not only the model pool: a pack or a search
        // arrives in its own order, and taking a prefix of that keeps whatever came first.
        ordered[source] = detail::spread(
            found == pools.end() ? std::vector<PoolEntry>{} : found->second);
        capacity[source] = static_cast<int64_t>(ordered[source].size());
    }

    allocation = allocate(count, capacity, shares);

    std::vector<PoolEntry> selected;
    for(const auto& source : corpusSources())
    {
        const auto& pool = ordered[source];
        const auto take = static_cast<size_t>(std::max<int64_t>(0, allocation[source]));
        selected.insert(selected.end(), pool.begin(), pool.begin() + static_cast<ptrdiff_t>(
                                                          std::min(take, pool.size())));
    }
    return selected;
}

} // namespace hipdnn_corpus_gen

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <optional>

/// @file FeasibleRegionProbe.hpp
/// @brief Locates the bounds of an engine's problem region by asking it (RFC 0019.13 §5.3).
///
/// The two halves of the feature space are not symmetric. A kernel knob is declared, with a
/// minimum, a maximum, a step and sometimes an explicit value list, which is why the variant
/// space can be enumerated. An operation parameter is declared nowhere: an engine advertises
/// its id, its knobs, its behaviour notes and its name, and says nothing about the tensor
/// dimensions it accepts. The only thing that knows is `is_applicable`, and it answers about
/// one shape at a time.
///
/// So bounds are measured rather than modelled. That is what makes the operations with no
/// closed-form footprint decomposition tractable -- a twelve-parameter convolution needs no
/// arithmetic if candidates can simply be tested.
///
/// The oracle is injected rather than being the backend predicate directly, because the
/// search is worth testing against regions whose shape is known, and because the caller is
/// what knows how to build a graph for a candidate.
namespace hipdnn_corpus_gen
{

/// Answers whether one value of the parameter under study is accepted, every other parameter
/// held fixed. In production this builds a graph and calls `is_applicable`; in tests it is a
/// region whose answer is already known.
using MembershipOracle = std::function<bool(int64_t)>;

/// What a search found, and how much to trust it.
struct ProbedBound
{
    /// Furthest accepted value found in the direction searched.
    int64_t value = 0;

    /// Oracle calls spent. Reported because a probe is cheap but not free -- it costs a graph
    /// build -- and because a search that spent far more than a bisection should is evidence
    /// the region is not shaped the way the search assumes.
    int64_t probes = 0;

    /// True when a value beyond @ref value was found to be accepted, which proves the region
    /// is not the interval bisection assumes.
    ///
    /// Read the negative carefully: false means the search found no contradiction, not that
    /// the region is an interval. Two blind spots make that gap real.
    ///
    /// The first is stride alignment. The search doubles, so its samples are @p knownGood
    /// times a power of two; a region admitting only multiples of 512, probed from 512,
    /// accepts every value it is asked about and reports nothing amiss. An alignment rule
    /// that happens to divide the search's own stride is invisible to it.
    ///
    /// The second is that only values *beyond* the bound are checked. Holes below it are
    /// never looked for, and a caller sampling the interior will meet them as rejections.
    ///
    /// So this flag is evidence of holes, never evidence of their absence. Interior sampling
    /// stays sound either way -- every candidate is still put to the predicate before it
    /// reaches a corpus -- but @ref value may not be described as the extent of the region.
    bool acceptedBeyondBound = false;
};

namespace detail
{

/// Bisects between an accepted @p accepted and a rejected @p rejected, in either direction,
/// until they are adjacent. Returns the last accepted value.
inline int64_t bisect(const MembershipOracle& oracle,
                      int64_t accepted,
                      int64_t rejected,
                      int64_t& probes)
{
    // Difference rather than (a+b)/2: the sum overflows for values near the type's limit, and
    // a dimension bounded only by "what fits in memory" gets close enough to matter.
    while(accepted < rejected ? rejected - accepted > 1 : accepted - rejected > 1)
    {
        const int64_t midpoint = accepted + ((rejected - accepted) / 2);
        ++probes;
        if(oracle(midpoint))
        {
            accepted = midpoint;
        }
        else
        {
            rejected = midpoint;
        }
    }
    return accepted;
}

/// Looks for an accepted value strictly beyond @p bound, in the direction of @p ceiling.
///
/// This is the falsification step, and it is the only thing standing between a probed bound
/// and a confidently wrong one. Deterministic by construction -- a fixed number of evenly
/// spaced samples, no RNG -- because §5.8 requires a corpus to be reproducible from its seed
/// and a bound that moved between runs would make the corpus that used it move with it.
inline bool foundAcceptedBeyond(const MembershipOracle& oracle,
                                int64_t bound,
                                int64_t ceiling,
                                int64_t samples,
                                int64_t& probes)
{
    const int64_t span = ceiling - bound;
    if(span <= 1 || samples <= 0)
    {
        return false;
    }

    for(int64_t sample = 1; sample <= samples; ++sample)
    {
        const int64_t candidate = bound + ((span * sample) / (samples + 1));
        if(candidate <= bound)
        {
            continue;
        }
        ++probes;
        if(oracle(candidate))
        {
            return true;
        }
    }
    return false;
}

} // namespace detail

/// @brief Highest accepted value at or below @p ceiling.
///
/// @param oracle       Membership test for one parameter.
/// @param knownGood    A value the caller already knows is accepted -- in practice one that
///                     came out of a successful footprint-first draw. The search grows from
///                     here rather than from a guess.
/// @param ceiling      Highest value worth asking about, e.g. the memory ceiling. The search
///                     never proposes beyond it.
/// @param verification Samples spent looking for acceptance beyond the bound found. Zero
///                     skips the check, which is faster and gives up the only evidence that
///                     the answer means what it appears to.
///
/// @returns nullopt when @p knownGood is itself rejected, which is a caller error rather than
///          an empty region: the search has no foothold and guessing one would be inventing
///          the answer.
inline std::optional<ProbedBound> probeUpperBound(const MembershipOracle& oracle,
                                                  int64_t knownGood,
                                                  int64_t ceiling,
                                                  int64_t verification = 4)
{
    ProbedBound result;
    ++result.probes;
    if(knownGood > ceiling || !oracle(knownGood))
    {
        return std::nullopt;
    }

    // Double until rejection or the ceiling, so an unbounded parameter costs a logarithmic
    // number of probes instead of a walk. A dimension accepted all the way to the ceiling is
    // reported at the ceiling: the region may well continue, but nothing above it can be
    // benchmarked, so the distinction has no consequence for a corpus.
    int64_t accepted = knownGood;
    int64_t rejected = 0;
    bool foundRejection = false;
    while(accepted < ceiling)
    {
        const int64_t next = (accepted > ceiling / 2) ? ceiling : accepted * 2;
        ++result.probes;
        if(oracle(next))
        {
            accepted = next;
            if(accepted == ceiling)
            {
                break;
            }
        }
        else
        {
            rejected = next;
            foundRejection = true;
            break;
        }
    }

    result.value = foundRejection ? detail::bisect(oracle, accepted, rejected, result.probes)
                                  : accepted;
    result.acceptedBeyondBound
        = detail::foundAcceptedBeyond(oracle, result.value, ceiling, verification, result.probes);
    return result;
}

/// @brief Lowest accepted value at or above @p floorValue. The mirror of probeUpperBound.
///
/// Halves toward @p floorValue rather than doubling. The floor is one element rather than
/// something rounder because degenerate shapes are real workloads -- `M = 1` is single-token
/// decode, a regime in its own right (RFC 0019.13 §5.3).
inline std::optional<ProbedBound> probeLowerBound(const MembershipOracle& oracle,
                                                  int64_t knownGood,
                                                  int64_t floorValue = 1,
                                                  int64_t verification = 4)
{
    ProbedBound result;
    ++result.probes;
    if(knownGood < floorValue || !oracle(knownGood))
    {
        return std::nullopt;
    }

    int64_t accepted = knownGood;
    int64_t rejected = 0;
    bool foundRejection = false;
    while(accepted > floorValue)
    {
        const int64_t next = (accepted / 2 < floorValue) ? floorValue : accepted / 2;
        ++result.probes;
        if(oracle(next))
        {
            accepted = next;
            if(accepted == floorValue)
            {
                break;
            }
        }
        else
        {
            rejected = next;
            foundRejection = true;
            break;
        }
    }

    result.value = foundRejection ? detail::bisect(oracle, accepted, rejected, result.probes)
                                  : accepted;
    result.acceptedBeyondBound
        = detail::foundAcceptedBeyond(oracle, result.value, floorValue, verification, result.probes);
    return result;
}

} // namespace hipdnn_corpus_gen

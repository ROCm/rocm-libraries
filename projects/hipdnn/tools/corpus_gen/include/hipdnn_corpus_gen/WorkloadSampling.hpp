// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

/// @file WorkloadSampling.hpp
/// @brief Drawing problems that look like workloads, from declared anchors (§5.2, §12.2).
///
/// The exploration in ProblemSpace.hpp answers "what does this engine serve?", and answers it
/// well: it finds the feasible region and spreads over it. That is the wrong distribution for
/// training, and measurably so -- a uniform spread over conv_fwd's feasible region put 1.0% of
/// its draws on a square image and 1.3% on channel counts aligned to eight, so 0.04% of ten
/// thousand shapes resembled a layer anyone runs. A model trained on that learns the region's
/// interior, which is not where the questions come from.
///
/// The fix is not to narrow the region -- an engine still has to answer for problems nobody
/// listed, and §5.4 keeps an exploration floor for exactly that reason. It is to draw *most*
/// of the corpus near recorded workloads, and to make "near" mean something structural:
///
///  - **Archetypes** are correlated tuples from real networks. A draw takes one value per
///    parameter from one archetype, so the joint fact survives; sampling each parameter from
///    its own marginal is what loses it.
///  - **Neighbourhoods** say how each parameter may move away from an anchor and remain
///    plausible: channels between aligned multiples, spatial extents by halving and doubling,
///    filters among the small odd sizes, widths following heights.
///
/// Nothing here is operation-specific. The declaration carries the workload knowledge; this is
/// one sampler for every operation, the same way there is one exploration.
namespace hipdnn_corpus_gen
{
namespace detail
{

/// A parameter's floor: its declared range's lower bound, or 1.
///
/// One rather than zero for an undeclared floor, because a zero extent is not a problem for any
/// operation declared so far and a zero reaching a graph builder makes a tensor with no
/// elements. A *declared* zero floor is honoured -- that is padding, where zero is the ordinary
/// case rather than a degenerate one.
inline int64_t parameterFloor(const Parameter& parameter)
{
    if(parameter.range.has_value())
    {
        return std::max<int64_t>(parameter.range->first, 0);
    }
    return 1;
}

/// Clamps @p value into whatever @p parameter declares, leaving an undeclared side alone.
inline int64_t clampToRange(const Parameter& parameter, int64_t value)
{
    int64_t clamped = std::max(value, parameterFloor(parameter));
    if(parameter.range.has_value())
    {
        // An undeclared ceiling is stored as int64 max, so this is a no-op there rather than a
        // special case -- see the range parsing in OperationMetadata.hpp.
        clamped = std::min(clamped, parameter.range->second);
    }
    return clamped;
}

/// Reads @p point's value for @p name as an integer, or nullopt if it is not numeric.
inline std::optional<int64_t> integerAt(const ProblemPoint& point, const std::string& name)
{
    const auto found = point.find(name);
    if(found == point.end())
    {
        return std::nullopt;
    }
    if(const auto* held = std::get_if<int64_t>(&found->second))
    {
        return *held;
    }
    if(const auto* held = std::get_if<double>(&found->second))
    {
        return static_cast<int64_t>(std::llround(*held));
    }
    return std::nullopt;
}

/// Converts one declared archetype value to a parameter value, following `$q.<other>` against
/// what has already been drawn.
///
/// Returns nullopt when a reference names something not yet drawn. Parsing rejects that case,
/// so reaching it means the declaration order changed underneath -- worth failing the draw
/// rather than substituting a floor and calling the result an anchored shape.
inline std::optional<ParameterValue> archetypeValue(const nlohmann::json& declared,
                                                    const ProblemPoint& drawnSoFar)
{
    if(declared.is_string())
    {
        const auto referenced = queryReference(declared.get<std::string>());
        if(!referenced.empty())
        {
            const auto found = drawnSoFar.find(referenced);
            if(found == drawnSoFar.end())
            {
                return std::nullopt;
            }
            return found->second;
        }
        return ParameterValue{declared.get<std::string>()};
    }
    if(declared.is_boolean())
    {
        return ParameterValue{declared.get<bool>()};
    }
    if(declared.is_number_integer())
    {
        return ParameterValue{declared.get<int64_t>()};
    }
    if(declared.is_number())
    {
        return ParameterValue{declared.get<double>()};
    }
    return std::nullopt;
}

/// @brief Draws one problem point from @p archetype, honouring an already-fixed @p categorical.
///
/// Returns nullopt when the archetype contradicts the categorical assignment -- a causal-only
/// attention archetype under `is_causal = false`, say. That is not an error: it says this
/// combination has no recorded workload, and the caller should fall back to exploration rather
/// than manufacture an anchor nobody claimed. Silently overriding the categorical instead would
/// give one combination another's problems and label them with the wrong dtype or mode.
///
/// A numeric parameter the archetype does not set takes its floor. Deliberately unambitious:
/// the archetype is the claim about realism, and inventing a value for something it left out
/// would be this code making that claim instead.
inline std::optional<ProblemPoint> drawFromArchetype(const OperationMetadata& metadata,
                                                     const Archetype& archetype,
                                                     const ProblemPoint& categorical,
                                                     std::mt19937_64& rng)
{
    ProblemPoint point;

    // Declaration order, so `$q.<other>` sees its referent already drawn.
    for(const auto& parameter : metadata.parameters)
    {
        const auto fixed = categorical.find(parameter.name);
        const auto declared = archetype.values.find(parameter.name);

        if(declared == archetype.values.end())
        {
            if(fixed != categorical.end())
            {
                point[parameter.name] = fixed->second;
            }
            else if(parameter.type == ParameterType::INT64
                    || parameter.type == ParameterType::FLOAT64)
            {
                point[parameter.name] = parameterFloor(parameter);
            }
            else
            {
                return std::nullopt; // categorical, unfixed and unset: no point to draw
            }
            continue;
        }

        std::uniform_int_distribution<size_t> pick(0, declared->second.size() - 1);
        auto value = archetypeValue(declared->second.at(pick(rng)), point);
        if(!value.has_value())
        {
            return std::nullopt;
        }

        if(fixed != categorical.end())
        {
            // The combination is already committed to a value; an archetype that wants a
            // different one simply does not describe this combination.
            if(!(*value == fixed->second))
            {
                return std::nullopt;
            }
            *value = fixed->second;
        }
        point[parameter.name] = *value;
    }
    return point;
}

/// One numeric parameter moved within its declared neighbourhood.
inline int64_t perturbOne(const Parameter& parameter,
                          const Neighbourhood& hood,
                          int64_t base,
                          const ProblemPoint& drawnSoFar,
                          std::mt19937_64& rng)
{
    const auto choose = [&rng](size_t count) {
        std::uniform_int_distribution<size_t> pick(0, count - 1);
        return pick(rng);
    };

    switch(hood.kind)
    {
    case Neighbourhood::Kind::SCALE:
    {
        const auto factor = hood.factors.at(choose(hood.factors.size()));
        return clampToRange(parameter,
                            static_cast<int64_t>(std::llround(static_cast<double>(base) * factor)));
    }
    case Neighbourhood::Kind::MULTIPLE:
    {
        // A value below the alignment is left alone. It is not a misaligned version of
        // something -- it is a distinguished small value the alignment cannot express, and the
        // motivating case is C=3. Rounding 3 up to 8 loses the three-channel input of every
        // vision network's first layer, and it does so in the 60% of the corpus that comes
        // from perturbation, so a ResNet stem survives only in the pure-archetype share.
        // Measured before this: C=3 was 4.3% of a MIOpen conv corpus and 1.9% of a hipkernel
        // one, against roughly a fifth of the archetypes declaring it.
        if(base < hood.of)
        {
            return clampToRange(parameter, base);
        }

        // At or above the alignment, align first: an anchor is usually already a multiple, but
        // one that is not would otherwise carry its misalignment through every perturbation,
        // and alignment is the property this kind exists to preserve.
        const auto aligned
            = std::max(hood.of, static_cast<int64_t>(std::llround(static_cast<double>(base)
                                                                  / static_cast<double>(hood.of)))
                                    * hood.of);
        const auto step = hood.steps.at(choose(hood.steps.size()));
        return clampToRange(parameter, std::max(hood.of, aligned + (step * hood.of)));
    }
    case Neighbourhood::Kind::VALUES:
        return clampToRange(parameter, hood.values.at(choose(hood.values.size())));
    case Neighbourhood::Kind::MIRROR:
    {
        const auto followed = integerAt(drawnSoFar, hood.mirrors);
        if(!followed.has_value())
        {
            return clampToRange(parameter, base);
        }
        const auto ratio = hood.ratios.empty() ? 1.0 : hood.ratios.at(choose(hood.ratios.size()));
        return clampToRange(
            parameter,
            static_cast<int64_t>(std::llround(static_cast<double>(*followed) * ratio)));
    }
    default:
        return clampToRange(parameter, base);
    }
}

/// @brief Moves @p anchor within the declared neighbourhood, leaving categoricals alone.
///
/// Parameters with no neighbourhood keep the anchor's value. That is the point: a declaration
/// says which parameters may drift, and padding or dilation that moved because it was numeric
/// would turn a recorded layer into a shape whose geometry nobody chose.
inline ProblemPoint perturbWithinNeighbourhood(const OperationMetadata& metadata,
                                               const ProblemPoint& anchor,
                                               std::mt19937_64& rng)
{
    ProblemPoint point;
    for(const auto& parameter : metadata.parameters) // declaration order, for mirrors
    {
        const auto found = anchor.find(parameter.name);
        if(found == anchor.end())
        {
            continue;
        }
        point[parameter.name] = found->second;

        if(parameter.type != ParameterType::INT64 && parameter.type != ParameterType::FLOAT64)
        {
            continue;
        }
        const auto hood = metadata.neighbourhood.find(parameter.name);
        const auto base = integerAt(anchor, parameter.name);
        if(hood == metadata.neighbourhood.end() || !base.has_value())
        {
            continue;
        }
        point[parameter.name] = perturbOne(parameter, hood->second, *base, point, rng);
    }
    return point;
}

} // namespace detail
} // namespace hipdnn_corpus_gen

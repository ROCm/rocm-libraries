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
/// Archetypes are correlated tuples from real networks; a draw takes one whole archetype so
/// joint facts survive. Neighbourhoods say how each parameter may move away from an anchor.
namespace hipdnn_corpus_gen
{
namespace detail
{

/// A parameter's floor: its declared range's lower bound, or 1 when undeclared (a zero extent
/// makes an empty tensor). A declared zero floor, e.g. padding, is honoured.
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
        // An undeclared ceiling is stored as int64 max, so this is a no-op there.
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
/// Returns nullopt when a reference names something not yet drawn; draw order should prevent
/// that, so fail the draw rather than substitute a floor.
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
/// Returns nullopt when the archetype contradicts @p categorical (no recorded workload for that
/// combination); the caller should fall back to exploration. A numeric parameter the archetype
/// does not set takes its floor rather than an invented value.
inline std::optional<ProblemPoint> drawFromArchetype(const OperationMetadata& metadata,
                                                     const Archetype& archetype,
                                                     const ProblemPoint& categorical,
                                                     std::mt19937_64& rng)
{
    ProblemPoint point;

    // Referents first (Archetype::drawOrder), so `$q.<other>` sees its referent already drawn.
    for(const auto index : archetype.drawOrder)
    {
        const auto& parameter = metadata.parameters.at(index);
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
        // A value below the alignment is left alone: it is a distinguished small value (e.g.
        // C=3 for an image input), not a misaligned one, and rounding it up would lose it.
        if(base < hood.of)
        {
            return clampToRange(parameter, base);
        }

        // At or above the alignment, align first so a misaligned anchor does not carry its
        // misalignment through every perturbation.
        const auto aligned = std::max(hood.of,
                                      static_cast<int64_t>(std::llround(
                                          static_cast<double>(base) / static_cast<double>(hood.of)))
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
            parameter, static_cast<int64_t>(std::llround(static_cast<double>(*followed) * ratio)));
    }
    default:
        return clampToRange(parameter, base);
    }
}

/// @brief Moves @p anchor within the declared neighbourhood, leaving categoricals alone.
///
/// Parameters with no declared neighbourhood keep the anchor's value, so e.g. padding does not
/// drift just because it is numeric.
inline ProblemPoint perturbWithinNeighbourhood(const OperationMetadata& metadata,
                                               const ProblemPoint& anchor,
                                               std::mt19937_64& rng)
{
    ProblemPoint point;
    for(const auto index : metadata.perturbationOrder) // followed parameters first, for mirrors
    {
        const auto& parameter = metadata.parameters.at(index);
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

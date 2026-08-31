// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

/// @file KnobEnumeration.hpp
/// @brief The configuration half of the joint space, read from the engine (RFC 0019.13 §5.6).
///
/// A problem is searched for, because nothing declares what an engine will accept. A
/// configuration is not: a knob declares either an explicit value list or a min/max/step
/// range, so the set of configurations can simply be read off. That asymmetry is why §5.6
/// treats the two halves differently, and it is why nothing here resembles the shape search.
///
/// The one judgement this file makes is where to stop. A knob range can be large, knobs
/// multiply, and the product is what gets benchmarked -- so an unbounded enumeration is a
/// corpus that never finishes. Truncation is reported rather than silent: a caller who thinks
/// it swept a knob it only sampled will read the resulting model's blind spot as noise.
namespace hipdnn_bench
{

/// Values of one knob, and whether that is all of them.
struct KnobValues
{
    std::string knobId;
    std::vector<int64_t> values;

    /// True when the knob offers more than @c values holds. The count alone cannot say so --
    /// a knob with exactly the cap's worth of values looks identical to one that was cut.
    bool truncated = false;
};

/// Largest number of values to take from one knob's range by default.
///
/// Sized so that a few knobs of this width remain a runnable sweep rather than a fleet-week.
/// Explicit value lists are never truncated: an engine that names its values has said the set
/// is small and meaningful, and dropping some of those is dropping a kernel variant.
constexpr size_t DEFAULT_VALUES_PER_KNOB = 32;

/// @brief The values @p knob offers, as far as @p limit allows.
///
/// Returns empty for a knob this tool cannot enumerate -- float and string knobs, and knobs
/// with no constraint at all. Empty means "not swept", and the caller keeps the engine's
/// default for it, which is honest: an unenumerable knob is not a knob with one value.
inline KnobValues enumerateKnob(const hipdnn_frontend::Knob& knob,
                                size_t limit = DEFAULT_VALUES_PER_KNOB)
{
    KnobValues result;
    result.knobId = knob.knobId();

    const auto* constraint = knob.constraint();
    if(constraint == nullptr
       || constraint->kind() != hipdnn_frontend::ConstraintKind::INT)
    {
        return result;
    }

    const auto* ints = dynamic_cast<const hipdnn_frontend::IntConstraint*>(constraint);
    if(ints == nullptr)
    {
        return result;
    }

    const auto& explicitValues = ints->getValidValues();
    if(!explicitValues.empty())
    {
        result.values.assign(explicitValues.begin(), explicitValues.end());
        // Sorted so a corpus generated twice lists its configurations in one order; the
        // constraint holds them in an unordered_set.
        std::sort(result.values.begin(), result.values.end());
        return result;
    }

    const auto step = ints->getStep() > 0 ? ints->getStep() : 1;
    for(int64_t value = ints->getMinValue(); value <= ints->getMaxValue(); value += step)
    {
        if(result.values.size() >= limit)
        {
            result.truncated = true;
            break;
        }
        result.values.push_back(value);
    }
    return result;
}

/// One point in the configuration space.
using Configuration = std::vector<hipdnn_frontend::KnobSetting>;

/// Every combination of the enumerable knobs in @p knobs, and what was left out.
struct ConfigurationSet
{
    std::vector<Configuration> configurations;

    /// Knobs that could not be enumerated, or were cut short. Reported so a caller can say
    /// which part of the configuration space its corpus does not cover, rather than leaving
    /// the gap to be inferred from a model that predicts badly in one region.
    std::vector<std::string> notFullyCovered;
};

/// @brief The Cartesian product of @p knobs' values, bounded by @p maxConfigurations.
///
/// Always contains at least one configuration -- the empty one, meaning engine defaults --
/// because an engine with no enumerable knobs still has exactly one way to run.
inline ConfigurationSet enumerateConfigurations(const std::vector<hipdnn_frontend::Knob>& knobs,
                                                size_t maxConfigurations = 1024,
                                                size_t valuesPerKnob = DEFAULT_VALUES_PER_KNOB)
{
    ConfigurationSet result;
    result.configurations.emplace_back();

    for(const auto& knob : knobs)
    {
        const auto values = enumerateKnob(knob, valuesPerKnob);
        if(values.values.empty())
        {
            result.notFullyCovered.push_back(values.knobId + " (not enumerable)");
            continue;
        }
        if(values.truncated)
        {
            result.notFullyCovered.push_back(values.knobId + " (truncated)");
        }

        if(result.configurations.size() * values.values.size() > maxConfigurations)
        {
            result.notFullyCovered.push_back(values.knobId + " (would exceed the product cap)");
            continue;
        }

        std::vector<Configuration> expanded;
        expanded.reserve(result.configurations.size() * values.values.size());
        for(const auto& base : result.configurations)
        {
            for(const auto value : values.values)
            {
                auto combination = base;
                combination.emplace_back(values.knobId, value);
                expanded.push_back(std::move(combination));
            }
        }
        result.configurations = std::move(expanded);
    }
    return result;
}

} // namespace hipdnn_bench

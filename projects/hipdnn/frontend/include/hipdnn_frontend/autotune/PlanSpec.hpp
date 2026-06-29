// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file PlanSpec.hpp
 * @brief Engine metadata and knob sweep types for autotuning
 *
 * Defines the user-facing types for engine discovery and knob sweep
 * specification.
 *
 * EngineConfigInfo, EngineVariant, KnobSweepAxis, and EngineSweepSpec are
 * user-facing types for the engine discovery and plan spec collection API.
 */

#pragma once

#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hipdnn_frontend
{

// Read-only snapshot of an engine's configuration. Provided for inspection and
// filtering only - do not modify fields directly. Use add_engine_*() to create
// plan specs from selected configs.

/**
 * @brief Rich engine metadata returned by get_engine_configs()
 *
 * Contains all discoverable information about an engine implementation
 * for a given operation graph. Used to inspect, filter, and select
 * engines before adding plan specs for autotuning.
 */
struct EngineConfigInfo
{
    int64_t engineId = -1; // Used by add_engine_configs() to create the plan spec
    std::string engineName; // Informational, for filtering and logging
    std::vector<Knob> knobs; // Informational, shows the engine's available knobs.
        // Ignored by add_engine_*() functions. Use add_engine_variants()
        // or add_engine() to set custom knobs on engines for autotune().
    bool supportsExhaustive = false; // Informational. For filtering exhaustive-capable engines
    int64_t estimatedWorkspaceSize
        = 0; // Informational, pre-compile workspace estimate, for filtering.
};

/**
 * @brief User-facing (engineId, knobSettings) pair for explicit variant autotuning
 *
 * Used with add_engine_variants() to add plan specs with explicit knob settings.
 * Each EngineVariant becomes one plan spec for autotuning.
 */
struct EngineVariant
{
    int64_t engineId = -1; ///< Engine to configure
    // std::map: deterministic key order; negligible performance impact.
    std::map<KnobType_t, KnobValueVariant> knobSettings; ///< Explicit knob values
};

/**
 * @brief One axis of a knob sweep for Cartesian product generation
 *
 * Represents a single knob with multiple candidate values. Used as input
 * to add_engine_sweep() within an EngineSweepSpec.
 */
struct KnobSweepAxis
{
    KnobType_t knobId; ///< Knob to sweep
    std::vector<KnobValueVariant> values; ///< Values to try for this knob
};

/**
 * @brief Cartesian product sweep specification for one engine
 *
 * Defines a set of knob axes to sweep (Cartesian product) plus fixed
 * settings applied to every combination. Used with add_engine_sweep().
 *
 * Example: 2 axes of 3 values each produces 9 plan specs, each with
 * the fixedSettings merged in.
 */
struct EngineSweepSpec
{
    int64_t engineId = -1; ///< Engine to sweep
    std::vector<KnobSweepAxis> axes; ///< Knobs to sweep (Cartesian product)
    // std::map: deterministic key order; negligible performance impact.
    std::map<KnobType_t, KnobValueVariant> fixedSettings; ///< Knobs held constant
};

} // namespace hipdnn_frontend

namespace hipdnn_frontend::autotune::detail
{

// Internal plan specification for autotuning deduplication.
//
// A PlanSpec captures the composite key (engineId, knobSettings) that
// uniquely identifies an autotuning candidate. Plan specs are stored
// on the Graph by add_engine_*() calls and compiled into execution
// plans by autotune().
//
// Deduplication uses operator==, which compares engineId and knob
// settings (sorted by knob ID for order-independent comparison).
// workspaceSize and supportsExhaustive are excluded from equality
// since two specs with the same engine and knobs always share them.
struct PlanSpec
{
    int64_t engineId = -1; // Engine ID for this candidate
    std::vector<KnobSetting> knobSettings; // Knob values for this candidate
    int64_t workspaceSize = 0; // Workspace bytes (from EngineConfigInfo at add time)
    bool supportsExhaustive
        = false; // Engine exposes the benchmarking knob (from EngineConfigInfo / knobs at add time)

    // Compare two PlanSpecs for equality (deduplication key).
    //
    // Two PlanSpecs are equal if they have the same engineId and
    // equivalent knob settings (order-independent). workspaceSize
    // is intentionally excluded from comparison.
    bool operator==(const PlanSpec& other) const
    {
        if(engineId != other.engineId)
        {
            return false;
        }

        if(knobSettings.size() != other.knobSettings.size())
        {
            return false;
        }

        // Sort copies by knob ID for order-independent comparison
        auto sortedThis = sortedKnobs(knobSettings);
        auto sortedOther = sortedKnobs(other.knobSettings);

        for(size_t i = 0; i < sortedThis.size(); ++i)
        {
            if(sortedThis[i].knobId() != sortedOther[i].knobId())
            {
                return false;
            }
            if(sortedThis[i].value() != sortedOther[i].value())
            {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const PlanSpec& other) const
    {
        return !(*this == other);
    }

private:
    // Sort knob settings by knob ID for order-independent comparison
    static std::vector<KnobSetting> sortedKnobs(const std::vector<KnobSetting>& knobs)
    {
        auto sorted = knobs;
        std::sort(sorted.begin(), sorted.end(), [](const KnobSetting& a, const KnobSetting& b) {
            return a.knobId() < b.knobId();
        });
        return sorted;
    }
};

} // namespace hipdnn_frontend::autotune::detail

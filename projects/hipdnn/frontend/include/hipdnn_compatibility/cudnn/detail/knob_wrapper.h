// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/knobs.h), used under the MIT license.

/**
 * @file knob_wrapper.h
 * @brief cuDNN-shaped knob projection helpers for the hipDNN cuDNN shim.
 *
 * cuDNN frontend exposes knobs as a fixed enum plus int64 min/max/stride
 * metadata. hipDNN native knobs are provider-defined string IDs with variant
 * values and richer constraints. The helpers here keep that conversion explicit
 * and lossy rather than aliasing incompatible types.
 */

#pragma once

#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_frontend/knob/Knob.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// hipDNN knob ids are namespaced (e.g. "global.workspace_size_limit") while
// cuDNN's KnobType_t is flat, so match on the final dot-separated segment: both
// "tile_size" and "provider.tile_size" project onto the same cuDNN knob.
inline std::string normalizedKnobId(std::string knobId)
{
    const auto lastDot = knobId.rfind('.');
    if(lastDot != std::string::npos)
    {
        knobId.erase(0, lastDot + 1);
    }

    for(auto& ch : knobId)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    // The one knob whose bare hipDNN name differs from its cuDNN counterpart.
    if(knobId == "workspace_size_limit")
    {
        return "workspace";
    }
    return knobId;
}

inline std::optional<KnobType_t> fromHipdnnKnobId(const std::string& knobId)
{
    const auto normalized = normalizedKnobId(knobId);
    if(normalized == "swizzle")
    {
        return KnobType_t::SWIZZLE;
    }
    if(normalized == "tile_size")
    {
        return KnobType_t::TILE_SIZE;
    }
    if(normalized == "edge")
    {
        return KnobType_t::EDGE;
    }
    if(normalized == "multiply")
    {
        return KnobType_t::MULTIPLY;
    }
    if(normalized == "split_k_buf")
    {
        return KnobType_t::SPLIT_K_BUF;
    }
    if(normalized == "tilek")
    {
        return KnobType_t::TILEK;
    }
    if(normalized == "stages")
    {
        return KnobType_t::STAGES;
    }
    if(normalized == "reduction_mode")
    {
        return KnobType_t::REDUCTION_MODE;
    }
    if(normalized == "split_k_slc")
    {
        return KnobType_t::SPLIT_K_SLC;
    }
    if(normalized == "idx_mode")
    {
        return KnobType_t::IDX_MODE;
    }
    if(normalized == "specfilt")
    {
        return KnobType_t::SPECFILT;
    }
    if(normalized == "kernel_cfg")
    {
        return KnobType_t::KERNEL_CFG;
    }
    if(normalized == "workspace")
    {
        return KnobType_t::WORKSPACE;
    }
    if(normalized == "tile_cga_m")
    {
        return KnobType_t::TILE_CGA_M;
    }
    if(normalized == "tile_cga_n")
    {
        return KnobType_t::TILE_CGA_N;
    }
    if(normalized == "block_size")
    {
        return KnobType_t::BLOCK_SIZE;
    }
    if(normalized == "occupancy")
    {
        return KnobType_t::OCCUPANCY;
    }
    if(normalized == "array_size_per_thread")
    {
        return KnobType_t::ARRAY_SIZE_PER_THREAD;
    }
    if(normalized == "split_cols")
    {
        return KnobType_t::SPLIT_COLS;
    }
    if(normalized == "tile_rows")
    {
        return KnobType_t::TILE_ROWS;
    }
    if(normalized == "tile_cols")
    {
        return KnobType_t::TILE_COLS;
    }
    if(normalized == "load_size")
    {
        return KnobType_t::LOAD_SIZE;
    }
    if(normalized == "cta_count")
    {
        return KnobType_t::CTA_COUNT;
    }
    if(normalized == "stream_k")
    {
        return KnobType_t::STREAM_K;
    }
    if(normalized == "split_p_slc")
    {
        return KnobType_t::SPLIT_P_SLC;
    }
    if(normalized == "tile_m")
    {
        return KnobType_t::TILE_M;
    }
    if(normalized == "tile_n")
    {
        return KnobType_t::TILE_N;
    }
    if(normalized == "warp_spec_cfg")
    {
        return KnobType_t::WARP_SPEC_CFG;
    }
    return std::nullopt;
}

// Diagnostic-only cuDNN spelling of a knob type. Deliberately not a hipDNN knob
// id: the id a knob type maps onto is resolved per engine, never guessed.
inline const char* knobTypeName(KnobType_t knobType)
{
    switch(knobType)
    {
    case KnobType_t::SWIZZLE:
        return "SWIZZLE";
    case KnobType_t::TILE_SIZE:
        return "TILE_SIZE";
    case KnobType_t::EDGE:
        return "EDGE";
    case KnobType_t::MULTIPLY:
        return "MULTIPLY";
    case KnobType_t::SPLIT_K_BUF:
        return "SPLIT_K_BUF";
    case KnobType_t::TILEK:
        return "TILEK";
    case KnobType_t::STAGES:
        return "STAGES";
    case KnobType_t::REDUCTION_MODE:
        return "REDUCTION_MODE";
    case KnobType_t::SPLIT_K_SLC:
        return "SPLIT_K_SLC";
    case KnobType_t::IDX_MODE:
        return "IDX_MODE";
    case KnobType_t::SPECFILT:
        return "SPECFILT";
    case KnobType_t::KERNEL_CFG:
        return "KERNEL_CFG";
    case KnobType_t::WORKSPACE:
        return "WORKSPACE";
    case KnobType_t::TILE_CGA_M:
        return "TILE_CGA_M";
    case KnobType_t::TILE_CGA_N:
        return "TILE_CGA_N";
    case KnobType_t::BLOCK_SIZE:
        return "BLOCK_SIZE";
    case KnobType_t::OCCUPANCY:
        return "OCCUPANCY";
    case KnobType_t::ARRAY_SIZE_PER_THREAD:
        return "ARRAY_SIZE_PER_THREAD";
    case KnobType_t::SPLIT_COLS:
        return "SPLIT_COLS";
    case KnobType_t::TILE_ROWS:
        return "TILE_ROWS";
    case KnobType_t::TILE_COLS:
        return "TILE_COLS";
    case KnobType_t::LOAD_SIZE:
        return "LOAD_SIZE";
    case KnobType_t::CTA_COUNT:
        return "CTA_COUNT";
    case KnobType_t::STREAM_K:
        return "STREAM_K";
    case KnobType_t::SPLIT_P_SLC:
        return "SPLIT_P_SLC";
    case KnobType_t::TILE_M:
        return "TILE_M";
    case KnobType_t::TILE_N:
        return "TILE_N";
    case KnobType_t::WARP_SPEC_CFG:
        return "WARP_SPEC_CFG";
    case KnobType_t::NOT_SET:
    default:
        return "NOT_SET";
    }
}

inline std::optional<Knob> projectNativeKnob(const hipdnn_frontend::Knob& nativeKnob)
{
    auto knobType = fromHipdnnKnobId(nativeKnob.knobId());
    if(!knobType.has_value())
    {
        HIPDNN_FE_LOG_WARN("[cudnn_frontend] Omitting hipDNN knob '"
                           << nativeKnob.knobId() << "'; it has no cuDNN KnobType_t mapping.");
        return std::nullopt;
    }

    if(nativeKnob.valueType() != hipdnn_frontend::KnobValueType::INT64)
    {
        HIPDNN_FE_LOG_WARN("[cudnn_frontend] Omitting hipDNN knob '"
                           << nativeKnob.knobId() << "'; cuDNN knobs only carry int64 values.");
        return std::nullopt;
    }

    const auto* rawConstraint = nativeKnob.constraint();
    if(rawConstraint == nullptr || rawConstraint->kind() != hipdnn_frontend::ConstraintKind::INT)
    {
        HIPDNN_FE_LOG_WARN("[cudnn_frontend] Omitting hipDNN knob '"
                           << nativeKnob.knobId()
                           << "'; its constraint cannot be represented as cuDNN min/max/stride.");
        return std::nullopt;
    }
    const auto* constraint = static_cast<const hipdnn_frontend::IntConstraint*>(rawConstraint);
    if(!constraint->getValidValues().empty())
    {
        HIPDNN_FE_LOG_WARN("[cudnn_frontend] Omitting hipDNN knob '"
                           << nativeKnob.knobId()
                           << "'; its constraint cannot be represented as cuDNN min/max/stride.");
        return std::nullopt;
    }

    return Knob{
        *knobType, constraint->getMaxValue(), constraint->getMinValue(), constraint->getStep()};
}

inline void projectNativeKnobs(const std::vector<hipdnn_frontend::Knob>& nativeKnobs,
                               std::vector<Knob>& cudnnKnobs)
{
    cudnnKnobs.clear();
    cudnnKnobs.reserve(nativeKnobs.size());
    for(const auto& nativeKnob : nativeKnobs)
    {
        auto projected = projectNativeKnob(nativeKnob);
        if(projected.has_value())
        {
            cudnnKnobs.push_back(*projected);
        }
    }
}

// Resolve cuDNN knob types against the knobs the engine actually exposes, so a
// choice lands on the provider's real id ("miopen.tile_size") rather than on a
// guessed bare name. Entries point into engineKnobs, which must outlive the map.
// Two engine knobs can normalize onto the same cuDNN type; that collision is
// stored as a null entry so it only fails if the caller sets that type.
inline std::unordered_map<KnobType_t, const hipdnn_frontend::Knob*>
    buildKnobTypeLookup(const std::vector<hipdnn_frontend::Knob>& engineKnobs)
{
    std::unordered_map<KnobType_t, const hipdnn_frontend::Knob*> lookup;
    lookup.reserve(engineKnobs.size());
    for(const auto& engineKnob : engineKnobs)
    {
        const auto knobType = fromHipdnnKnobId(engineKnob.knobId());
        if(!knobType.has_value())
        {
            continue;
        }
        auto [entry, inserted] = lookup.emplace(*knobType, &engineKnob);
        if(!inserted)
        {
            entry->second = nullptr;
        }
    }
    return lookup;
}

// cuDNN carries knob values as int64 only, so widen to the native knob's declared
// type where that is lossless and reject otherwise, rather than letting a
// mistyped value fail later inside constraint validation.
inline error_t appendNativeKnobSetting(const hipdnn_frontend::Knob& engineKnob,
                                       int64_t value,
                                       std::vector<hipdnn_frontend::KnobSetting>& nativeSettings)
{
    switch(engineKnob.valueType())
    {
    case hipdnn_frontend::KnobValueType::INT64:
        nativeSettings.emplace_back(engineKnob.knobId(), value);
        return {};
    case hipdnn_frontend::KnobValueType::FLOAT64:
    {
        // Past 2^53 the widening to double silently rounds.
        constexpr int64_t EXACT_DOUBLE_LIMIT = int64_t{1} << 53;
        if(value > EXACT_DOUBLE_LIMIT || value < -EXACT_DOUBLE_LIMIT)
        {
            return {error_code_t::INVALID_VALUE,
                    "cuDNN knob value " + std::to_string(value) + " does not fit hipDNN knob '"
                        + engineKnob.knobId() + "' without rounding"};
        }
        nativeSettings.emplace_back(engineKnob.knobId(), static_cast<double>(value));
        return {};
    }
    case hipdnn_frontend::KnobValueType::STRING:
    case hipdnn_frontend::KnobValueType::NOT_SET:
    default:
        return {error_code_t::INVALID_VALUE,
                "hipDNN knob '" + engineKnob.knobId()
                    + "' does not take an integer value and cannot be set through the cuDNN "
                      "knob API"};
    }
}

inline error_t makeNativeKnobSettings(const std::unordered_map<KnobType_t, int64_t>& cudnnChoices,
                                      const std::vector<hipdnn_frontend::Knob>& engineKnobs,
                                      std::vector<hipdnn_frontend::KnobSetting>& nativeSettings)
{
    nativeSettings.clear();
    nativeSettings.reserve(cudnnChoices.size());

    const auto lookup = buildKnobTypeLookup(engineKnobs);
    for(const auto& [knobType, value] : cudnnChoices)
    {
        const auto entry = lookup.find(knobType);
        if(entry == lookup.end())
        {
            return {error_code_t::INVALID_VALUE,
                    std::string("cuDNN knob ") + knobTypeName(knobType)
                        + " is not exposed by this engine"};
        }
        if(entry->second == nullptr)
        {
            return {error_code_t::INVALID_VALUE,
                    std::string("cuDNN knob ") + knobTypeName(knobType)
                        + " is ambiguous: this engine exposes several knobs that map onto it"};
        }

        auto error = appendNativeKnobSetting(*entry->second, value, nativeSettings);
        if(error.is_bad())
        {
            return error;
        }
    }
    return {};
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail

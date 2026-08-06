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

inline std::optional<std::string> toHipdnnKnobId(KnobType_t knobType)
{
    switch(knobType)
    {
    case KnobType_t::SWIZZLE:
        return "swizzle";
    case KnobType_t::TILE_SIZE:
        return "tile_size";
    case KnobType_t::EDGE:
        return "edge";
    case KnobType_t::MULTIPLY:
        return "multiply";
    case KnobType_t::SPLIT_K_BUF:
        return "split_k_buf";
    case KnobType_t::TILEK:
        return "tilek";
    case KnobType_t::STAGES:
        return "stages";
    case KnobType_t::REDUCTION_MODE:
        return "reduction_mode";
    case KnobType_t::SPLIT_K_SLC:
        return "split_k_slc";
    case KnobType_t::IDX_MODE:
        return "idx_mode";
    case KnobType_t::SPECFILT:
        return "specfilt";
    case KnobType_t::KERNEL_CFG:
        return "kernel_cfg";
    case KnobType_t::WORKSPACE:
        return "global.workspace_size_limit";
    case KnobType_t::TILE_CGA_M:
        return "tile_cga_m";
    case KnobType_t::TILE_CGA_N:
        return "tile_cga_n";
    case KnobType_t::BLOCK_SIZE:
        return "block_size";
    case KnobType_t::OCCUPANCY:
        return "occupancy";
    case KnobType_t::ARRAY_SIZE_PER_THREAD:
        return "array_size_per_thread";
    case KnobType_t::SPLIT_COLS:
        return "split_cols";
    case KnobType_t::TILE_ROWS:
        return "tile_rows";
    case KnobType_t::TILE_COLS:
        return "tile_cols";
    case KnobType_t::LOAD_SIZE:
        return "load_size";
    case KnobType_t::CTA_COUNT:
        return "cta_count";
    case KnobType_t::STREAM_K:
        return "stream_k";
    case KnobType_t::SPLIT_P_SLC:
        return "split_p_slc";
    case KnobType_t::TILE_M:
        return "tile_m";
    case KnobType_t::TILE_N:
        return "tile_n";
    case KnobType_t::WARP_SPEC_CFG:
        return "warp_spec_cfg";
    case KnobType_t::NOT_SET:
    default:
        return std::nullopt;
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

inline error_t makeNativeKnobSettings(const std::unordered_map<KnobType_t, int64_t>& cudnnChoices,
                                      std::vector<hipdnn_frontend::KnobSetting>& nativeSettings)
{
    nativeSettings.clear();
    nativeSettings.reserve(cudnnChoices.size());
    for(const auto& [knobType, value] : cudnnChoices)
    {
        auto knobId = toHipdnnKnobId(knobType);
        if(!knobId.has_value())
        {
            return {error_code_t::INVALID_VALUE, "Unsupported cuDNN knob type for hipDNN shim"};
        }
        nativeSettings.emplace_back(*knobId, value);
    }
    return {};
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail

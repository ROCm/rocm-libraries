// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file ArchMatch.hpp
 * @brief Match a candidate arch against a device's raw gcnArchName, for use by
 * plugin/provider code.
 */

#include <string_view>

namespace hipdnn_plugin_sdk
{

/// How a candidate arch string is matched against a device's raw gcnArchName.
///
/// Named after the matching algorithm rather than "strict"/"loose" because
/// STRICT collides with a Windows SDK macro (minwindef.h: #define STRICT 1).
enum class ArchMatchMode
{
    /// Candidate must be the base-arch prefix of the device string, terminated
    /// by ':' or end-of-string. "gfx942" matches "gfx942:sramecc+:xnack-" but
    /// "gfx94" does NOT match "gfx942". Use for an exact base-arch gate (e.g.
    /// "gfx90a" only). Note: a bare family stem like "gfx115" will NOT match
    /// "gfx1150" under this mode -- use SUBSTRING for family matching.
    PREFIX,
    /// Candidate is any literal substring of the device string. "gfx10" matches
    /// "gfx1030". Use for arch-family gates where one stem (e.g. "gfx115") is
    /// meant to cover a whole family (gfx1150, gfx1151, ...).
    SUBSTRING,
};

/// Does `candidate` match `deviceArch` under the given mode?
///
/// This is the matcher used by the engine providers (arch-gated workarounds).
/// See ArchMatchMode for the semantics of each mode.
inline bool archMatches(std::string_view deviceArch, std::string_view candidate, ArchMatchMode mode)
{
    if(mode == ArchMatchMode::SUBSTRING)
    {
        return deviceArch.find(candidate) != std::string_view::npos;
    }

    // PREFIX: candidate is a prefix of deviceArch followed by ':' or end.
    // e.g. candidate "gfx942" matches device "gfx942:sramecc+:xnack-".
    return deviceArch.size() >= candidate.size()
           && deviceArch.compare(0, candidate.size(), candidate) == 0
           && (deviceArch.size() == candidate.size() || deviceArch[candidate.size()] == ':');
}

} // namespace hipdnn_plugin_sdk

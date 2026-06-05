// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace hipdnn_test_sdk::utilities
{

/// How a candidate arch string is matched against a device's raw gcnArchName.
enum class ArchMatchMode
{
    /// Candidate must be the base-arch prefix of the device string, terminated
    /// by ':' or end-of-string. "gfx942" matches "gfx942:sramecc+:xnack-" but
    /// "gfx94" does NOT match "gfx942". Use for arch-locking golden data, where
    /// data generated on gfx942 must not run on gfx940.
    STRICT,
    /// Candidate is any literal substring of the device string. "gfx10" matches
    /// "gfx1030". Use for human-written test-skip patterns where one entry
    /// (e.g. "gfx10") is meant to cover a whole architecture family.
    LOOSE,
};

/// Does `candidate` match `deviceArch` under the given mode?
///
/// This is the single source of truth for arch matching across the test
/// infrastructure (golden-data arch guard + test-skip system). See
/// ArchMatchMode for the semantics of each mode.
inline bool archMatches(std::string_view deviceArch, std::string_view candidate, ArchMatchMode mode)
{
    if(mode == ArchMatchMode::LOOSE)
    {
        return deviceArch.find(candidate) != std::string_view::npos;
    }

    // Strict: candidate is a prefix of deviceArch followed by ':' or end.
    // e.g. candidate "gfx942" matches device "gfx942:sramecc+:xnack-".
    return deviceArch.size() >= candidate.size()
           && deviceArch.compare(0, candidate.size(), candidate) == 0
           && (deviceArch.size() == candidate.size() || deviceArch[candidate.size()] == ':');
}

} // namespace hipdnn_test_sdk::utilities

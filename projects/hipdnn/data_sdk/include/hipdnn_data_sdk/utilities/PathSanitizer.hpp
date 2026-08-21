// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// PathSanitizer turns an arbitrary string (e.g. a scoped engine name such as
// "hipkernel:Pointwise") into a string that is legal as a single path component on every
// platform hipDNN supports, without needing a global registry of previously-seen names.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <string>
#include <string_view>

namespace hipdnn_data_sdk::utilities
{

/// Sanitizes @p raw into a single path component safe to use as a directory or file name
/// component on Linux and Windows alike.
///
/// The result is always `<human-readable sanitized stem>-<fixed-width hash of raw>`: the
/// hash suffix is unconditional, appended to every result regardless of whether the
/// sanitized stem alone would have collided with another input. Because the raw input's
/// own hash is always part of the output, two distinct raw inputs never produce the same
/// result -- the mapping is trivially injective and needs no runtime collision detection
/// or registry of names seen so far.
///
/// The sanitized stem portion handles, at minimum:
///  - the colon (every conforming scoped engine name contains exactly one; colons are
///    replaced, not merely tolerated, since they are illegal in a Windows path component);
///  - Windows-reserved stems (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9), matched
///    case-insensitively against the stem before any extension, and altered so the result
///    is not itself a reserved name;
///  - leading and trailing dots, which are stripped (a lone leading dot makes a Unix
///    dotfile, a lone trailing dot is silently dropped by some Windows APIs);
///  - an overall length cap, so the sanitized stem plus hash suffix never exceeds a
///    filesystem's practical component-length limit.
///
/// @param raw The arbitrary, caller-supplied string to sanitize (e.g. an engine name).
///     May be empty; every std::string_view is a valid input.
/// @return A single, non-empty path component safe to use on Linux and Windows. Never
///     throws.
inline std::string sanitizeForPath(std::string_view raw)
{
    // hash first: computed over the untouched raw input, so it is unaffected by anything
    // the stem-sanitization below does to the same characters.
    const uint64_t hash = fnv1aHash(raw);

    // Length cap for the human-readable stem, leaving headroom for the "-<16 hex digits>"
    // suffix within a filesystem's typical 255-byte path-component limit.
    constexpr size_t MAX_STEM_LENGTH = 96;

    // Characters illegal (or reserved) in a Windows path component: colon (the one every
    // conforming scoped engine name is guaranteed to contain), plus the rest of Windows'
    // reserved set, replaced defensively even though most never appear in an engine name.
    auto isIllegal = [](char c) {
        switch(c)
        {
        case ':':
        case '/':
        case '\\':
        case '*':
        case '?':
        case '"':
        case '<':
        case '>':
        case '|':
            return true;
        default:
            return false;
        }
    };

    std::string stem;
    stem.reserve(std::min(raw.size(), MAX_STEM_LENGTH));
    for(const char c : raw)
    {
        if(stem.size() >= MAX_STEM_LENGTH)
        {
            break;
        }
        stem.push_back(isIllegal(c) ? '_' : c);
    }

    // Strip leading/trailing dots -- a lone leading dot makes a Unix dotfile, a lone
    // trailing dot is silently dropped by some Windows APIs.
    const size_t firstNonDot = stem.find_first_not_of('.');
    if(firstNonDot == std::string::npos)
    {
        stem.clear();
    }
    else
    {
        const size_t lastNonDot = stem.find_last_not_of('.');
        stem = stem.substr(firstNonDot, lastNonDot - firstNonDot + 1);
    }

    // Windows-reserved stems (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9), matched
    // case-insensitively against the whole stem (there is no extension to strip first --
    // sanitizeForPath's output never carries one).
    static constexpr std::array<std::string_view, 22> RESERVED_STEMS = {
        "CON",  "PRN",  "AUX",  "NUL",  "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
        "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    };
    const bool isReserved = [&stem] {
        std::string upper;
        upper.reserve(stem.size());
        for(const char c : stem)
        {
            upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        }
        for(const std::string_view reserved : RESERVED_STEMS)
        {
            if(upper == reserved)
            {
                return true;
            }
        }
        return false;
    }();

    if(stem.empty() || isReserved)
    {
        // Reserved names are Windows-illegal regardless of case; an empty stem (raw was
        // empty, all-illegal, or all dots) still needs a non-empty component. Either way,
        // appending one underscore both breaks the reserved-name match and yields a
        // non-empty stem -- the hash suffix below is what actually disambiguates it.
        stem.push_back('_');
    }

    std::array<char, 17> hexBuffer{};
    std::snprintf(
        hexBuffer.data(), hexBuffer.size(), "%016llx", static_cast<unsigned long long>(hash));

    return stem + "-" + hexBuffer.data();
}

} // namespace hipdnn_data_sdk::utilities

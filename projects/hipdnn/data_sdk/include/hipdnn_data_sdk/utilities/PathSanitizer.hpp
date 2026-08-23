// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// PathSanitizer turns an arbitrary string (e.g. a scoped engine name such as
// "hipkernel:Pointwise") into a string legal as a single path component on every platform
// hipDNN supports.

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
/// The result is always `<sanitized stem>-<fixed-width hash of raw>`. The hash suffix is
/// unconditional, so two distinct inputs never produce the same result without needing a
/// registry of names seen so far.
///
/// The stem handles the colon (illegal on Windows), Windows-reserved names (CON, PRN,
/// COM1-9, LPT1-9, matched case-insensitively), leading/trailing dots (stripped -- a lone
/// leading dot makes a Unix dotfile, a trailing one is dropped by some Windows APIs), and
/// an overall length cap so stem plus hash stay within a filesystem's component limit.
///
/// @param raw May be empty; every std::string_view is a valid input.
/// @return A single, non-empty path component safe on Linux and Windows. Never throws.
inline std::string sanitizeForPath(std::string_view raw)
{
    const uint64_t hash = fnv1aHash(raw);

    constexpr size_t MAX_STEM_LENGTH = 96;

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
        // An empty stem (raw was empty, all-illegal, or all dots) still needs a non-empty
        // component; the hash suffix is what actually disambiguates it.
        stem.push_back('_');
    }

    std::array<char, 17> hexBuffer{};
    std::snprintf(
        hexBuffer.data(), hexBuffer.size(), "%016llx", static_cast<unsigned long long>(hash));

    return stem + "-" + hexBuffer.data();
}

} // namespace hipdnn_data_sdk::utilities

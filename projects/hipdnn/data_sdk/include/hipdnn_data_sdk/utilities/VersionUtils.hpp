// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>

namespace hipdnn_data_sdk::utilities
{
namespace detail
{
// Parses one unsigned decimal version component at the start of `text` and removes it from
// `text`. Fails on a missing digit, a sign, or a value that does not fit `int`.
inline bool consumeVersionComponent(std::string_view& text, int& value)
{
    if(text.empty() || text.front() < '0' || text.front() > '9')
    {
        return false;
    }
    const auto* first = text.data();
    const auto* last = first + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value, 10);
    if(ec != std::errc{})
    {
        return false;
    }
    text.remove_prefix(static_cast<size_t>(ptr - first));
    return true;
}

inline bool consumeVersionDot(std::string_view& text)
{
    if(text.empty() || text.front() != '.')
    {
        return false;
    }
    text.remove_prefix(1);
    return true;
}

inline bool isVersionTweak(std::string_view text)
{
    if(text.empty())
    {
        return false;
    }
    for(const char c : text)
    {
        const bool isAlnum
            = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        if(!isAlnum)
        {
            return false;
        }
    }
    return true;
}
} // namespace detail

struct Version
{
    int major, minor, patch;

    Version()
        : major(0)
        , minor(0)
        , patch(0)
    {
    }

    Version(int inMajor, int inMinor, int inPatch)
        : major(inMajor)
        , minor(inMinor)
        , patch(inPatch)
    {
    }

    /**
     * @brief Parses "MAJOR.MINOR.PATCH" with an optional ".TWEAK" suffix
     *
     * MAJOR, MINOR and PATCH are unsigned decimal numbers that fit `int`; no sign or
     * whitespace is accepted. TWEAK is one or more ASCII letters or digits (shipped version
     * strings carry the git hash or `unknown` there) and is ignored.
     *
     * @throws std::invalid_argument when the string does not match this grammar
     */
    Version(const std::string& versionStr)
    {
        std::string_view rest(versionStr);
        const bool coreParsed
            = detail::consumeVersionComponent(rest, major) && detail::consumeVersionDot(rest)
              && detail::consumeVersionComponent(rest, minor) && detail::consumeVersionDot(rest)
              && detail::consumeVersionComponent(rest, patch);
        const bool suffixValid
            = rest.empty() || (detail::consumeVersionDot(rest) && detail::isVersionTweak(rest));
        if(!coreParsed || !suffixValid)
        {
            throw std::invalid_argument("Version string does not match required format. String = "
                                        + versionStr
                                        + " Required format: \"MAJOR.MINOR.PATCH[.TWEAK]\"");
        }
    }

    Version(std::string_view versionStr)
        : Version(std::string(versionStr))
    {
    }

    std::string str() const
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

inline bool operator==(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           == std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator!=(const Version& lhs, const Version& rhs)
{
    return !(lhs == rhs);
}

inline bool operator<(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           < std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator>(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           > std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator<=(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           <= std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator>=(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           >= std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}
}

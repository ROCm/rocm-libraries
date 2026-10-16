// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

/// @file CliNumbers.hpp
/// @brief Strict numeric command-line values: the whole text is the number, or there is none.
///
/// `strtol` and friends read "abc" as 0 and "5x" as 5, so a mistyped value runs as a different
/// request than the one asked for. These refuse both.
namespace hipdnn_tools
{

/// A decimal integer filling all of @p text.
inline std::optional<int64_t> parseInteger(const std::string& text)
{
    int64_t value = 0;
    const auto* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if(text.empty() || error != std::errc{} || stop != end)
    {
        return std::nullopt;
    }
    return value;
}

/// A finite real number filling all of @p text.
inline std::optional<double> parseReal(const std::string& text)
{
    double value = 0.0;
    const auto* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if(text.empty() || error != std::errc{} || stop != end || !std::isfinite(value))
    {
        return std::nullopt;
    }
    return value;
}

/// An engine id: signed or unsigned decimal, or 0x-prefixed hex as `hipdnn_list_engines`
/// prints it. Ids are 64-bit patterns, so an unsigned value of 2^63 or more names the id with
/// the same bits (18446744073709551615 is -1).
inline std::optional<int64_t> parseEngineId(const std::string& text)
{
    if(const auto id = hipdnn_data_sdk::utilities::detail::parseEngineNumericId(text))
    {
        return id;
    }
    uint64_t value = 0;
    const auto* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if(text.empty() || error != std::errc{} || stop != end)
    {
        return std::nullopt;
    }
    return static_cast<int64_t>(value);
}

} // namespace hipdnn_tools

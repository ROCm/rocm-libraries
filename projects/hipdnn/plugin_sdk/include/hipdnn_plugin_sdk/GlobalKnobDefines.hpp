// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <charconv>
#include <cstddef>
#include <optional>
#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_plugin_sdk
{
static constexpr const char* BENCHMARKING_KNOB_NAME = "global.benchmarking";
static constexpr const char* WORKSPACE_SIZE_LIMIT_KNOB_NAME = "global.workspace_size_limit";

/// Process-wide override for `global.benchmarking`, independent of any engine's knob.
/// Not gated on HIPDNN_ENABLE_KERNEL_INGESTOR: MIOpen consumes it in flag-off builds too.
static constexpr const char* FORCE_BENCHMARKING_ENV_NAME = "HIPDNN_FORCE_BENCHMARKING";

/// Reads and parses HIPDNN_FORCE_BENCHMARKING per call, never cached. Normalizes with
/// toLower(trim(...)) and matches a literal set, following
/// hipdnn_data_sdk::logging::detail::stringToSeverity(). Unset, empty, whitespace-only,
/// or any unrecognized value returns std::nullopt.
///
/// Unset and empty are indistinguishable -- getEnv() yields "" for both -- and are
/// silent. Any other value that fails to parse logs a WARN naming the variable and the
/// value.
inline std::optional<bool> benchmarkingOverrideFromEnv()
{
    const auto raw = hipdnn_data_sdk::utilities::getEnv(FORCE_BENCHMARKING_ENV_NAME);
    const std::string normalized
        = hipdnn_data_sdk::utilities::toLower(hipdnn_data_sdk::utilities::trim(raw));

    if(raw.empty())
    {
        return std::nullopt;
    }
    if(normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes"
       || normalized == "enable" || normalized == "enabled")
    {
        return true;
    }
    if(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no"
       || normalized == "disable" || normalized == "disabled")
    {
        return false;
    }

    HIPDNN_PLUGIN_LOG_WARN("ignoring unrecognized value for " << FORCE_BENCHMARKING_ENV_NAME
                                                              << ": '" << raw << "'");
    return std::nullopt;
}

/// Process-wide cap on how many of the best-ranked candidates the kernel ingestor builds a
/// plan from. Read by GenericPlanBuilder only; an engine with thousands of applicable
/// kernels otherwise loads and benchmarks every one of them on a cold plan build.
///
/// Two effects worth knowing before setting it: (1) if every kernel in the shortlist fails
/// to prepare, the build throws rather than falling through to a lower-ranked kernel outside
/// the limit -- a build that would otherwise have succeeded can now fail outright. (2) any
/// limit, even one larger than the knob-filtered candidate set, makes a benchmarked winner
/// record that only covers that (smaller) set serve the build instead of being ignored as
/// partial coverage -- so turning the limit on and off changes whether an existing record is
/// trusted, not just how many candidates are tried.
static constexpr const char* INGESTOR_SHORTLIST_LIMIT_ENV_NAME = "HIPDNN_INGESTOR_SHORTLIST_LIMIT";

/// Reads and parses HIPDNN_INGESTOR_SHORTLIST_LIMIT per call, never cached. A positive
/// decimal integer, surrounding whitespace allowed, is the limit. Unset, empty, and
/// whitespace-only are silent and return std::nullopt, meaning no limit. Anything else --
/// zero, a sign, trailing text, a value past size_t -- logs a WARN naming the variable and
/// the value and also returns std::nullopt: a typo must not quietly shortlist.
inline std::optional<size_t> ingestorShortlistLimitFromEnv()
{
    const auto raw = hipdnn_data_sdk::utilities::getEnv(INGESTOR_SHORTLIST_LIMIT_ENV_NAME);
    const std::string normalized = hipdnn_data_sdk::utilities::trim(raw);
    if(normalized.empty())
    {
        return std::nullopt;
    }

    size_t limit = 0;
    const auto* begin = normalized.data();
    const auto* end = begin + normalized.size();
    const auto [ptr, ec] = std::from_chars(begin, end, limit);
    if(ec == std::errc{} && ptr == end && limit > 0)
    {
        return limit;
    }

    HIPDNN_PLUGIN_LOG_WARN("ignoring invalid value for " << INGESTOR_SHORTLIST_LIMIT_ENV_NAME
                                                         << ": '" << raw
                                                         << "'; expected a positive integer");
    return std::nullopt;
}
}

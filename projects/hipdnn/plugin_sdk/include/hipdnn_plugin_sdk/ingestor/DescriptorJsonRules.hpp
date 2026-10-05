// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

/// @file DescriptorJsonRules.hpp
/// @brief The error type and unknown-key rule every descriptor JSON reader shares: the
///        descriptor loader and the `graph_match.nodes` pattern parser.

namespace hipdnn_plugin_sdk::ingestor::detail
{

/// Every parse violation leaves through here, so the caller catches one type. The message
/// carries the file path only because `where` is the path: the caller logs it too, and the
/// duplication is worth an exception that is readable on its own.
[[noreturn]] inline void fail(const std::string& message)
{
    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}

/// Extension data has to look like extension data: a key starting `x-` or `_`, plus
/// `provenance`, the one unprefixed block the descriptor packager emits. Those warn and
/// are ignored, so a descriptor can carry tracking fields the loader has no use for.
///
/// Anything else the struct does not spell fails the file, because the alternative is
/// silent damage. A UED spelling `heuristik` has no heuristic key, and absence is legal:
/// the engine loads and ranks by the fallback heuristic for the rest of its life. A KDP
/// spelling `arh` is arch-independent and dispatches its kernels on every GPU. The
/// default log level is `off` (LogLevel.hpp), so a warning about either reaches nobody.
/// A producer that wants a key of its own prefixes it -- a one-line change there, against
/// a descriptor tree that otherwise cannot be trusted to mean what it spells.
inline bool isExtensionKey(std::string_view key)
{
    return key.rfind("x-", 0) == 0 || key.rfind('_', 0) == 0 || key == "provenance";
}

inline void requireKnownKeys(const nlohmann::json& object,
                             const std::string_view* firstAllowed,
                             const std::string_view* lastAllowed,
                             const std::string& where)
{
    for(const auto& item : object.items())
    {
        if(std::find(firstAllowed, lastAllowed, item.key()) != lastAllowed)
        {
            continue;
        }
        if(!isExtensionKey(item.key()))
        {
            fail("unknown key '" + item.key() + "' in " + where
                 + "; extension keys must start with 'x-' or '_'");
        }
        HIPDNN_PLUGIN_LOG_WARN("descriptor loader: extension key '" << item.key() << "' in "
                                                                    << where << "; ignoring it");
    }
}

inline void requireKnownKeys(const nlohmann::json& object,
                             std::initializer_list<std::string_view> allowed,
                             const std::string& where)
{
    requireKnownKeys(object, allowed.begin(), allowed.end(), where);
}

template <size_t N>
inline void requireKnownKeys(const nlohmann::json& object,
                             const std::array<std::string_view, N>& allowed,
                             const std::string& where)
{
    requireKnownKeys(object, allowed.data(), allowed.data() + N, where);
}

} // namespace hipdnn_plugin_sdk::ingestor::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

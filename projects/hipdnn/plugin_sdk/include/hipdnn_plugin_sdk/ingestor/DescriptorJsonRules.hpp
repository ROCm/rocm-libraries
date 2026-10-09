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
/// @brief Error helper and unknown-key rule shared by the descriptor loader and the
///        `graph_match.nodes` pattern parser.

namespace hipdnn_plugin_sdk::ingestor::detail
{

/// Throws the one exception type every parse violation uses. Callers build `message` with
/// a locator that names the descriptor file and, for a pattern error, the engine, so the
/// message reads on its own even though the caller also logs the path.
[[noreturn]] inline void fail(const std::string& message)
{
    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}

/// True for an extension key: one starting `x-` or `_`, or `provenance`, the one
/// unprefixed block the descriptor packager emits. Extension keys log a warning and are
/// ignored, so a descriptor can carry tracking fields the loader does not use.
///
/// Any other key the struct does not define fails the file, because a typo would otherwise
/// do silent damage. A UED spelling `heuristik` has no heuristic, which is legal, so the
/// engine ranks by the fallback heuristic forever. A KDP spelling `arh` becomes
/// arch-independent and dispatches its kernels on every GPU. The default log level is
/// `off` (LogLevel.hpp), so a warning would reach nobody. A producer that needs its own key
/// gives it a prefix.
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

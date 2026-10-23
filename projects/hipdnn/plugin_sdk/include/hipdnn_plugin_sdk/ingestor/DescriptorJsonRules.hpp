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
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

/// @file DescriptorJsonRules.hpp
/// @brief Error helper, identifier rule, and required-key, unknown-key and duplicate checks
///        shared by the descriptor loader and the `graph_match.nodes` pattern parser.

namespace hipdnn_plugin_sdk::ingestor::detail
{

/// Throws the one exception type every parse violation uses. Callers build `message` with
/// a locator that names the descriptor file and, for a pattern error, the engine, so the
/// message reads on its own even though the caller also logs the path.
[[noreturn]] inline void fail(const std::string& message)
{
    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}

/// The descriptor packager's one unprefixed tracking block.
inline constexpr std::string_view PROVENANCE_KEY = "provenance";

/// True for an extension key: one starting `x-` or `_`, or `provenance`, the one
/// unprefixed block the descriptor packager emits. Extension keys are ignored, so a
/// descriptor can carry tracking fields the loader does not use. Prefixed keys log a
/// warning; `provenance` does not. The packager keeps it inline only on a KDP header (a
/// packed UKD's lives in the provenance sidecar), but unpacked and authored trees carry it
/// on any descriptor, so it is ignored silently rather than warning once per kernel.
///
/// Any other key the struct does not define fails the file, because a typo would otherwise
/// do silent damage. A UED spelling `heuristik` has no heuristic, which is legal, so the
/// engine ranks by the fallback heuristic forever. A KDP spelling `arh` becomes
/// arch-independent and dispatches its kernels on every GPU. The default log level is
/// `off` (LogLevel.hpp), so a warning would reach nobody. A producer that needs its own key
/// gives it a prefix.
inline bool isExtensionKey(std::string_view key)
{
    return key.rfind("x-", 0) == 0 || key.rfind('_', 0) == 0 || key == PROVENANCE_KEY;
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
        if(item.key() == PROVENANCE_KEY)
        {
            continue;
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

/// True for `[A-Za-z_][A-Za-z0-9_]*`, the form of a token root.
inline bool isIdentifier(std::string_view text)
{
    const auto isLeading
        = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    if(text.empty() || !isLeading(text.front()))
    {
        return false;
    }
    return std::all_of(text.begin() + 1, text.end(), [&isLeading](char c) {
        return isLeading(c) || (c >= '0' && c <= '9');
    });
}

inline void requireObject(const nlohmann::json& value, const std::string& where)
{
    if(!value.is_object())
    {
        fail(where + " must be a JSON object");
    }
}

inline const nlohmann::json&
    requireKey(const nlohmann::json& object, std::string_view key, const std::string& where)
{
    const auto it = object.find(std::string(key));
    if(it == object.end())
    {
        fail("missing required key '" + std::string(key) + "' in " + where);
    }
    return *it;
}

/// Duplicates in any of the UED's three lists are authoring mistakes rather than
/// redundancies to collapse: a repeated knob is reported twice in EngineDetails, and a
/// repeated note twice in diagnostics (RFC 0020 §4.2, `uniqueItems`).
inline void requireNoDuplicates(const std::vector<std::string>& values,
                                std::string_view what,
                                const std::string& where)
{
    for(auto value = values.begin(); value != values.end(); ++value)
    {
        if(std::find(values.begin(), value, *value) != value)
        {
            fail(std::string(what) + " '" + *value + "' is listed twice in " + where);
        }
    }
}

} // namespace hipdnn_plugin_sdk::ingestor::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

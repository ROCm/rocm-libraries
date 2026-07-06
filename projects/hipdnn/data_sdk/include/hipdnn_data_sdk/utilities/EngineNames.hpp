// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cctype>
#include <charconv>
#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Converts an engine name string to a deterministic int64_t ID
 *
 * This function uses the FNV-1a hash algorithm to convert engine names
 * to unique IDs. The hash is deterministic - the same input will always
 * produce the same output.
 *
 * @param engineName The name of the engine to convert to an ID
 * @return int64_t The unique engine ID
 */
inline int64_t engineNameToId(const char* engineName) noexcept
{
    return static_cast<int64_t>(fnv1aHash(engineName));
}

/**
 * @brief Overload for std::string
 */
inline int64_t engineNameToId(const std::string& engineName)
{
    return static_cast<int64_t>(fnv1aHash(engineName));
}

/**
 * @brief Overload for std::string_view
 */
inline int64_t engineNameToId(std::string_view engineName)
{
    return static_cast<int64_t>(fnv1aHash(engineName));
}

// Internal namespace for mutable access (used only by EngineRegistrar)
namespace detail
{
inline std::set<std::string_view>& getMutableEngineNames()
{
    static std::set<std::string_view> s_allEngines;
    return s_allEngines;
}

inline std::unordered_map<int64_t, std::string_view>& getMutableEngineIdToNameMap()
{
    static std::unordered_map<int64_t, std::string_view> s_engineIdToNameMap;
    return s_engineIdToNameMap;
}
} // namespace detail

// Public const access functions
inline const std::set<std::string_view>& getAllEngineNames()
{
    return detail::getMutableEngineNames();
}

inline const std::unordered_map<int64_t, std::string_view>& getEngineIdToNameMap()
{
    return detail::getMutableEngineIdToNameMap();
}

// Helper function to check if an engine name is registered
inline bool isEngineNameRegistered(std::string_view name)
{
    return getAllEngineNames().find(name) != getAllEngineNames().end();
}

// Helpers for parsing serialized numeric engine IDs.
namespace detail
{
inline bool hasLeadingWhitespace(std::string_view value)
{
    return !value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0;
}

inline std::optional<int64_t> parseEngineNumericId(std::string_view value)
{
    if(value.empty() || hasLeadingWhitespace(value))
    {
        return std::nullopt;
    }

    if(value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X'))
    {
        uint64_t parsed = 0;
        const auto* first = value.data() + 2;
        const auto* last = value.data() + value.size();
        const auto [ptr, ec] = std::from_chars(first, last, parsed, 16);
        if(ec == std::errc{} && ptr == last)
        {
            return static_cast<int64_t>(parsed);
        }
        return std::nullopt;
    }

    int64_t parsed = 0;
    const bool hasLeadingPlus = value.front() == '+';
    const auto* first = value.data() + (hasLeadingPlus ? 1 : 0);
    const auto* last = value.data() + value.size();
    if(first == last)
    {
        return std::nullopt;
    }
    const auto [ptr, ec] = std::from_chars(first, last, parsed, 10);
    if(ec == std::errc{} && ptr == last)
    {
        return parsed;
    }
    return std::nullopt;
}
} // namespace detail

inline int64_t engineNameOrIdToId(std::string_view engineName)
{
    if(isEngineNameRegistered(engineName))
    {
        return engineNameToId(engineName);
    }
    if(const auto parsed = detail::parseEngineNumericId(engineName); parsed.has_value())
    {
        return *parsed;
    }
    return engineNameToId(engineName);
}

// Helper to format engine ID as hex string
inline std::string formatEngineIdHex(int64_t id)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << id;
    return oss.str();
}

// Helper function to get engine name from ID (returns empty if not found)
inline std::string_view getEngineNameFromId(int64_t id)
{
    auto& idToName = getEngineIdToNameMap();
    auto it = idToName.find(id);
    if(it != idToName.end())
    {
        return it->second;
    }

    throw std::out_of_range("Engine ID " + formatEngineIdHex(id)
                            + " not found in registered engines");
}

struct EngineRegistrar
{
    EngineRegistrar(std::string_view name)
    {
        auto id = engineNameToId(name);

        // Check for duplicate registration or hash collision BEFORE inserting
        auto& idToNameMap = detail::getMutableEngineIdToNameMap();
        auto it = idToNameMap.find(id);
        if(it != idToNameMap.end())
        {
            if(it->second == name)
            {
                throw std::runtime_error("Duplicate engine registration detected! '"
                                         + std::string(name) + "' is already registered with ID: "
                                         + formatEngineIdHex(id));
            }

            throw std::runtime_error("Engine name collision detected! '" + std::string(it->second)
                                     + "' and '" + std::string(name)
                                     + "' both hash to ID: " + formatEngineIdHex(id));
        }

        detail::getMutableEngineNames().insert(name);
        idToNameMap[id] = name;
    }
};

/// @cond INTERNAL
#define DETAIL_HIPDNN_REGISTER_ENGINE_1(name)                                           \
    inline constexpr const char* name##_NAME = #name;                                   \
    inline const int64_t name##_ID = hipdnn_data_sdk::utilities::engineNameToId(#name); \
    inline const hipdnn_data_sdk::utilities::EngineRegistrar name##_registrar{#name};

#define DETAIL_HIPDNN_REGISTER_ENGINE_2(name, value)                                    \
    inline constexpr const char* name##_NAME = value;                                   \
    inline const int64_t name##_ID = hipdnn_data_sdk::utilities::engineNameToId(value); \
    inline const hipdnn_data_sdk::utilities::EngineRegistrar name##_registrar{value};

#define DETAIL_HIPDNN_GET_REGISTER_ENGINE_MACRO(_1, _2, NAME, ...) NAME
/// @endcond

/**
 * @def HIPDNN_REGISTER_ENGINE
 * @brief Macro that defines an engine and automatically registers it.
 *
 * Supports 1 or 2 arguments:
 * - `HIPDNN_REGISTER_ENGINE(MyEngine)` → identifier: `MyEngine`, name: `"MyEngine"`
 * - `HIPDNN_REGISTER_ENGINE(MyEngine, "MyEngine")` → identifier: `MyEngine`, name: `"MyEngine"`
 *
 * The single-parameter form should be used whenever possible for simplicity.
 *
 * @param name The identifier used for generated constants (e.g., `name_NAME`, `name_ID`)
 * @param value (optional) Custom string name for the engine. Defaults to stringified `name`.
 */
#define HIPDNN_REGISTER_ENGINE(...)                                                            \
    DETAIL_HIPDNN_GET_REGISTER_ENGINE_MACRO(                                                   \
        __VA_ARGS__, DETAIL_HIPDNN_REGISTER_ENGINE_2, DETAIL_HIPDNN_REGISTER_ENGINE_1, unused) \
    (__VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////////////////
// Define all engines using the HIPDNN_REGISTER_ENGINE() macro.
// Note: Once an engine is named here, it should never be renamed.  Renaming an engine will
// change the generated uint64_t ID.
////////////////////////////////////////////////////////////////////////////////////////////
// NOLINTBEGIN(bugprone-throwing-static-initialization) collision detection requires throw

HIPDNN_REGISTER_ENGINE(HIP_MLOPS_ENGINE)
HIPDNN_REGISTER_ENGINE(HIPBLASLT_ENGINE)
HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE)
HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE_DETERMINISTIC)
HIPDNN_REGISTER_ENGINE(ASM_SDPA_ENGINE)
HIPDNN_REGISTER_ENGINE(ROCKE_ENGINE)

// NOLINTEND(bugprone-throwing-static-initialization)
////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hipdnn_data_sdk::utilities

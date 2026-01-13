// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <set>
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
inline int64_t engineNameToId(const char* engineName)
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

inline std::set<std::string_view>& getAllEngineNames()
{
    static std::set<std::string_view> s_allEngines;
    return s_allEngines;
}

inline std::unordered_map<int64_t, std::string_view>& getEngineIdToNameMap()
{
    static std::unordered_map<int64_t, std::string_view> s_engineIdToNameMap;
    return s_engineIdToNameMap;
}

// Helper function to check if an engine name is registered
inline bool isEngineNameRegistered(std::string_view name)
{
    return getAllEngineNames().find(name) != getAllEngineNames().end();
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
    return "";
}

struct EngineRegistrar
{
    EngineRegistrar(std::string_view name)
    {
        getAllEngineNames().insert(name);
        auto id = engineNameToId(name.data());
        getEngineIdToNameMap()[id] = name;

        // Check for collisions
        for(const auto& [existingId, existingName] : getEngineIdToNameMap())
        {
            if(existingId == id && existingName != name)
            {
                HIPDNN_LOG_ERROR(
                    "Engine name collision detected! '{}' and '{}' both hash to ID: 0x{:016X}",
                    existingName,
                    name,
                    id);
            }
        }
    }
};

// Macro that defines engine and automatically registers it
#define HIPDNN_REGISTER_ENGINE(name, value)                 \
    inline constexpr const char* name##_NAME = value;       \
    inline const int64_t name##_ID = engineNameToId(value); \
    inline const EngineRegistrar name##_registrar{value};

// Define all engines using the macro
HIPDNN_REGISTER_ENGINE(MIOPEN_PLUGIN, "MIOPEN_PLUGIN")

} // namespace hipdnn_data_sdk::utilities

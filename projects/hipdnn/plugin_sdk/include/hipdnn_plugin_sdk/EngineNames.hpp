// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineIdHash.hpp>
#include <set>
#include <string_view>
#include <unordered_map>

namespace hipdnn_plugin_sdk::engine_names
{

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
        auto id = hipdnn_data_sdk::engineNameToId(name.data());
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
#define HIPDNN_REGISTER_ENGINE(name, value)                                  \
    inline constexpr const char* name = value;                               \
    inline const int64_t name##_ID = hipdnn_data_sdk::engineNameToId(value); \
    inline const EngineRegistrar name##_registrar{value};

// Define all engines using the macro
HIPDNN_REGISTER_ENGINE(MIOPEN_PLUGIN, "MIOPEN_PLUGIN")

} // namespace hipdnn_plugin_sdk::engine_names

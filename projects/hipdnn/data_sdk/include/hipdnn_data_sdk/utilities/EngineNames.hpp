// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <iomanip>
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
        detail::getMutableEngineNames().insert(name);
        auto id = engineNameToId(name.data());
        detail::getMutableEngineIdToNameMap()[id] = name;

        // Check for collisions
        for(const auto& [existingId, existingName] : getEngineIdToNameMap())
        {
            if(existingId == id && existingName != name)
            {
                throw std::runtime_error("Engine name collision detected! '"
                                         + std::string(existingName) + "' and '" + std::string(name)
                                         + "' both hash to ID: " + formatEngineIdHex(id));
            }
        }
    }
};

// Macro that defines engine and automatically registers it
#define HIPDNN_REGISTER_ENGINE(name, value)                 \
    inline constexpr const char* name##_NAME = value;       \
    inline const int64_t name##_ID = engineNameToId(value); \
    inline const EngineRegistrar name##_registrar{value};

//Note: Once an engine is named here, it should never be renamed.  Renaming an engine will
// change the generated uint64_t ID.

// Define all engines using the macro
HIPDNN_REGISTER_ENGINE(FUSILLI_ENGINE, "FUSILLI_ENGINE")

HIPDNN_REGISTER_ENGINE(HIPBLASLT_ENGINE, "HIPBLASLT_ENGINE")

HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE, "MIOPEN_ENGINE")
HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE_DETERMINISTIC, "MIOPEN_ENGINE_DETERMINISTIC")

HIPDNN_REGISTER_ENGINE(HIP_KERNEL_ENGINE, "HIP_KERNEL_ENGINE")

} // namespace hipdnn_data_sdk::utilities

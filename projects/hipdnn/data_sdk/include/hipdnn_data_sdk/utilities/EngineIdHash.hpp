// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace hipdnn_data_sdk
{

/**
 * @brief Converts an engine name string to a deterministic int64_t ID
 *
 * This function implements a FNV-1a hash algorithm to convert engine names
 * to unique IDs. The hash is deterministic - the same input will always
 * produce the same output.
 *
 * @param engineName The name of the engine to convert to an ID
 * @return int64_t The unique engine ID
 */
inline int64_t engineNameToId(const char* engineName)
{
    if(engineName == nullptr || engineName[0] == '\0')
    {
        return 0;
    }

    // FNV-1a hash algorithm constants for 64-bit
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    uint64_t hash = FNV_OFFSET_BASIS;

    for(const char* p = engineName; *p != '\0'; ++p)
    {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
        hash *= FNV_PRIME;
    }

    // Cast to int64_t for the return value
    return static_cast<int64_t>(hash);
}

/**
 * @brief Overload for std::string
 */
inline int64_t engineNameToId(const std::string& engineName)
{
    return engineNameToId(engineName.c_str());
}

/**
 * @brief Overload for std::string_view
 */
inline int64_t engineNameToId(std::string_view engineName)
{
    return engineNameToId(std::string(engineName).c_str());
}

} // namespace hipdnn_data_sdk

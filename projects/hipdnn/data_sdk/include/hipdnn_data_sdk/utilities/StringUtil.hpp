// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Computes a FNV-1a hash of a string
 *
 * This function implements a FNV-1a hash algorithm to convert strings
 * to deterministic uint64_t values. The hash is deterministic - the same
 * input will always produce the same output.
 *
 * @param str The string to hash
 * @return uint64_t The hash value
 */
inline uint64_t fnv1aHash(const char* str)
{
    if(str == nullptr || str[0] == '\0')
    {
        return 0;
    }

    // FNV-1a hash algorithm constants for 64-bit
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    uint64_t hash = FNV_OFFSET_BASIS;

    for(const char* p = str; *p != '\0'; ++p)
    {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
        hash *= FNV_PRIME;
    }

    return hash;
}

/**
 * @brief Overload for std::string
 */
inline uint64_t fnv1aHash(const std::string& str)
{
    return fnv1aHash(str.c_str());
}

/**
 * @brief Overload for std::string_view
 */
inline uint64_t fnv1aHash(std::string_view str)
{
    return fnv1aHash(std::string(str).c_str());
}

inline void copyMaxSizeWithNullTerminator(char* destination, const char* source, size_t maxSize)
{
    if(source == nullptr || destination == nullptr || maxSize == 0)
    {
        return;
    }

#ifdef _WIN32
    strncpy_s(destination, maxSize, source, maxSize - 1);
#else
    std::strncpy(destination, source, maxSize - 1);
#endif
    destination[maxSize - 1] = '\0';
}

inline std::string toLower(const std::string& str)
{
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

inline std::string removeNewlines(const std::string& str)
{
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    return result;
}

// Converts a vector of numbers to a string "[1, 2, 3...]".
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string> // Restrict template to numeric types
    vecToString(const std::vector<T>& vec)
{
    std::string result = "[";
    if(!vec.empty())
    {
        result += std::to_string(vec[0]);
        for(size_t i = 1; i < vec.size(); ++i)
        {
            result += ", " + std::to_string(vec[i]);
        }
    }
    return result + "]";
}

// Converts a vector of objects to an ostream "[A, B, C...]".
template <typename T>
inline void vecToStream(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    if(!vec.empty())
    {
        os << vec[0];
        for(size_t i = 1; i < vec.size(); ++i)
        {
            os << ", " << vec[i];
        }
    }

    os << "]";
}

// Converts a vector of strings to an ostream "[ "A", "B" , "C" ...]".
inline void stringVecToStream(std::ostream& os, const std::vector<std::string>& vec)
{
    os << "[";
    for(size_t i = 0; i < vec.size(); ++i)
    {
        if(i > 0)
        {
            os << ", ";
        }
        os << "\"" << vec[i] << "\"";
    }
    os << "]";
}

// Converts a vector of objects to a string "[A, B, C...]".
template <typename T>
std::enable_if_t<!std::is_arithmetic_v<T>, std::string> vecToString(const std::vector<T>& vec)
{
    std::stringstream stream;
    vecToStream(stream, vec);
    return stream.str();
}

} // namespace hipdnn_data_sdk::utilities

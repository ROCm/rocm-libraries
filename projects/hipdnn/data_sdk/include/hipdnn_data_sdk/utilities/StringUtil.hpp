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
inline uint64_t fnv1aHash(const uint8_t* data, size_t size) noexcept
{
    if(data == nullptr || size == 0)
    {
        return 0;
    }

    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    uint64_t hash = FNV_OFFSET_BASIS;

    for(size_t i = 0; i < size; ++i)
    {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_PRIME;
    }

    return hash;
}

/**
 * @brief Computes a FNV-1a hash of a null-terminated string
 *
 * @param str The string to hash
 * @return uint64_t The hash value, or 0 for null/empty input
 */
inline uint64_t fnv1aHash(const char* str) noexcept
{
    if(str == nullptr)
    {
        return 0;
    }
    return fnv1aHash(reinterpret_cast<const uint8_t*>(str), std::strlen(str));
}

/**
 * @brief Overload for std::string
 */
inline uint64_t fnv1aHash(const std::string& str) noexcept
{
    return fnv1aHash(str.c_str());
}

/**
 * @brief Overload for std::string_view
 */
inline uint64_t fnv1aHash(std::string_view str) noexcept
{
    return fnv1aHash(reinterpret_cast<const uint8_t*>(str.data()), str.size());
}

// Builds a std::string from a length-delimited char buffer, stripping a
// trailing NUL if the producer included it in the reported length.
inline std::string bufferToString(const std::vector<char>& buf, size_t len)
{
    if(len > 0 && buf[len - 1] == '\0')
    {
        --len;
    }
    return {buf.data(), len};
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
    std::string lowerStr = str; // NOLINT(misc-const-correctness)
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

inline std::string trim(const std::string& str)
{
    const auto start = str.find_first_not_of(" \t\n\r\f\v");
    if(start == std::string::npos)
    {
        return "";
    }
    const auto end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

inline std::string removeNewlines(const std::string& str)
{
    std::string result = str; // NOLINT(misc-const-correctness)
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace hipdnn_data_sdk::utilities
{

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

// Converts a vector of objects to a string "[A, B, C...]".
template <typename T>
std::enable_if_t<!std::is_arithmetic_v<T>, std::string> vecToString(const std::vector<T>& vec)
{
    std::stringstream stream;
    vecToStream(stream, vec);
    return stream.str();
}

} // namespace hipdnn_data_sdk::utilities

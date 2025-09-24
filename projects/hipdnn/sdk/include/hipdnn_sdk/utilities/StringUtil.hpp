// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace hipdnn_sdk::utilities
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

template <typename T>
inline void vecToStream(std::ostream& os, const std::vector<T>& vec)
{
    if(vec.empty())
    {
        os << "[]";
        return;
    }

    os << "[";
    std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(os, ", "));
    os << vec.back();
    os << "]";
}

}

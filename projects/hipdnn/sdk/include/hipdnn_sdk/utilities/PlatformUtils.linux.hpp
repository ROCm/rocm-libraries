// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__linux__)
#include <filesystem>
namespace hipdnn_sdk
{
namespace utilities
{

constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";
constexpr const char* EXECUTABLE_EXT = "";

inline std::string getEnv(const char* var, const char* defaultValue = nullptr)
{
    std::string result = defaultValue != nullptr ? defaultValue : "";

    const char* value = std::getenv(var);

    if(value != nullptr)
    {
        result = value;
    }

    return result;
}

inline void setEnv(const char* var, const char* value)
{
    if(value != nullptr)
    {
        setenv(var, value, 1);
    }
}

inline void unsetEnv(const char* var)
{
    unsetenv(var);
}

inline bool pathCompEq(const std::filesystem::path& a, const std::filesystem::path& b)
{
    return a.native() == b.native();
}

}
}

#else

#error "Do not include PlatformUtils.linux.hpp in non-linux builds"

#endif

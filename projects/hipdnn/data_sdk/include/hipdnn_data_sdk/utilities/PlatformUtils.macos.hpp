// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__APPLE__)
#include <array>
#include <cstdlib>
#include <filesystem>
#include <mach-o/dyld.h>
#include <stdexcept>
#include <string>
#include <sys/syslimits.h>
#include <unistd.h>
#include <vector>

namespace hipdnn_data_sdk::utilities
{

constexpr const char* SHARED_LIB_EXT = ".dylib";
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

inline std::filesystem::path getCurrentExecutableDirectory()
{
    std::array<char, PATH_MAX> initialBuffer{};
    auto bufferSize = static_cast<uint32_t>(initialBuffer.size());

    if(_NSGetExecutablePath(initialBuffer.data(), &bufferSize) == 0)
    {
        return std::filesystem::path(initialBuffer.data()).parent_path();
    }

    std::vector<char> buffer(bufferSize);
    if(_NSGetExecutablePath(buffer.data(), &bufferSize) != 0)
    {
        throw std::runtime_error("Failed to get executable path");
    }

    return std::filesystem::path(buffer.data()).parent_path();
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.macos.hpp in non-macOS builds"

#endif

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <array>
#include <cwctype>
#include <filesystem>
#include <windows.h>

#include "StringUtil.hpp"

namespace hipdnn_data_sdk::utilities
{

constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";
constexpr const char* EXECUTABLE_EXT = ".exe";

inline std::string getEnv(const char* var, const char* defaultValue = nullptr)
{
    std::string result = defaultValue != nullptr ? defaultValue : "";

    GetEnvironmentVariableA(var, nullptr, 0);

    DWORD size = GetEnvironmentVariableA(var, nullptr, 0);
    if(size > 0)
    {
        char* dst = new char[size];
        GetEnvironmentVariableA(var, dst, size);
        result = dst;
        delete[] dst;
    }

    return result;
}

inline void setEnv(const char* var, const char* value)
{
    if(value != nullptr)
    {
        SetEnvironmentVariableA(var, value);
    }
}

inline void unsetEnv(const char* var)
{
    SetEnvironmentVariableA(var, nullptr);
}

inline bool pathCompEq(const std::filesystem::path& a, const std::filesystem::path& b)
{
    return toLower(a.string()) == toLower(b.string());
}

inline std::filesystem::path getCurrentExecutableDirectory()
{
    std::array<wchar_t, MAX_PATH> result{};
    DWORD length = GetModuleFileNameW(nullptr, result.data(), MAX_PATH);
    if(length == 0 || length == MAX_PATH)
    {
        throw std::runtime_error("Failed to get executable path");
    }
    return std::filesystem::path(result.data()).parent_path();
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.windows.hpp in non-windows builds"

#endif

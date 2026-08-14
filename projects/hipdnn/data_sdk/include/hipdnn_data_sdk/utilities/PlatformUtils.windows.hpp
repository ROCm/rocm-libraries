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
#include <stdexcept>
#include <string>
#include <system_error>
#include <windows.h>

#include "StringUtil.hpp"

namespace hipdnn_data_sdk::utilities
{

constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";
constexpr const char* EXECUTABLE_EXT = ".exe";
using SharedLibraryHandle = HMODULE;

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

inline SharedLibraryHandle openLibrary(const std::filesystem::path& libraryPath)
{
    auto handle = LoadLibraryW(libraryPath.wstring().c_str());
    if(handle == nullptr)
    {
        throw std::runtime_error("Failed to load library: " + libraryPath.string()
                                 + " (Error Code: " + std::to_string(GetLastError()) + ")");
    }
    return handle;
}

inline SharedLibraryHandle openLoadedLibrary(const std::filesystem::path& libraryPath)
{
    HMODULE handle = nullptr;
    if(GetModuleHandleExW(0, libraryPath.wstring().c_str(), &handle) == FALSE)
    {
        return nullptr;
    }
    return handle;
}

inline void closeLibrary(SharedLibraryHandle handle)
{
    FreeLibrary(handle);
}

inline void* getSymbol(SharedLibraryHandle handle, const char* symbolName)
{
    return reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
}

inline std::filesystem::path getLoadedLibraryDirectory(const char* libraryName)
{
    auto handle = GetModuleHandleW(std::filesystem::path(libraryName).wstring().c_str());
    if(handle == nullptr)
    {
        throw std::runtime_error("Failed to find loaded library: " + std::string(libraryName));
    }

    std::array<wchar_t, MAX_PATH> result{};
    const auto length = GetModuleFileNameW(handle, result.data(), result.size());
    if(length == 0 || length >= result.size())
    {
        throw std::runtime_error("Failed to get loaded library path: " + std::string(libraryName));
    }

    return std::filesystem::path(result.data()).parent_path();
}

/// The directory of the module @p address belongs to -- the Windows counterpart of the
/// dladdr() form: it works however the module was loaded and needs nothing exported, so
/// prefer it over the by-name form when asking "where am I loaded from" about the calling
/// module itself. Uses GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT since the caller isn't
/// taking ownership.
inline std::filesystem::path getLoadedLibraryDirectoryForAddress(const void* address)
{
    HMODULE handle = nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(address),
                          &handle)
       == 0)
    {
        throw std::runtime_error("Failed to find loaded library for address");
    }

    std::array<wchar_t, MAX_PATH> result{};
    const auto length = GetModuleFileNameW(handle, result.data(), result.size());
    if(length == 0 || length >= result.size())
    {
        throw std::runtime_error("Failed to get loaded library path for address");
    }

    // Canonicalized so a module reached through a symlink answers with the directory its
    // siblings actually sit in, same as the dladdr form on Linux.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(result.data(), failed);
    return (failed ? std::filesystem::path(result.data()) : resolved).parent_path();
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.windows.hpp in non-windows builds"

#endif

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32

// Keep caller-defined macros; undefine only those introduced for this include.
#ifndef NOMINMAX
#define NOMINMAX
#define HIPDNN_UNDEF_NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef HIPDNN_UNDEF_NOMINMAX
#undef NOMINMAX
#undef HIPDNN_UNDEF_NOMINMAX
#endif
#ifdef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif

#include <algorithm>
#include <array>
#include <cwctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include "StringUtil.hpp"

namespace hipdnn_data_sdk::utilities
{

constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";
constexpr const char* EXECUTABLE_EXT = ".exe";
using SharedLibraryHandle = HMODULE;

inline std::string getEnv(const char* var, const char* defaultValue = nullptr)
{
    // The sizing call counts the terminator, the fetching call does not.
    const DWORD size = GetEnvironmentVariableA(var, nullptr, 0);
    if(size == 0)
    {
        return defaultValue != nullptr ? defaultValue : "";
    }

    std::string value(size, '\0');
    value.resize(GetEnvironmentVariableA(var, value.data(), size));
    return value;
}

/// Reads native UTF-16 without getEnv()'s lossy ANSI conversion.
/// Use for native Windows paths.
inline std::wstring getEnvW(const wchar_t* var, const wchar_t* defaultValue = nullptr)
{
    // The sizing call counts the terminator, the fetching call does not.
    const DWORD size = GetEnvironmentVariableW(var, nullptr, 0);
    if(size == 0)
    {
        return defaultValue != nullptr ? defaultValue : L"";
    }

    std::wstring value(size, L'\0');
    value.resize(GetEnvironmentVariableW(var, value.data(), size));
    return value;
}

/// Always false: Windows has no AT_SECURE-equivalent execution mode.
inline bool isSecureExecution()
{
    return false;
}

/// Code-loading environment lookup; equivalent to getEnv() on Windows.
inline std::string getSecureEnv(const char* var, const char* defaultValue = nullptr)
{
    if(isSecureExecution())
    {
        return defaultValue != nullptr ? defaultValue : "";
    }
    return getEnv(var, defaultValue);
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

/// Expands leading `~` or case-insensitive `%USERPROFILE%` to USERPROFILE only
/// when alone or followed by a path separator. Returns @p path unchanged if no
/// token qualifies or USERPROFILE is unset/empty. Never throws.
inline std::string expandUser(const std::string& path)
{
    const bool hasLeadingTilde = !path.empty() && path.front() == '~'
                                 && (path.size() == 1 || path[1] == '/' || path[1] == '\\');

    static const std::string kUserProfileToken = "%userprofile%";
    const std::string lowerPath = toLower(path);
    const bool hasLeadingToken
        = lowerPath.size() >= kUserProfileToken.size()
          && lowerPath.compare(0, kUserProfileToken.size(), kUserProfileToken) == 0
          && (lowerPath.size() == kUserProfileToken.size() || path[kUserProfileToken.size()] == '/'
              || path[kUserProfileToken.size()] == '\\');

    if(!hasLeadingTilde && !hasLeadingToken)
    {
        return path;
    }

    const std::string userProfile = getEnv("USERPROFILE");
    if(userProfile.empty())
    {
        return path;
    }

    const size_t tokenLength = hasLeadingTilde ? 1 : kUserProfileToken.size();
    return userProfile + path.substr(tokenLength);
}

/// UTF-16 expandUser() with the same leading-token and fallback rules.
/// Use for native paths to preserve non-ASCII USERPROFILE values. Never throws.
inline std::wstring expandUserW(const std::wstring& path)
{
    const bool hasLeadingTilde = !path.empty() && path.front() == L'~'
                                 && (path.size() == 1 || path[1] == L'/' || path[1] == L'\\');

    static const std::wstring kUserProfileToken = L"%userprofile%";
    std::wstring lowerPath = path;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::towlower);
    const bool hasLeadingToken
        = lowerPath.size() >= kUserProfileToken.size()
          && lowerPath.compare(0, kUserProfileToken.size(), kUserProfileToken) == 0
          && (lowerPath.size() == kUserProfileToken.size() || path[kUserProfileToken.size()] == L'/'
              || path[kUserProfileToken.size()] == L'\\');

    if(!hasLeadingTilde && !hasLeadingToken)
    {
        return path;
    }

    const std::wstring userProfile = getEnvW(L"USERPROFILE");
    if(userProfile.empty())
    {
        return path;
    }

    const size_t tokenLength = hasLeadingTilde ? 1 : kUserProfileToken.size();
    return userProfile + path.substr(tokenLength);
}

inline bool pathCompEq(const std::filesystem::path& a, const std::filesystem::path& b)
{
    return CompareStringOrdinal(a.native().c_str(),
                                static_cast<int>(a.native().size()),
                                b.native().c_str(),
                                static_cast<int>(b.native().size()),
                                TRUE)
           == CSTR_EQUAL;
}

inline std::filesystem::path getCurrentExecutableDirectory()
{
    std::array<wchar_t, MAX_PATH> result{};
    const DWORD length = GetModuleFileNameW(nullptr, result.data(), MAX_PATH);
    if(length == 0 || length == MAX_PATH)
    {
        throw std::runtime_error("Failed to get executable path");
    }
    return std::filesystem::path(result.data()).parent_path();
}

inline SharedLibraryHandle openLibrary(const std::filesystem::path& libraryPath)
{
    auto handle = LoadLibraryW(libraryPath.c_str());
    if(handle == nullptr)
    {
        const DWORD error = GetLastError();
        throw std::runtime_error("Failed to load library: " + libraryPath.u8string()
                                 + " (Error Code: " + std::to_string(error) + ")");
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

/// Directory of exactly @p handle's module, without address or name lookup.
inline std::filesystem::path getLoadedLibraryOrigin(SharedLibraryHandle handle)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("Failed to get library origin: null handle");
    }

    std::array<wchar_t, MAX_PATH> result{};
    const auto length = GetModuleFileNameW(handle, result.data(), result.size());
    if(length == 0 || length >= result.size())
    {
        throw std::runtime_error(
            "Failed to get library origin (Error Code: " + std::to_string(GetLastError()) + ")");
    }

    // Resolve symlinks to find siblings; retain the module path on failure.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(result.data(), failed);
    return (failed ? std::filesystem::path(result.data()) : resolved).parent_path();
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

/// Directory owning @p address, regardless of exports or how the module was loaded.
/// Borrows the module handle without changing its reference count.
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

    // Resolve symlinks to find siblings; retain the module path on failure.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(result.data(), failed);
    return (failed ? std::filesystem::path(result.data()) : resolved).parent_path();
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.windows.hpp in non-windows builds"

#endif

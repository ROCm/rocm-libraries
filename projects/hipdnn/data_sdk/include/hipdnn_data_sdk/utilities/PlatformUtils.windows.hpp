// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
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

/// Wide sibling of getEnv(): reads through GetEnvironmentVariableW, which hands back the
/// environment's native UTF-16 verbatim. getEnv() above goes through
/// GetEnvironmentVariableA instead, which transcodes that same UTF-16 through the
/// process's ANSI code page -- a lossy, locale-dependent step for any non-ASCII
/// character. A caller that must feed the result to something UTF-16-native (notably
/// std::filesystem::path's wstring constructor, or CreateFileW) should read wide from the
/// start rather than narrow-then-reinterpret.
inline std::wstring getEnvW(const wchar_t* var, const wchar_t* defaultValue = nullptr)
{
    // Same two-call sizing convention as getEnv(): the sizing call counts the
    // terminator, the fetching call does not.
    const DWORD size = GetEnvironmentVariableW(var, nullptr, 0);
    if(size == 0)
    {
        return defaultValue != nullptr ? defaultValue : L"";
    }

    std::wstring value(size, L'\0');
    value.resize(GetEnvironmentVariableW(var, value.data(), size));
    return value;
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

/// Expands a **leading** `~` or **leading** `%USERPROFILE%` in @p path to the current
/// user's home directory; see PlatformUtils.linux.hpp's expandUser() for the full
/// leading-token/fallback contract, which applies here with `%USERPROFILE%` (matched
/// case-insensitively) accepted alongside `~`.
///
/// @param path The path string to expand, e.g. as read from a config value or env var.
/// @return @p path with a qualifying leading `~` or `%USERPROFILE%` replaced by
///     `%USERPROFILE%`'s value, or @p path unchanged if no leading token qualifies or
///     `USERPROFILE` is unset/empty. Never throws.
inline std::string expandUser(const std::string& path)
{
    // A leading '~' qualifies only alone or followed by a path separator.
    const bool hasLeadingTilde = !path.empty() && path.front() == '~'
                                 && (path.size() == 1 || path[1] == '/' || path[1] == '\\');

    // "%USERPROFILE%" matched as a literal leading token, case-insensitively.
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

/// Wide sibling of expandUser(): identical leading-token/fallback contract, but composed
/// entirely in UTF-16 via getEnvW() so a non-ASCII `%USERPROFILE%` value (e.g. a
/// non-ASCII Windows account name) survives intact. expandUser() above narrows through
/// getEnv()/GetEnvironmentVariableA, which is lossy for exactly that case -- see getEnvW()
/// for why -- and callers composing a std::filesystem::path (which MSVC's
/// std::filesystem interprets as UTF-8, not the ANSI code page) must use this instead.
///
/// @param path The path string to expand, as UTF-16.
/// @return @p path with a qualifying leading `~` or `%USERPROFILE%` replaced by
///     `%USERPROFILE%`'s value, or @p path unchanged if no leading token qualifies or
///     `USERPROFILE` is unset/empty. Never throws.
inline std::wstring expandUserW(const std::wstring& path)
{
    // A leading '~' qualifies only alone or followed by a path separator.
    const bool hasLeadingTilde = !path.empty() && path.front() == L'~'
                                 && (path.size() == 1 || path[1] == L'/' || path[1] == L'\\');

    // "%USERPROFILE%" matched as a literal leading token, case-insensitively.
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
    return toLower(a.string()) == toLower(b.string());
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

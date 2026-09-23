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
#include <cwctype>
#include <filesystem>
#include <limits>
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
    // The sizing call counts the terminator, the fetching call does not, so a fetch that
    // fits always reports less than it was given -- including zero for a variable that is
    // set to an empty value, which is a successful read and not an absent one. The
    // environment block is private to this process, so no other process can reach in
    // between the two calls, but another thread in this process can replace the value; when
    // the new value no longer fits, the fetch writes nothing and returns the size it now
    // requires, terminator included, so retry with that size until the fetch reports a
    // length that fits.
    DWORD size = GetEnvironmentVariableA(var, nullptr, 0);
    while(size != 0)
    {
        std::string value(size, '\0');
        const DWORD copied = GetEnvironmentVariableA(var, value.data(), size);
        if(copied < size)
        {
            value.resize(copied);
            return value;
        }
        size = copied;
    }

    return defaultValue != nullptr ? defaultValue : "";
}

/// Reads native UTF-16 without getEnv()'s lossy ANSI conversion.
/// Use for native Windows paths.
inline std::wstring getEnvW(const wchar_t* var, const wchar_t* defaultValue = nullptr)
{
    // Sized and retried exactly as getEnv() above; see that comment for the same-process
    // growth race and for why a zero-length fetch is an empty value rather than an absent
    // one.
    DWORD size = GetEnvironmentVariableW(var, nullptr, 0);
    while(size != 0)
    {
        std::wstring value(size, L'\0');
        const DWORD copied = GetEnvironmentVariableW(var, value.data(), size);
        if(copied < size)
        {
            value.resize(copied);
            return value;
        }
        size = copied;
    }

    return defaultValue != nullptr ? defaultValue : L"";
}

/// Always false: an unprivileged invoker's environment does not survive into a privileged
/// image on Windows. Elevation re-launches the image from the elevating shell and the new
/// process inherits that shell's environment, so there is no equivalent of a set-user-ID
/// execve where an attacker-controlled environment crosses a privilege boundary intact.
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

/// Wide counterpart of getSecureEnv(); equivalent to getEnvW() on Windows.
inline std::wstring getSecureEnvW(const wchar_t* var, const wchar_t* defaultValue = nullptr)
{
    if(isSecureExecution())
    {
        return defaultValue != nullptr ? defaultValue : L"";
    }
    return getEnvW(var, defaultValue);
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
/// Not secure-execution aware: USERPROFILE is read with getEnv(), not getSecureEnv().
/// Never use on a path that will subsequently be loaded as code.
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
/// Not secure-execution aware, exactly as expandUser() above.
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

namespace detail
{
/// Full path of @p handle's module, sized to whatever the loader reports it needs.
/// A null handle names the current executable.
/// @p what names the lookup being attempted in the failure messages, and @p subject, when
/// non-empty, names the module; callers that reach a module by handle or by address have
/// no name to report.
inline std::wstring moduleFileName(HMODULE handle, const char* what, const char* subject = nullptr)
{
    const auto describe = [what, subject] {
        std::string message = "Failed to get ";
        message += what;
        if(subject != nullptr && *subject != '\0')
        {
            message += ": ";
            message += subject;
        }
        return message;
    };

    // Extended-length paths exceed MAX_PATH, and GetModuleFileNameW reports truncation
    // by filling the buffer rather than failing. Retry with a larger one instead of
    // discarding a path this module genuinely has.
    std::wstring modulePath(MAX_PATH, L'\0');
    DWORD capacity = static_cast<DWORD>(modulePath.size());
    DWORD length = GetModuleFileNameW(handle, modulePath.data(), capacity);
    if(length == 0)
    {
        const DWORD error = GetLastError();
        throw std::runtime_error(describe() + " (Error Code: " + std::to_string(error) + ")");
    }

    while(length >= capacity)
    {
        constexpr DWORD MAX_CAPACITY = (std::numeric_limits<DWORD>::max)() / 2;
        if(capacity > MAX_CAPACITY)
        {
            throw std::runtime_error(describe() + ": path is implausibly long");
        }
        capacity *= 2;

        modulePath.assign(capacity, L'\0');
        length = GetModuleFileNameW(handle, modulePath.data(), capacity);
        if(length == 0)
        {
            const DWORD error = GetLastError();
            throw std::runtime_error(describe() + " (Error Code: " + std::to_string(error) + ")");
        }
    }

    modulePath.resize(length);
    return modulePath;
}
} // namespace detail

inline std::filesystem::path getCurrentExecutableDirectory()
{
    return std::filesystem::path(detail::moduleFileName(nullptr, "executable path")).parent_path();
}

namespace detail
{
/// Opens @p libraryPath through @p flags; zero is the plain LoadLibraryW search order.
inline SharedLibraryHandle openLibraryWithFlags(const std::filesystem::path& libraryPath,
                                                DWORD flags)
{
    HMODULE handle = LoadLibraryExW(libraryPath.c_str(), nullptr, flags);
    if(handle == nullptr)
    {
        const DWORD error = GetLastError();
        // The error code is captured above, before any formatting that could fail.
        throw std::runtime_error("Failed to load library: " + pathForDiagnostic(libraryPath)
                                 + " (Error Code: " + std::to_string(error) + ")");
    }
    return handle;
}
} // namespace detail

/// Opens @p libraryPath on the loader's standard search order, which resolves the opened
/// module's own dependents from the application directory. Libraries that ship in a
/// subdirectory of the application -- engine and heuristic plugins -- depend on that, since
/// Windows has no RPATH equivalent to point them back at their dependents.
inline SharedLibraryHandle openLibrary(const std::filesystem::path& libraryPath)
{
    return detail::openLibraryWithFlags(libraryPath, 0);
}

/// Opens @p libraryPath with its own directory searched first for its first-level
/// dependents. The alternate order substitutes that directory for the application
/// directory rather than adding to it, so a dependent shipped beside the executable and
/// not beside @p libraryPath resolves through %PATH%; use openLibrary() unless the opened
/// module is known to sit with its dependents. LOAD_WITH_ALTERED_SEARCH_PATH is documented
/// as undefined for a relative path, so one stays on the standard order.
inline SharedLibraryHandle
    openLibraryWithOwnDirectoryFirst(const std::filesystem::path& libraryPath)
{
    return detail::openLibraryWithFlags(
        libraryPath, libraryPath.is_absolute() ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
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

    const std::wstring modulePath = detail::moduleFileName(handle, "library origin");

    // Resolve symlinks to find siblings; retain the module path on failure.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(modulePath, failed);
    return (failed ? std::filesystem::path(modulePath) : resolved).parent_path();
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

    return std::filesystem::path(detail::moduleFileName(handle, "loaded library path", libraryName))
        .parent_path();
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

    const std::wstring modulePath
        = detail::moduleFileName(handle, "loaded library path for address");

    // Resolve symlinks to find siblings; retain the module path on failure.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(modulePath, failed);
    return (failed ? std::filesystem::path(modulePath) : resolved).parent_path();
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.windows.hpp in non-windows builds"

#endif

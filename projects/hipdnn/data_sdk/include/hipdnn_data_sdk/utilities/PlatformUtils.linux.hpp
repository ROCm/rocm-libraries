// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__linux__)
#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <stdexcept>
#include <string>
#include <sys/auxv.h>
#include <system_error>
#include <unistd.h>

namespace hipdnn_data_sdk::utilities
{

constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";
constexpr const char* EXECUTABLE_EXT = "";
using SharedLibraryHandle = void*;

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

/// Reads AT_SECURE for privilege-elevated execution without relying on glibc's
/// secure_getenv(). Fails closed: an unavailable entry is treated as secure.
inline bool isSecureExecution()
{
    // Distinguish a missing AT_SECURE entry from a genuine zero.
    errno = 0;
    const unsigned long secure = getauxval(AT_SECURE);
    if(secure == 0 && errno != 0)
    {
        return true;
    }
    return secure != 0;
}

/// Reads environment values that control code loading or execution.
/// In secure execution, ignores the invoker-controlled environment and returns
/// @p defaultValue, or an empty string if no default is supplied.
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
        setenv(var, value, 1);
    }
}

inline void unsetEnv(const char* var)
{
    unsetenv(var);
}

/// Expands a leading `~` to HOME only when alone or followed by `/`; never expands `~user`.
/// Returns @p path unchanged if no token qualifies or HOME is unset/empty.
/// Never throws.
inline std::string expandUser(const std::string& path)
{
    if(path.empty() || path.front() != '~')
    {
        return path;
    }

    if(path.size() > 1 && path[1] != '/')
    {
        return path;
    }

    const std::string home = getEnv("HOME");
    if(home.empty())
    {
        return path;
    }

    return home + path.substr(1);
}

inline bool pathCompEq(const std::filesystem::path& a, const std::filesystem::path& b)
{
    return a.native() == b.native();
}

inline std::filesystem::path getCurrentExecutableDirectory()
{
    std::array<char, PATH_MAX + 1> result{}; // +1 for trailing null termination
    const ssize_t count = readlink("/proc/self/exe", result.data(), PATH_MAX);
    if(count == -1)
    {
        throw std::runtime_error("Failed to get executable path");
    }
    return std::filesystem::path(std::string(result.data(), static_cast<size_t>(count)))
        .parent_path();
}

inline SharedLibraryHandle openLibrary(const std::filesystem::path& libraryPath)
{
    auto* handle = dlopen(libraryPath.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if(handle == nullptr)
    {
        const char* error = dlerror();
        throw std::runtime_error("Failed to load library: " + libraryPath.string() + " ("
                                 + (error != nullptr ? std::string(error) : "Unknown error") + ")");
    }
    return handle;
}

inline SharedLibraryHandle openLoadedLibrary(const std::filesystem::path& libraryPath)
{
    return dlopen(libraryPath.string().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
}

/// The canonical parent of an already-open library's absolute loader path.
/// Relative or empty loader names cannot reliably identify an origin.
inline std::filesystem::path getLoadedLibraryOrigin(SharedLibraryHandle handle)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("Failed to get library origin: null handle");
    }

    link_map* map = nullptr;
    if(dlinfo(handle, RTLD_DI_LINKMAP, static_cast<void*>(&map)) != 0)
    {
        const char* error = dlerror();
        throw std::runtime_error("Failed to get library origin ("
                                 + (error != nullptr ? std::string(error) : "Unknown error") + ")");
    }

    if(map == nullptr || map->l_name == nullptr || map->l_name[0] != '/')
    {
        throw std::runtime_error("Failed to get library origin: no absolute loader path");
    }

    const std::filesystem::path libraryPath(map->l_name);
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(libraryPath, failed);
    return (failed ? libraryPath : resolved).parent_path();
}

inline void closeLibrary(SharedLibraryHandle handle)
{
    dlclose(handle);
}

inline void* getSymbol(SharedLibraryHandle handle, const char* symbolName)
{
    auto _ = dlerror();
    return dlsym(handle, symbolName);
}

/// The directory of the module owning @p address. Normally launched dynamic
/// executables use /proc/self/exe; other images use their canonicalized loader path.
inline std::filesystem::path getLoadedLibraryDirectoryForAddress(const void* address)
{
    Dl_info info{};
    void* owner = nullptr;
    if(dladdr1(address, &info, &owner, RTLD_DL_LINKMAP) == 0 || owner == nullptr)
    {
        throw std::runtime_error("Failed to find loaded library for address");
    }
    const auto* map = static_cast<const link_map*>(owner);
    if(map->l_name == nullptr)
    {
        throw std::runtime_error("Failed to find loaded library for address");
    }
    // An explicitly invoked ld.so is /proc/self/exe, not the address owner.
    if(map->l_name[0] == '\0' && getauxval(AT_BASE) != 0)
    {
        return getCurrentExecutableDirectory();
    }
    if(info.dli_fname == nullptr || info.dli_fname[0] == '\0')
    {
        throw std::runtime_error("Failed to find loaded library for address");
    }

    // Keep the loader path if best-effort canonicalization fails.
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(info.dli_fname, failed);
    return (failed ? std::filesystem::path(info.dli_fname) : resolved).parent_path();
}

/// Directory exporting @p symbolName in the dynamic linker's default scope.
/// Use getLoadedLibraryDirectoryForAddress() to identify a specific module.
inline std::filesystem::path getLoadedLibraryDirectoryForSymbol(const char* symbolName)
{
    auto _ = dlerror();
    void* symbol = dlsym(RTLD_DEFAULT, symbolName);
    const char* error = dlerror();
    if(error != nullptr)
    {
        throw std::runtime_error("Failed to find loaded symbol: " + std::string(symbolName) + " ("
                                 + error + ")");
    }

    try
    {
        return getLoadedLibraryDirectoryForAddress(symbol);
    }
    catch(const std::runtime_error&)
    {
        throw std::runtime_error("Failed to find loaded library for symbol: "
                                 + std::string(symbolName));
    }
}

} // namespace hipdnn_data_sdk::utilities

#else

#error "Do not include PlatformUtils.linux.hpp in non-linux builds"

#endif

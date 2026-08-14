// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__linux__)
#include <array>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
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

inline void closeLibrary(SharedLibraryHandle handle)
{
    dlclose(handle);
}

inline void* getSymbol(SharedLibraryHandle handle, const char* symbolName)
{
    auto _ = dlerror();
    return dlsym(handle, symbolName);
}

/// The directory of the module @p address belongs to.
///
/// Works under every dlopen flag and needs nothing exported: the address already names
/// the module. Prefer this over the symbol-name form when asking "where am I loaded
/// from" about the calling module itself -- pass the address of one of its own functions.
inline std::filesystem::path getLoadedLibraryDirectoryForAddress(const void* address)
{
    Dl_info info{};
    if(dladdr(address, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0')
    {
        throw std::runtime_error("Failed to find loaded library for address");
    }

    return std::filesystem::path(info.dli_fname).parent_path();
}

/// The directory of the module exporting @p symbolName.
///
/// Resolves through the dynamic linker's default scope, so the answer depends on what is
/// loaded and how. To ask about the calling module itself, use
/// getLoadedLibraryDirectoryForAddress() instead -- it cannot pick a different module.
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

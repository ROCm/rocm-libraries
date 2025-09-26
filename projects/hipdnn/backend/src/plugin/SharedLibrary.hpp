// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlatformUtils.hpp"
#include <filesystem>
#include <string_view>

namespace hipdnn_backend
{
namespace plugin
{

class SharedLibrary
{
public:
    SharedLibrary();
    explicit SharedLibrary(const std::filesystem::path& libraryPath);
    SharedLibrary(SharedLibrary&& other) noexcept;
    SharedLibrary(const SharedLibrary&) = delete;

    ~SharedLibrary();

    SharedLibrary& operator=(SharedLibrary&& other) noexcept;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    // This function loads a shared library from the specified path.
    // On Windows, it adds a ".dll" extension if no extension exists.
    // On Linux, it adds a "lib" prefix to the filename and a ".so" extension if no extension exists.
    void load(const std::filesystem::path& libraryPath);

    // This function unloads the shared library.
    void unload() noexcept;

    void* getSymbol(std::string_view symbolName) const;

    template <class T>
    T getSymbol(std::string_view symbolName) const
    {
        void* symbol = getSymbol(symbolName);
        return reinterpret_cast<T>(symbol);
    }

    const std::filesystem::path& libraryPath() const;

private:
    std::filesystem::path _libraryPath;

    hipdnn_backend::platform_utilities::PluginLibHandle _libraryHandle;
};

} // namespace plugin
} // hipdnn_backend

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SharedLibrary.hpp"
#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "logging/Logging.hpp"

namespace hipdnn_backend
{
namespace plugin
{

SharedLibrary::SharedLibrary()
    : _libraryHandle(nullptr)
{
    // Default constructor does nothing
}

SharedLibrary::SharedLibrary(const std::filesystem::path& libraryPath)
    : _libraryHandle(nullptr)
{
    load(libraryPath);
}

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept
    : _libraryHandle(other._libraryHandle)
{
    other._libraryHandle = nullptr;
}

SharedLibrary::~SharedLibrary()
{
    unload();
}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept
{
    if(this != &other)
    {
        // Release current resources
        unload();

        // Transfer ownership
        _libraryHandle = other._libraryHandle;
        other._libraryHandle = nullptr;
    }
    return *this;
}

// This function loads a shared library from the specified path.
// On Windows, it adds a ".dll" extension if no extension exists.
// On Linux, it adds a "lib" prefix to the filename and a ".so" extension if no extension exists.
void SharedLibrary::load(const std::filesystem::path& libraryPath)
{
    if(_libraryHandle != nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Library is already loaded.");
    }

    auto modifiedLibraryPath = libraryPath;

    // Check file extension and add prefix/suffix if needed
    if(modifiedLibraryPath.has_extension())
    {
        if(modifiedLibraryPath.extension() != hipdnn_sdk::utilities::SHARED_LIB_EXT)
        {
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                                  std::string("Invalid file extension. Expected ")
                                      + hipdnn_sdk::utilities::SHARED_LIB_EXT);
        }
    }
    else
    {
        auto libraryName = hipdnn_sdk::utilities::getLibraryName(
            modifiedLibraryPath.filename().string().c_str());
        modifiedLibraryPath = modifiedLibraryPath.parent_path() / libraryName;
    }

    if(modifiedLibraryPath.is_relative())
    {
        // Paths are typically resolved by here, but this is a fallback for some unit tests
        modifiedLibraryPath
            = hipdnn_backend::platform_utilities::getCurrentModuleDirectory() / modifiedLibraryPath;
    }

    _libraryPath = std::filesystem::weakly_canonical(modifiedLibraryPath);

    // Needs to be a resolved, weakly canonical path at this point
    if(!std::filesystem::exists(_libraryPath))
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Shared libary: plugin file does not exist: "
                                  + _libraryPath.string());
    }

    HIPDNN_LOG_INFO("SharedLibrary: Attempting to load shared library from final absolute path: {}",
                    _libraryPath.string());

    _libraryHandle = platform_utilities::openLibrary(_libraryPath);
}

void SharedLibrary::unload() noexcept
{
    if(_libraryHandle != nullptr)
    {
        platform_utilities::closeLibrary(_libraryHandle);
        _libraryHandle = nullptr;
    }
}

void* SharedLibrary::getSymbol(std::string_view symbolName) const
{
    if(_libraryHandle == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Library is not loaded. Cannot get symbol: "
                                  + std::string(symbolName));
    }

    return platform_utilities::getSymbol(_libraryHandle, symbolName.data());
}

const std::filesystem::path& SharedLibrary::libraryPath() const
{
    return _libraryPath;
}

} // namespace plugin
} // hipdnn_backend

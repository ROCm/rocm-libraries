// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CustomLibraryAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <dlfcn.h> // POSIX dlopen / dlsym / dlclose

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

namespace
{

/// C ABI scorer function signature.
using ScorerFunc = double (*)(const double*, size_t);

} // namespace

std::unique_ptr<CustomLibraryAdapter>
    CustomLibraryAdapter::load(const std::string& libraryPath,
                                const std::string& symbolName,
                                size_t numFeatures,
                                const std::string& expectedFeaturesHash)
{
    if(libraryPath.empty())
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: libraryPath is empty");
        return nullptr;
    }

    if(symbolName.empty())
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbolName is empty for library "
                             << libraryPath);
        return nullptr;
    }

    // dlopen the library (RTLD_NOW to catch missing symbols at load time)
    void* libHandle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(libHandle == nullptr)
    {
        const char* err = dlerror();
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: dlopen failed for " << libraryPath << ": "
                                                                         << (err ? err : "unknown"));
        return nullptr;
    }

    // Reset dlerror before dlsym (POSIX requirement)
    dlerror();

    // Lookup the scorer symbol
    void* sym = dlsym(libHandle, symbolName.c_str());
    const char* err = dlerror();
    if(err != nullptr || sym == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: dlsym failed for symbol '" << symbolName
                                                                                << "' in "
                                                                                << libraryPath
                                                                                << ": " << (err ? err : "symbol not found"));
        dlclose(libHandle);
        return nullptr;
    }

    HIPDNN_SDK_LOG_INFO("CustomLibraryAdapter: loaded " << libraryPath << " symbol '"
                                                         << symbolName << "' (features="
                                                         << numFeatures << ")");

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - dlsym returns void*
    return std::unique_ptr<CustomLibraryAdapter>(new CustomLibraryAdapter(
        libHandle, sym, numFeatures, expectedFeaturesHash, libraryPath));
}

CustomLibraryAdapter::CustomLibraryAdapter(void* libHandle,
                                           void* scorerFunc,
                                           size_t numFeatures,
                                           std::string featuresHash,
                                           std::string libraryPath)
    : _libHandle(libHandle),
      _scorerFunc(scorerFunc),
      _numFeatures(numFeatures),
      _featuresHash(std::move(featuresHash)),
      _libraryPath(std::move(libraryPath))
{
}

CustomLibraryAdapter::~CustomLibraryAdapter()
{
    if(_libHandle != nullptr)
    {
        if(dlclose(_libHandle) != 0)
        {
            const char* err = dlerror();
            HIPDNN_SDK_LOG_WARN("CustomLibraryAdapter: dlclose failed for " << _libraryPath
                                                                             << ": " << (err ? err : "unknown"));
        }
    }
}

double CustomLibraryAdapter::score(const std::vector<double>& features) const
{
    if(features.size() != _numFeatures)
    {
        std::ostringstream oss;
        oss << "CustomLibraryAdapter: feature count mismatch. Expected " << _numFeatures
            << ", got " << features.size();
        throw std::invalid_argument(oss.str());
    }

    // Cast to the C ABI function pointer and call
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - C ABI requires function pointer cast
    auto scorer = reinterpret_cast<ScorerFunc>(_scorerFunc);
    return scorer(features.data(), features.size());
}

} // namespace hipdnn_backend::heuristics::uhd

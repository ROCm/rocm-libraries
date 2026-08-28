// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CustomLibraryAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
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

    // plugin::SharedLibrary rather than a dlopen here. The POSIX call this replaced made
    // the whole ingestor unbuildable on Windows -- <dlfcn.h> does not exist there, and
    // HIPDNN_ENABLE_KERNEL_INGESTOR=ON is set on the Windows CI lane. The backend already
    // owned a cross-platform loader for exactly this; a second port would have been a
    // second thing to keep correct.
    //
    // It throws where this function returns nullptr, so the conversion happens here: a
    // custom_library artifact is author-supplied and may simply be absent or wrong, which
    // RFC 0019 §5 wants degraded to declared order rather than failed.
    plugin::SharedLibrary library;
    try
    {
        library.load(libraryPath);
    }
    catch(const std::exception& error)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: failed to load " << libraryPath << ": "
                                                                     << error.what());
        return nullptr;
    }

    void* sym = nullptr;
    try
    {
        sym = library.getSymbol(symbolName);
    }
    catch(const std::exception& error)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbol '"
                             << symbolName << "' not found in " << libraryPath << ": "
                             << error.what());
        return nullptr;
    }
    if(sym == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbol '" << symbolName << "' resolved to "
                                                              << "null in " << libraryPath);
        return nullptr;
    }

    HIPDNN_SDK_LOG_INFO("CustomLibraryAdapter: loaded " << libraryPath << " symbol '"
                                                         << symbolName << "' (features="
                                                         << numFeatures << ")");

    return std::unique_ptr<CustomLibraryAdapter>(new CustomLibraryAdapter(
        std::move(library), sym, numFeatures, expectedFeaturesHash, libraryPath));
}

CustomLibraryAdapter::CustomLibraryAdapter(plugin::SharedLibrary library,
                                           void* scorerFunc,
                                           size_t numFeatures,
                                           std::string featuresHash,
                                           std::string libraryPath)
    : _library(std::move(library)),
      _scorerFunc(scorerFunc),
      _numFeatures(numFeatures),
      _featuresHash(std::move(featuresHash)),
      _libraryPath(std::move(libraryPath))
{
}

/// Declared, not defaulted in the header, because ~SharedLibrary needs its complete type
/// and the destructor is the only member that requires it here.
CustomLibraryAdapter::~CustomLibraryAdapter() = default;

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

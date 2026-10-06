// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/IUhdAdapter.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file CustomLibraryAdapter.hpp
/// @brief RFC 0019 §7.3 escape hatch: a scorer the in-tree walker cannot express.
/// Header-only so the shared adapter factory can construct it.
namespace hipdnn_plugin_sdk::uhd
{

namespace detail
{

/// Whether @p path's bytes hash to @p expectedHash. Hashes the whole file so appended bytes
/// are caught, and bounds the size before allocating for an unverified file.
inline bool artifactHashMatches(const std::filesystem::path& path, const std::string& expectedHash)
{
    constexpr std::streamoff MAX_ARTIFACT_BYTES = std::streamoff{256} * 1024 * 1024;
    const auto shown = hipdnn_data_sdk::utilities::detail::pathForDiagnostic(path);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: cannot open " << shown
                                                                  << " to verify its hash");
        return false;
    }
    const auto size = file.tellg();
    if(size <= 0 || size > MAX_ARTIFACT_BYTES)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: " << shown
                                                      << " is empty or exceeds the artifact "
                                                         "size bound; hash not verified");
        return false;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: " << shown << " could not be read in full; "
                                                      << "hash not verified");
        return false;
    }

    const auto actual = sha256(bytes.data(), bytes.size());
    if(actual != expectedHash)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: model hash mismatch for "
                             << shown << " - expected='" << expectedHash << "' actual='" << actual
                             << "'; the model is not used -- ranking degrades to static_order "
                                "and an engine estimate is reported as 0");
        return false;
    }
    return true;
}

} // namespace detail

/// @brief Custom library adapter for compiled scorers (RFC 0019 §7.3).
///
/// Loads a shipped shared library (e.g. Treelite output) and calls a C ABI scorer:
///
///     extern "C" double <symbol>(const double* features, size_t num_features);
///
/// The library exports no feature contract: the feature count and features hash it reports
/// are the descriptor's own (RFC 0019 OQ11).
class CustomLibraryAdapter : public IUhdAdapter
{
public:
    /// @brief Loads a custom library scorer, searching the library's own directory first
    /// for its dependents.
    /// @param expectedModelHash SHA-256 of the library bytes (`custom_library.hash`); empty
    ///        when undeclared (RFC 0019 §4.1). A declared hash must match.
    /// @return nullptr on any load failure, so a malformed descriptor degrades to
    ///         static_order (RFC 0019 §5) instead of failing the request.
    static std::unique_ptr<CustomLibraryAdapter> load(const std::filesystem::path& libraryPath,
                                                      const std::string& symbolName,
                                                      size_t numFeatures,
                                                      const std::string& expectedFeaturesHash,
                                                      const std::string& expectedModelHash = "");

    ~CustomLibraryAdapter() override
    {
        hipdnn_data_sdk::utilities::closeLibrary(_libHandle);
    }

    /// Non-copyable and non-movable: the handle is unloaded exactly once.
    CustomLibraryAdapter(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter& operator=(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter(CustomLibraryAdapter&&) = delete;
    CustomLibraryAdapter& operator=(CustomLibraryAdapter&&) = delete;

    double score(const std::vector<double>& features) const override
    {
        if(features.size() != _numFeatures)
        {
            // The callee reads num_features entries through a raw pointer, so a short row
            // would be an out-of-bounds read.
            std::ostringstream message;
            message << "CustomLibraryAdapter: feature count mismatch. Expected " << _numFeatures
                    << ", got " << features.size();
            throw std::invalid_argument(message.str());
        }

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - C ABI function pointer
        auto* scorer = reinterpret_cast<ScorerFunc>(_scorerFunc);
        return scorer(features.data(), features.size());
    }

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

private:
    /// Same signature as UhdScoreFn, so one scorer can serve `native` or this adapter.
    using ScorerFunc = double (*)(const double*, size_t);

    CustomLibraryAdapter(hipdnn_data_sdk::utilities::SharedLibraryHandle libHandle,
                         void* scorerFunc,
                         size_t numFeatures,
                         std::string featuresHash)
        : _libHandle(libHandle)
        , _scorerFunc(scorerFunc)
        , _numFeatures(numFeatures)
        , _featuresHash(std::move(featuresHash))
    {
    }

    hipdnn_data_sdk::utilities::SharedLibraryHandle _libHandle;
    void* _scorerFunc; ///< function pointer, cast before calling
    size_t _numFeatures;
    std::string _featuresHash;
};

inline std::unique_ptr<CustomLibraryAdapter>
    CustomLibraryAdapter::load(const std::filesystem::path& libraryPath,
                               const std::string& symbolName,
                               size_t numFeatures,
                               const std::string& expectedFeaturesHash,
                               const std::string& expectedModelHash)
{
    namespace platform = hipdnn_data_sdk::utilities;
    if(libraryPath.empty())
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: libraryPath is empty");
        return nullptr;
    }
    const auto shown = platform::detail::pathForDiagnostic(libraryPath);
    if(symbolName.empty())
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbolName is empty for library " << shown);
        return nullptr;
    }

    // Verify before opening: loading runs the library's initialisers.
    if(!expectedModelHash.empty() && !detail::artifactHashMatches(libraryPath, expectedModelHash))
    {
        return nullptr;
    }

    platform::SharedLibraryHandle libHandle = nullptr;
    try
    {
        libHandle = platform::openLibraryWithOwnDirectoryFirst(libraryPath);
    }
    catch(const std::exception& error)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: " << error.what());
        return nullptr;
    }

    void* symbol = platform::getSymbol(libHandle, symbolName.c_str());
    if(symbol == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbol '" << symbolName << "' not found in "
                                                              << shown);
        platform::closeLibrary(libHandle);
        return nullptr;
    }

    HIPDNN_SDK_LOG_INFO("CustomLibraryAdapter: loaded " << shown << " symbol '" << symbolName
                                                        << "' (features=" << numFeatures << ")");

    return std::unique_ptr<CustomLibraryAdapter>(
        new CustomLibraryAdapter(libHandle, symbol, numFeatures, expectedFeaturesHash));
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

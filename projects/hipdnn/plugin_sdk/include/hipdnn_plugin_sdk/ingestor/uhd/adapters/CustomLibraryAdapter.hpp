// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/IUhdAdapter.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <dlfcn.h> // POSIX dlopen / dlsym / dlclose

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file CustomLibraryAdapter.hpp
/// @brief RFC 0019 §7.2's escape hatch: a scorer the in-tree walker cannot express.
///
/// Moved here from the backend. It was implemented beside a kernel-ranking path that RFC 0019
/// §5 assigns to the engine ("the engine owns the UHD that ranks it"), and the adapter factory
/// on the live path knew only tree_data, table and native -- so this adapter existed and was
/// unreachable. Header-only to match its three siblings, which is what lets the same factory
/// construct it.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Custom library adapter for compiled scorers (RFC 0019 §7.2).
///
/// dlopen's a `.so` shipped with the engine and calls a C ABI score function:
///
///     extern "C" double <symbol>(const double* features, size_t num_features);
///
/// This mirrors RFC 0017's native-predicate pattern: the engine ships a `.so` alongside its
/// descriptor set and the provider dlopen's it rather than linking it statically. The
/// motivating case is a Treelite-generated `.so` from a GBDT the in-tree walker cannot read.
class CustomLibraryAdapter : public IUhdAdapter
{
public:
    /// @brief Loads a custom library scorer.
    ///
    /// @param libraryPath          Absolute path to the shared object.
    /// @param symbolName           C ABI scorer function name.
    /// @param numFeatures          Expected feature-row length.
    /// @param expectedFeaturesHash SHA-256 of the feature signature.
    /// @return Adapter on success, nullptr on any load failure.
    ///
    /// Returns nullptr rather than throwing: a descriptor set is drop-in data from a
    /// potentially third-party author, and RFC 0019 §5 step 7 requires a malformed one to
    /// degrade to static_order rather than fail the request.
    static std::unique_ptr<CustomLibraryAdapter> load(const std::string& libraryPath,
                                                      const std::string& symbolName,
                                                      size_t numFeatures,
                                                      const std::string& expectedFeaturesHash);

    ~CustomLibraryAdapter() override
    {
        if(_libHandle != nullptr)
        {
            if(dlclose(_libHandle) != 0)
            {
                const char* err = dlerror();
                HIPDNN_SDK_LOG_WARN("CustomLibraryAdapter: dlclose failed for "
                                    << _libraryPath << ": " << (err != nullptr ? err : "unknown"));
            }
        }
    }

    /// Non-copyable and non-movable: the handle is owned and dlclose'd exactly once.
    CustomLibraryAdapter(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter& operator=(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter(CustomLibraryAdapter&&) = delete;
    CustomLibraryAdapter& operator=(CustomLibraryAdapter&&) = delete;

    double score(const std::vector<double>& features) const override
    {
        if(features.size() != _numFeatures)
        {
            // Throws rather than scoring a short row: the callee reads num_features entries
            // through a raw pointer, so a mismatch is an out-of-bounds read inside code this
            // process does not own.
            std::ostringstream message;
            message << "CustomLibraryAdapter: feature count mismatch. Expected " << _numFeatures
                    << ", got " << features.size();
            throw std::invalid_argument(message.str());
        }

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - C ABI function pointer
        auto* scorer = reinterpret_cast<ScorerFunc>(_scorerFunc);
        return scorer(features.data(), features.size());
    }

    UhdAdapterType type() const override
    {
        return UhdAdapterType::CUSTOM_LIBRARY;
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
    /// C ABI scorer signature, identical to UhdScoreFn so one implementation can be reached
    /// either way -- registered in-process for `native`, or exported from a `.so` for this.
    using ScorerFunc = double (*)(const double*, size_t);

    CustomLibraryAdapter(void* libHandle,
                         void* scorerFunc,
                         size_t numFeatures,
                         std::string featuresHash,
                         std::string libraryPath)
        : _libHandle(libHandle)
        , _scorerFunc(scorerFunc)
        , _numFeatures(numFeatures)
        , _featuresHash(std::move(featuresHash))
        , _libraryPath(std::move(libraryPath))
    {
    }

    void* _libHandle;  ///< dlopen handle, opaque on both POSIX and Windows
    void* _scorerFunc; ///< function pointer, cast before calling
    size_t _numFeatures;
    std::string _featuresHash;
    std::string _libraryPath; ///< kept for diagnostics only
};

inline std::unique_ptr<CustomLibraryAdapter>
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

    // RTLD_NOW so a missing symbol surfaces here rather than at the first score() call, and
    // RTLD_LOCAL so a third-party scorer cannot export names into the process's global scope.
    void* libHandle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(libHandle == nullptr)
    {
        const char* err = dlerror();
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: dlopen failed for "
                             << libraryPath << ": " << (err != nullptr ? err : "unknown"));
        return nullptr;
    }

    dlerror(); // POSIX: clear before dlsym so the error check below is about this lookup

    void* symbol = dlsym(libHandle, symbolName.c_str());
    const char* err = dlerror();
    if(err != nullptr || symbol == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: dlsym failed for symbol '"
                             << symbolName << "' in " << libraryPath << ": "
                             << (err != nullptr ? err : "symbol not found"));
        dlclose(libHandle);
        return nullptr;
    }

    HIPDNN_SDK_LOG_INFO("CustomLibraryAdapter: loaded " << libraryPath << " symbol '" << symbolName
                                                        << "' (features=" << numFeatures << ")");

    return std::unique_ptr<CustomLibraryAdapter>(
        new CustomLibraryAdapter(libHandle, symbol, numFeatures, expectedFeaturesHash,
                                 libraryPath));
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

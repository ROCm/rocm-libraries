// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "IUhdAdapter.hpp"
#include "plugin/SharedLibrary.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Custom library adapter for compiled scorers (RFC 0019 §7.2).
///
/// The CUSTOM_LIBRARY adapter is an escape hatch for models the in-tree walker
/// doesn't cover. It dlopen's a .so shipped with the engine and calls a C ABI
/// score function:
///
///     extern "C" double <symbol>(const double* features, size_t num_features);
///
/// The .so is loaded lazily on first score() call and cached for the process.
/// This mirrors RFC 0017's native-predicate pattern: the engine ships a .so
/// alongside its descriptor set, and the provider dlopen's it without linking
/// it statically.
///
/// Example use case: Treelite-generated .so from a GBDT model.
class CustomLibraryAdapter : public IUhdAdapter
{
public:
    /// @brief Load a custom library scorer from a .so file.
    ///
    /// @param libraryPath Absolute path to the .so (or .dll on Windows).
    /// @param symbolName Name of the C ABI scorer function (e.g., "my_scorer").
    /// @param numFeatures Expected feature count.
    /// @param expectedFeaturesHash SHA-256 hash of the feature signature.
    /// @return Adapter on success, nullptr on load failure.
    ///
    /// The scorer function signature must be:
    ///     extern "C" double <symbolName>(const double* features, size_t num_features);
    static std::unique_ptr<CustomLibraryAdapter> load(const std::string& libraryPath,
                                                       const std::string& symbolName,
                                                       size_t numFeatures,
                                                       const std::string& expectedFeaturesHash);

    ~CustomLibraryAdapter() override;

    // Disable copy/move - the shared library handle is not copyable
    CustomLibraryAdapter(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter& operator=(const CustomLibraryAdapter&) = delete;
    CustomLibraryAdapter(CustomLibraryAdapter&&) = delete;
    CustomLibraryAdapter& operator=(CustomLibraryAdapter&&) = delete;

    double score(const std::vector<double>& features) const override;
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
    CustomLibraryAdapter(plugin::SharedLibrary library,
                         void* scorerFunc,
                         size_t numFeatures,
                         std::string featuresHash,
                         std::string libraryPath);

    /// Owns the loaded module. Unloading is its destructor's job, which is why this
    /// class declares no unload of its own.
    plugin::SharedLibrary _library;
    void* _scorerFunc; // Function pointer (opaque - cast before calling)
    size_t _numFeatures;
    std::string _featuresHash;
    std::string _libraryPath; // For error messages
};

} // namespace hipdnn_backend::heuristics::uhd

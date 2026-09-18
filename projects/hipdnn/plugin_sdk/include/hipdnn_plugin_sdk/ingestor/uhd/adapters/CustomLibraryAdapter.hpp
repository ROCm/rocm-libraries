// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/IUhdAdapter.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define HIPDNN_CUSTOM_LIBRARY_DEFINED_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#define HIPDNN_CUSTOM_LIBRARY_DEFINED_NOMINMAX
#endif
#include <windows.h>
#ifdef HIPDNN_CUSTOM_LIBRARY_DEFINED_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef HIPDNN_CUSTOM_LIBRARY_DEFINED_LEAN_AND_MEAN
#endif
#ifdef HIPDNN_CUSTOM_LIBRARY_DEFINED_NOMINMAX
#undef NOMINMAX
#undef HIPDNN_CUSTOM_LIBRARY_DEFINED_NOMINMAX
#endif
#else
#include <dlfcn.h> // POSIX dlopen / dlsym / dlclose
#endif

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

namespace detail
{

/// The three dynamic-loading calls this adapter needs, on both platforms it builds for.
///
/// A shim rather than `#ifdef`s at the call sites: the POSIX error protocol (clear
/// `dlerror()`, call, read it back) and the Windows one (`GetLastError()`) do not
/// interleave, so spelling them inline three times invites getting one of them subtly
/// wrong. The handle stays `void*` in the member, which both platforms can carry.

inline void* sharedLibraryOpen(const char* path)
{
#ifdef _WIN32
    return static_cast<void*>(::LoadLibraryA(path));
#else
    // RTLD_NOW so a missing symbol surfaces at load rather than at the first score() call,
    // and RTLD_LOCAL so a third-party scorer cannot export names into the global scope.
    return ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

/// Clears any pending error, so the caller's error check is about this lookup only.
inline void* sharedLibrarySymbol(void* handle, const char* name)
{
#ifdef _WIN32
    ::SetLastError(0);
    return reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    ::dlerror();
    return ::dlsym(handle, name);
#endif
}

/// True when the handle was released.
inline bool sharedLibraryClose(void* handle)
{
#ifdef _WIN32
    return ::FreeLibrary(static_cast<HMODULE>(handle)) != 0;
#else
    return ::dlclose(handle) == 0;
#endif
}

/// The last failure, or empty when the platform reports none. Empty is not "succeeded":
/// POSIX only guarantees a message after a call that failed.
inline std::string sharedLibraryError()
{
#ifdef _WIN32
    const auto code = ::GetLastError();
    if(code == 0)
    {
        return {};
    }
    std::ostringstream text;
    text << "system error " << code;
    return text.str();
#else
    const char* err = ::dlerror();
    return err != nullptr ? std::string(err) : std::string();
#endif
}

} // namespace detail

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
            if(!detail::sharedLibraryClose(_libHandle))
            {
                const auto err = detail::sharedLibraryError();
                HIPDNN_SDK_LOG_WARN("CustomLibraryAdapter: unload failed for "
                                    << _libraryPath << ": " << (err.empty() ? "unknown" : err));
            }
        }
    }

    /// Non-copyable and non-movable: the handle is owned and unloaded exactly once.
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

    void* libHandle = detail::sharedLibraryOpen(libraryPath.c_str());
    if(libHandle == nullptr)
    {
        const auto err = detail::sharedLibraryError();
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: load failed for "
                             << libraryPath << ": " << (err.empty() ? "unknown" : err));
        return nullptr;
    }

    void* symbol = detail::sharedLibrarySymbol(libHandle, symbolName.c_str());
    if(symbol == nullptr)
    {
        // Only consulted once the symbol is known missing: a null result is the failure,
        // and on POSIX a non-null symbol may legitimately leave a stale message behind.
        const auto err = detail::sharedLibraryError();
        HIPDNN_SDK_LOG_ERROR("CustomLibraryAdapter: symbol lookup failed for '"
                             << symbolName << "' in " << libraryPath << ": "
                             << (err.empty() ? "symbol not found" : err));
        detail::sharedLibraryClose(libHandle);
        return nullptr;
    }

    HIPDNN_SDK_LOG_INFO("CustomLibraryAdapter: loaded " << libraryPath << " symbol '" << symbolName
                                                        << "' (features=" << numFeatures << ")");

    return std::unique_ptr<CustomLibraryAdapter>(
        new CustomLibraryAdapter(libHandle, symbol, numFeatures, expectedFeaturesHash,
                                 libraryPath));
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

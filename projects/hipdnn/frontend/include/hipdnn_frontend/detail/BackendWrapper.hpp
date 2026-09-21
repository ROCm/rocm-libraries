// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>
#include <hipdnn_frontend/detail/IncompatibleBackend.hpp>
#include <hipdnn_frontend/version.h>

#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
#include <hipdnn_frontend/detail/HipdnnDirectBackendWrapper.hpp>
#else
#include <atomic>

#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>
#endif

namespace hipdnn_frontend::detail
{

// Parses the backend version string and checks it against the frontend's major
// version. Returns the parsed version on success, or std::nullopt (after
// logging) if the version is missing, malformed, or incompatible. Shared by the
// direct-link and runtime-load factories.
inline std::optional<hipdnn_data_sdk::utilities::Version> checkBackendVersion(const char* version)
{
    using namespace hipdnn_data_sdk::utilities;

    if(version == nullptr)
    {
        HIPDNN_FE_LOG_ERROR("Error parsing backend version: version is nullptr");
        return std::nullopt;
    }

    Version backendVersion;
    try
    {
        backendVersion = Version{std::string{version}};
    }
    catch(const std::invalid_argument& error)
    {
        HIPDNN_FE_LOG_ERROR("Error parsing backend version: " + std::string{error.what()});
        return std::nullopt;
    }

    if(HIPDNN_FRONTEND_VERSION_MAJOR != backendVersion.major)
    {
        HIPDNN_FE_LOG_ERROR("Backend major version (" + std::to_string(backendVersion.major)
                            + ") does not match frontend major version ("
                            + std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + ")");
        return std::nullopt;
    }

    return backendVersion;
}

#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

// Attempts to create a direct-link backend interface, falling back to
// IncompatibleBackend if it fails to satisfy requirements. version is taken as
// an argument to facilitate easier testing.
inline std::shared_ptr<IHipdnnBackend> tryToUseBackendInterface(const char* version)
{
    auto backendVersion = checkBackendVersion(version);
    if(!backendVersion)
    {
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    return std::make_shared<HipdnnDirectBackendWrapper>(*backendVersion);
}

#else

// Attempts to create a runtime-load backend interface. Loads the backend
// library, resolves the version string via dlsym, validates it, and falls back
// to IncompatibleBackend on any failure.
inline std::shared_ptr<IHipdnnBackend> tryToUseDynamicBackendInterface()
{
    if(backendLibraryHandle() == nullptr)
    {
        HIPDNN_FE_LOG_ERROR("Failed to load hipDNN backend library for runtime backend loading");
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    // Resolve the version string without an instance so the version can be
    // validated before the wrapper is constructed. decltype is unevaluated and
    // does not create a link-time reference to the symbol.
    static std::atomic<void*> s_versionCache{nullptr};
    auto versionFn = resolveBackendSymbol<decltype(&hipdnnVersionString_ext)>(
        s_versionCache, "hipdnnVersionString_ext");
    const char* version = versionFn != nullptr ? versionFn() : nullptr;

    auto backendVersion = checkBackendVersion(version);
    if(!backendVersion)
    {
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    return std::make_shared<HipdnnDynamicBackendWrapper>(*backendVersion);
}

#endif // HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

// Allow overriding the backend implementation by setting a custom backend instance.
inline static std::shared_ptr<IHipdnnBackend> hipdnnBackend()
{
    return IHipdnnBackend::getOrCreateInstance([] {
#ifdef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
        return tryToUseDynamicBackendInterface();
#else
        return tryToUseBackendInterface(hipdnnVersionString_ext());
#endif
    });
}

} // namespace hipdnn_frontend::detail

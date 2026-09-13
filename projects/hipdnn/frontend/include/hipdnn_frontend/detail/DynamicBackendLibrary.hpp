// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file DynamicBackendLibrary.hpp
 * @brief Runtime resolution of the hipDNN backend shared library.
 *
 * With @ref HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND, entry points are resolved at
 * first use without a direct backend link dependency. Resolvers share one handle
 * per module (executable or shared library), caching load failures too.
 *
 * Explicit paths avoid ASan resolving bare sonames against its own RUNPATH.
 * See @ref hipdnn_frontend::detail::resolveBackendLibrary for the search order.
 *
 * Diagnostics go directly to stderr: logging would re-enter this loader while
 * resolving the logging callback.
 */

#pragma once

#include <atomic>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <hip/hip_version.h>

#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_frontend::detail
{

HIPDNN_HIDDEN inline hipdnn_data_sdk::utilities::SharedLibraryHandle backendLibraryHandle();

/// Process-wide directory override, joined with the platform's backend filename.
/// Empty or relative values are reported on stderr and ignored.
constexpr const char* BACKEND_LIBRARY_PATH_ENV = "HIPDNN_BACKEND_LIBRARY_PATH";

/// Production inputs come from @ref backendResolutionInputs(); tests use synthetic layouts.
struct BackendResolutionInputs
{
    /// See @ref resolveBackendLibrary() for secure-execution exclusions.
    bool secureExecution = false;

    /// Engaged even for an invalid request; disengaged when no override was requested.
    std::optional<std::filesystem::path> overrideDirectory;

    std::string overrideSource;

    /// Calling module's directory (executable or shared library); empty when unknown.
    std::filesystem::path selfDirectory;

    /// Directory the HIP runtime is loaded from; empty when unknown.
    std::filesystem::path hipAnchorDirectory;
};

struct BackendLibraryResolution
{
    /// Open handle to the backend, or `nullptr` if no candidate could be loaded.
    hipdnn_data_sdk::utilities::SharedLibraryHandle handle = nullptr;

    /// The candidate @ref handle was opened from; empty when unresolved.
    std::filesystem::path path;

    /// One indented line per candidate that was not used, and why.
    std::string diagnostics;
};

struct BackendLibraryOverrideState
{
    std::mutex mutex;
    std::optional<std::filesystem::path> directory;
    bool resolutionStarted = false;
};

/// @ref HIPDNN_HIDDEN keeps state local to each executable or shared library.
HIPDNN_HIDDEN inline BackendLibraryOverrideState& backendLibraryOverrideState()
{
    static BackendLibraryOverrideState s_state;
    return s_state;
}

/// Prevents further setter calls before returning the override.
HIPDNN_HIDDEN inline std::optional<std::filesystem::path> takeBackendLibraryOverride()
{
    auto& state = backendLibraryOverrideState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.resolutionStarted = true;
    return state.directory;
}

/// Loaded HIP runtime's directory, or empty if unavailable.
/// Query the module: a symbol address may name a non-PIE executable's PLT entry.
/// RTLD_NOLOAD acquires a reference without loading; the reference must be released.
HIPDNN_HIDDEN inline std::filesystem::path hipRuntimeDirectory()
{
    namespace utilities = hipdnn_data_sdk::utilities;

#ifdef _WIN32
    const std::string versionedName
        = std::string("amdhip64_") + std::to_string(HIP_VERSION_MAJOR) + ".dll";
#else
    const std::string versionedName
        = utilities::getLibraryName("amdhip64") + "." + std::to_string(HIP_VERSION_MAJOR);
#endif
    const std::string unversionedName = utilities::getLibraryName("amdhip64");

    for(const std::string& name : {versionedName, unversionedName})
    {
        const auto handle = utilities::openLoadedLibrary(name);
        if(handle == nullptr)
        {
            continue;
        }

        std::filesystem::path origin;
        try
        {
            origin = utilities::getLoadedLibraryOrigin(handle);
        }
        catch(const std::exception&)
        {
            origin.clear();
        }

        utilities::closeLibrary(handle);
        if(!origin.empty())
        {
            return origin;
        }
    }

    return {};
}

/// Search the override, this module's directory, sibling lib/lib64 directories,
/// the HIP runtime's directory, then the bare library name. Self-relative paths
/// precede installed backends to avoid stale development-build dependencies.
/// Secure execution allows only the programmatic override and loader search:
/// computed paths bypass the loader's secure-execution restrictions.
/// Missing or unloadable candidates are recorded and skipped.
HIPDNN_HIDDEN inline BackendLibraryResolution
    resolveBackendLibrary(const BackendResolutionInputs& inputs)
{
    namespace utilities = hipdnn_data_sdk::utilities;

    const std::string fileName = utilities::getLibraryName("hipdnn_backend");

    struct Candidate
    {
        std::filesystem::path path;
        /// Bare names bypass the filesystem check and use the loader's search.
        bool mustExist = true;
    };

    std::vector<Candidate> candidates;
    const auto addDirectory = [&candidates, &fileName](const std::filesystem::path& directory) {
        if(directory.empty())
        {
            return;
        }
        const std::filesystem::path candidate = directory / fileName;
        for(const Candidate& existing : candidates)
        {
            if(utilities::pathCompEq(existing.path, candidate))
            {
                return;
            }
        }
        candidates.push_back({candidate, true});
    };

    if(inputs.overrideDirectory.has_value())
    {
        if(inputs.overrideDirectory->empty() || !inputs.overrideDirectory->is_absolute())
        {
            std::fprintf(stderr,
                         "hipDNN: ignoring %s: expected a non-empty absolute directory, got "
                         "\"%s\"\n",
                         inputs.overrideSource.c_str(),
                         inputs.overrideDirectory->u8string().c_str());
        }
        else
        {
            addDirectory(*inputs.overrideDirectory);
        }
    }

    if(!inputs.secureExecution)
    {
        addDirectory(inputs.selfDirectory);
        if(!inputs.selfDirectory.empty())
        {
            // Support both GNUInstallDirs library layouts.
            const std::filesystem::path parent = inputs.selfDirectory.parent_path();
            addDirectory(parent / "lib");
            addDirectory(parent / "lib64");
        }
        addDirectory(inputs.hipAnchorDirectory);
    }

    candidates.push_back({std::filesystem::path(fileName), false});

    BackendLibraryResolution resolution;
    for(const Candidate& candidate : candidates)
    {
        if(candidate.mustExist)
        {
            std::error_code failed;
            if(!std::filesystem::exists(candidate.path, failed) || failed)
            {
                resolution.diagnostics += "\n  " + candidate.path.u8string() + ": not present";
                continue;
            }
        }

        try
        {
            resolution.handle = utilities::openLibrary(candidate.path);
            resolution.path = candidate.path;
            return resolution;
        }
        catch(const std::exception& e)
        {
            resolution.diagnostics += "\n  " + std::string(e.what());
        }
        catch(...)
        {
            resolution.diagnostics += "\n  " + candidate.path.u8string() + ": unknown error";
        }
    }

    if(inputs.secureExecution)
    {
        resolution.diagnostics
            += "\n  (secure execution: only the loader's own search was consulted)";
    }

    return resolution;
}

/// Production values for @ref resolveBackendLibrary().
HIPDNN_HIDDEN inline BackendResolutionInputs backendResolutionInputs()
{
    namespace utilities = hipdnn_data_sdk::utilities;

    BackendResolutionInputs inputs;
    inputs.secureExecution = utilities::isSecureExecution();

    // Close the setter before searching, regardless of which candidate succeeds.
    inputs.overrideDirectory = takeBackendLibraryOverride();
    inputs.overrideSource = "setBackendLibraryPath()";

    if(inputs.secureExecution)
    {
        return inputs;
    }

    if(!inputs.overrideDirectory.has_value())
    {
        // Distinguish an unset variable from an explicitly empty, invalid override.
#ifdef _WIN32
        constexpr const wchar_t* UNSET = L"\x01unset";
        const std::wstring value = utilities::getEnvW(L"HIPDNN_BACKEND_LIBRARY_PATH", UNSET);
#else
        constexpr const char* UNSET = "\x01unset";
        const std::string value = utilities::getSecureEnv(BACKEND_LIBRARY_PATH_ENV, UNSET);
#endif
        if(value != UNSET)
        {
            inputs.overrideDirectory = std::filesystem::path(value);
            inputs.overrideSource = BACKEND_LIBRARY_PATH_ENV;
        }
    }

    try
    {
        inputs.selfDirectory = utilities::getLoadedLibraryDirectoryForAddress(
            reinterpret_cast<const void*>(&backendLibraryHandle));
    }
    catch(const std::exception&)
    {
        inputs.selfDirectory.clear();
    }

    inputs.hipAnchorDirectory = hipRuntimeDirectory();

    return inputs;
}

/// Resolves on first call and caches success or failure.
/// HIPDNN_HIDDEN isolates each executable or shared library's resolution.
HIPDNN_HIDDEN inline const BackendLibraryResolution& backendLibraryResolution()
{
    static BackendLibraryResolution s_resolution;
    static std::once_flag s_once;

    std::call_once(s_once, [] {
        try
        {
            s_resolution = resolveBackendLibrary(backendResolutionInputs());
        }
        catch(const std::exception& e)
        {
            s_resolution.handle = nullptr;
            s_resolution.path.clear();
            s_resolution.diagnostics = "\n  " + std::string(e.what());
        }
        catch(...)
        {
            s_resolution.handle = nullptr;
            s_resolution.path.clear();
            s_resolution.diagnostics = "\n  unknown error";
        }

        if(s_resolution.handle == nullptr)
        {
            std::fprintf(stderr,
                         "hipDNN: failed to load backend library; tried:%s\n",
                         s_resolution.diagnostics.c_str());
        }
    });

    return s_resolution;
}

/// Returns the lazily resolved backend handle, or nullptr if loading failed.
HIPDNN_HIDDEN inline hipdnn_data_sdk::utilities::SharedLibraryHandle backendLibraryHandle()
{
    return backendLibraryResolution().handle;
}

/// Triggers resolution on first call and returns the selected path.
/// Empty on failure; a bare library name identifies loader-search fallback.
HIPDNN_HIDDEN inline std::filesystem::path resolveBackendLibraryPath()
{
    return backendLibraryResolution().path;
}

/// Resolves a backend symbol; returns nullptr if loading or lookup fails. Never logs.
HIPDNN_HIDDEN inline void* resolveSymbol(const char* symbolName)
{
    const auto handle = backendLibraryHandle();
    if(handle == nullptr)
    {
        return nullptr;
    }
    return hipdnn_data_sdk::utilities::getSymbol(handle, symbolName);
}

/**
 * @brief Resolve a backend entry point on first use and cache the result.
 *
 * @tparam Fn         Function-pointer type of the entry point.
 * @param cache       Per-entry-point cache (start at `nullptr`).
 * @param symbolName  C symbol name to resolve.
 * @return The resolved function pointer, or `nullptr` if it could not be found.
 */
template <typename Fn>
HIPDNN_HIDDEN inline Fn resolveBackendSymbol(std::atomic<void*>& cache, const char* symbolName)
{
    void* resolved = cache.load(std::memory_order_acquire);
    if(resolved == nullptr)
    {
        resolved = resolveSymbol(symbolName);
        if(resolved != nullptr)
        {
            cache.store(resolved, std::memory_order_release);
        }
    }
    return reinterpret_cast<Fn>(resolved);
}

} // namespace hipdnn_frontend::detail

namespace hipdnn_frontend
{

/**
 * @brief Point backend resolution at @p directory, ahead of every other location.
 *
 * Applies only to the calling module (executable or shared library), not other
 * modules' frontend instances. For process-wide injection, use
 * @ref hipdnn_frontend::detail::BACKEND_LIBRARY_PATH_ENV; this setter takes
 * precedence within its module.
 *
 * @param directory Directory joined with the platform's backend filename.
 *     Empty or relative values are reported on stderr and ignored at resolution.
 * @return `true` if stored; `false` without changes once resolution has started,
 *     including after a cached failure.
 */
HIPDNN_HIDDEN inline bool setBackendLibraryPath(const std::filesystem::path& directory)
{
    auto& state = detail::backendLibraryOverrideState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if(state.resolutionStarted)
    {
        return false;
    }
    state.directory = directory;
    return true;
}

} // namespace hipdnn_frontend

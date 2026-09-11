// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file DynamicBackendLibrary.hpp
 * @brief Runtime resolution of the hipDNN backend shared library.
 *
 * When the frontend is built in runtime-load mode
 * (@ref HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND), backend entry points are resolved
 * at first use via the cross-platform loader in @c hipdnn_data_sdk::utilities
 * (`dlopen`/`dlsym` on Linux, `LoadLibrary`/`GetProcAddress` on Windows) instead
 * of being linked directly. This keeps a header-only consumer from inheriting a
 * hard dependency on `libhipdnn_backend.so`.
 *
 * The library is located by a path this code computes, not by a search the loader
 * may or may not perform on its behalf: a sanitizer runtime that intercepts
 * `dlopen` resolves a bare soname against *its own* RUNPATH rather than the
 * caller's, so a bare-name load fails even when the backend sits exactly where the
 * consumer's RUNPATH points. @ref hipdnn_frontend::detail::resolveBackendLibrary
 * documents the search order.
 *
 * The library handle is opened exactly once and shared by every entry-point
 * resolver. These helpers deliberately never emit log messages: the logging
 * callback itself is resolved through here during frontend logging
 * initialization, so logging from this layer would re-enter the loader. The
 * failure and warning paths below therefore write to stderr directly.
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

/// Directory holding the backend shared library, as a process-wide override. Joined with
/// the platform's own backend filename, so it selects a location and never an arbitrary
/// object. Must be non-empty and absolute; any other value is reported on stderr and
/// ignored.
constexpr const char* BACKEND_LIBRARY_PATH_ENV = "HIPDNN_BACKEND_LIBRARY_PATH";

/// The inputs backend resolution consults, gathered up front so the search order itself
/// is a pure function of them. Production values come from
/// @ref backendResolutionInputs(); tests substitute a synthetic layout.
struct BackendResolutionInputs
{
    /// Whether the process runs with privileges its invoker does not have. True disables
    /// every caller-influenced tier -- see @ref resolveBackendLibrary().
    bool secureExecution = false;

    /// Tier 0. Engaged means an override was *requested*, including a request whose value
    /// is invalid; disengaged means none was.
    std::optional<std::filesystem::path> overrideDirectory;

    /// What asked for @ref overrideDirectory, for the rejection message.
    std::string overrideSource;

    /// Directory this shared object is loaded from; tiers 1 and 2. Empty when unknown.
    std::filesystem::path selfDirectory;

    /// Directory the HIP runtime is loaded from; tier 3. Empty when unknown.
    std::filesystem::path hipAnchorDirectory;
};

/// Outcome of one resolution attempt over a candidate list.
struct BackendLibraryResolution
{
    /// Open handle to the backend, or `nullptr` if no candidate could be loaded.
    hipdnn_data_sdk::utilities::SharedLibraryHandle handle = nullptr;

    /// The candidate @ref handle was opened from; empty when unresolved.
    std::filesystem::path path;

    /// One indented line per candidate that was not used, and why.
    std::string diagnostics;
};

/// Mutable tier-0 state, per shared object.
struct BackendLibraryOverrideState
{
    std::mutex mutex;
    std::optional<std::filesystem::path> directory;
    bool resolutionStarted = false;
};

/// @ref HIPDNN_HIDDEN so each shared object gets its own copy -- see
/// @ref hipdnn_frontend::setBackendLibraryPath() for why that is the intended scope.
HIPDNN_HIDDEN inline BackendLibraryOverrideState& backendLibraryOverrideState()
{
    static BackendLibraryOverrideState s_state;
    return s_state;
}

/// Reads the programmatic override and latches resolution as started, so any later
/// @ref hipdnn_frontend::setBackendLibraryPath() call reports that it changed nothing.
HIPDNN_HIDDEN inline std::optional<std::filesystem::path> takeBackendLibraryOverride()
{
    auto& state = backendLibraryOverrideState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.resolutionStarted = true;
    return state.directory;
}

/// The directory the HIP runtime is loaded from, or empty if it is not loaded.
///
/// Asked of the runtime's own module handle rather than of a HIP symbol's address: a
/// non-PIE consumer resolves `&hipSymbol` to a canonical PLT entry inside itself, which
/// would answer with the executable's directory instead. `RTLD_NOLOAD` never loads the
/// runtime -- if HIP is absent, tier 3 is simply skipped -- but it does take a reference,
/// so the handle is released again here.
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

/// Locate and open the backend, most specific location first.
///
/// Ordinary process: tier 0 the explicit override, tier 1 `<this module's dir>`, tier 2
/// `<this module's dir>/../lib`, tier 3 the HIP runtime's directory, tier 4 the bare
/// library name left to the loader. Self-relative outranks the HIP anchor so a
/// development build cannot silently pick up a stale backend from an installed ROCm.
///
/// Secure execution (@ref BackendResolutionInputs::secureExecution) uses tier 0 and tier 4
/// only, and tier 0 there can only have come from @ref hipdnn_frontend::setBackendLibraryPath():
/// a call inside the process is the program speaking for itself, whereas the loader
/// restricts its own search for such a process -- `$ORIGIN` expansion, `LD_*` -- and a
/// directory this code computed is subject to no such restriction. Offering tiers 1-3 there
/// would open a filesystem door beside the closed environment one.
///
/// No candidate failure is terminal: a candidate that is absent, and equally one that
/// exists but will not load, is recorded and the search continues. A stale or corrupt
/// backend in an early tier must not deny the backend altogether.
HIPDNN_HIDDEN inline BackendLibraryResolution
    resolveBackendLibrary(const BackendResolutionInputs& inputs)
{
    namespace utilities = hipdnn_data_sdk::utilities;

    const std::string fileName = utilities::getLibraryName("hipdnn_backend");

    struct Candidate
    {
        std::filesystem::path path;
        /// False for the bare name, which has no path to test and is the loader's job.
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
                         inputs.overrideDirectory->string().c_str());
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
            // Both GNUInstallDirs layouts: an executable in bin/ has its libraries in the
            // sibling lib/ or lib64/, and this header cannot know which one the hipDNN it
            // talks to was installed with.
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
                resolution.diagnostics += "\n  " + candidate.path.string() + ": not present";
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
            resolution.diagnostics += "\n  " + candidate.path.string() + ": unknown error";
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

    // Latched even when the value goes unused, so the setter's "resolution has already
    // run" answer does not depend on which tier won.
    inputs.overrideDirectory = takeBackendLibraryOverride();
    inputs.overrideSource = "setBackendLibraryPath()";

    if(inputs.secureExecution)
    {
        // Tiers 0-3 are not consulted, so nothing below is even asked.
        return inputs;
    }

    if(!inputs.overrideDirectory.has_value())
    {
        // A sentinel default separates "set to something unusable" -- which is worth
        // reporting -- from "not set", which the environment cannot otherwise express:
        // both arrive as an empty string.
        constexpr const char* UNSET = "\x01unset";
        const std::string value = utilities::getSecureEnv(BACKEND_LIBRARY_PATH_ENV, UNSET);
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

/// The resolution performed on first call and reused thereafter, failure included.
///
/// `HIPDNN_HIDDEN` gives each shared object its own resolution, matching the per-SO
/// isolation of the backend instance accessor.
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
            // Report via stderr directly rather than the frontend logging facility: that
            // callback is itself resolved through this loader, so logging here would
            // re-enter it.
            std::fprintf(stderr,
                         "hipDNN: failed to load backend library; tried:%s\n",
                         s_resolution.diagnostics.c_str());
        }
    });

    return s_resolution;
}

/**
 * @brief Return the lazily-opened handle to the hipDNN backend shared library.
 *
 * @return The library handle, or `nullptr` if the backend could not be loaded.
 */
HIPDNN_HIDDEN inline hipdnn_data_sdk::utilities::SharedLibraryHandle backendLibraryHandle()
{
    return backendLibraryResolution().handle;
}

/**
 * @brief The path the backend shared library was loaded from.
 *
 * Triggers resolution on first call, exactly as @ref backendLibraryHandle() does.
 *
 * @return The resolved path, or an empty path if the backend could not be loaded. A
 *     resolution that fell through to the loader's own search reports the bare library
 *     name, which is distinguishable from any directory-qualified tier.
 */
HIPDNN_HIDDEN inline std::filesystem::path resolveBackendLibraryPath()
{
    return backendLibraryResolution().path;
}

/**
 * @brief Resolve a symbol from the already-opened backend library.
 *
 * Returns `nullptr` if the library could not be loaded or the symbol is not
 * found. Never logs.
 */
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
 * **Scope is the calling shared object, not the process.** The state lives beside the
 * `HIPDNN_HIDDEN` backend handle, which exists precisely so that each shared object
 * loading the frontend gets its own; a call from an executable therefore cannot pin the
 * backend for, say, `libMIOpen.so`'s copy. For process-wide injection set
 * @ref hipdnn_frontend::detail::BACKEND_LIBRARY_PATH_ENV instead -- this setter is the
 * calling module's own escape hatch, and it outranks that variable for that module.
 *
 * @param directory Directory holding the backend library; it is joined with the
 *     platform's backend filename. Must be non-empty and absolute -- any other value is
 *     reported on stderr and ignored when resolution runs.
 * @return `true` if the directory was stored, `false` if resolution has already run --
 *     including a run that failed and cached the failure -- in which case nothing changed.
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

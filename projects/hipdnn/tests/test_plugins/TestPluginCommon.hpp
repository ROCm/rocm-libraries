// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/version.h>

struct HipdnnEnginePluginHandle
{
public:
    virtual ~HipdnnEnginePluginHandle() = default;
};

struct HipdnnEnginePluginExecutionContext
{
};

inline const char* apiVersionWithoutTweak()
{
    static const std::string s_versionStr = std::to_string(HIPDNN_PLUGIN_SDK_VERSION_MAJOR) + "."
                                            + std::to_string(HIPDNN_PLUGIN_SDK_VERSION_MINOR) + "."
                                            + std::to_string(HIPDNN_PLUGIN_SDK_VERSION_PATCH);
    return s_versionStr.c_str();
}

/**
 * @brief Identifies which execute entry point a fake plugin most recently
 *        serviced. Used by RFC 0008 Phase 1 dispatch tests to assert that
 *        the host selected the override-aware path versus the original path.
 */
enum class TestPluginExecuteEntry : uint8_t
{
    NONE = 0,
    OP_GRAPH = 1, ///< hipdnnEnginePluginExecuteOpGraph was called.
    OP_GRAPH_WITH_OVERRIDES = 2, ///< hipdnnEnginePluginExecuteOpGraphWithOverrides was called.
};

/// Maximum number of overrides a single capture slot can hold. Override count
/// in real graphs is small; tests today exercise at most 2. Eight is generous.
inline constexpr std::size_t K_MAX_TEST_OVERRIDES = 8;

/// Maximum tensor rank captured per override. Matches HIPDNN's structural
/// tensor rank cap.
inline constexpr std::size_t K_MAX_TEST_OVERRIDE_RANK = 8;

/**
 * @brief Per-thread record of the most recent execute call observed by a
 *        fake test plugin (RFC 0008 Phase 1, design point D5).
 *
 * Each fake plugin maintains exactly one `thread_local` instance (defined
 * via `DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(<suffix>)`); tests inspect it
 * through the suffixed `getLastCallRecord_<suffix>()` /
 * `resetLastCallRecord_<suffix>()` C entry points (resolved via
 * `lookupLastCallRecordAccessor` / `lookupLastCallRecordResetter` since
 * the symbols live inside dlopen'd `.so` files). The record captures the
 * override selectors by *value* so tests can assert that the host
 * faithfully forwarded the variant-pack contents and so that any
 * plugin-side use of the captured pointers after host buffers go out of
 * scope is observable.
 *
 * Storage is fixed-capacity `std::array` (not `std::vector`) because the
 * record lives in a `thread_local` inside a dlopen'd plugin .so. Any
 * thread_local with a non-trivial destructor registers a callback through
 * `__cxa_thread_atexit`, and that callback holds a reference into the .so's
 * text segment until thread exit — which prevents `dlclose` from actually
 * unmapping the library and breaks the unloading-mode integration tests
 * (`IntegrationSetPluginUnloadingModeExt.*`). Trivially-destructible POD
 * storage avoids the registration entirely. The static_assert below pins
 * this contract so future changes to the struct surface the constraint at
 * compile time.
 */
struct TestPluginLastCallRecord
{
    TestPluginExecuteEntry whichEntry = TestPluginExecuteEntry::NONE;
    uint32_t numOverrides = 0;
    std::array<int64_t, K_MAX_TEST_OVERRIDES> capturedUniqueIds{};
    std::array<uint32_t, K_MAX_TEST_OVERRIDES> capturedLengths{};
    std::array<std::array<int64_t, K_MAX_TEST_OVERRIDE_RANK>, K_MAX_TEST_OVERRIDES>
        capturedShapes{};
    std::array<std::array<int64_t, K_MAX_TEST_OVERRIDE_RANK>, K_MAX_TEST_OVERRIDES>
        capturedStrides{};
};

static_assert(std::is_trivially_destructible_v<TestPluginLastCallRecord>,
              "TestPluginLastCallRecord must be trivially destructible. A non-trivial dtor "
              "would register __cxa_thread_atexit callbacks for the per-plugin thread_local "
              "instance, pinning the plugin .so under glibc and breaking "
              "IntegrationSetPluginUnloadingModeExt.* unload checks.");

/// Macro emitted by exactly one TU per fake plugin to define the
/// thread-local backing storage and the suffixed C-API observation entry
/// points. The `suffix` token is concatenated onto every emitted symbol so
/// each fake plugin's `.so` exports a unique pair of accessors and the
/// shared `TestPluginCommon.hpp` code does not generate symbol collisions
/// when several plugin TUs are linked through the same address space.
#define DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(suffix)                            \
    namespace                                                                   \
    {                                                                           \
    thread_local TestPluginLastCallRecord s_record_##suffix; /* NOLINT */       \
    }                                                                           \
    /* NOLINTBEGIN(readability-identifier-naming) suffixed C symbols */         \
    /* Internal accessor used by the plugin's executeGraph* overrides. */       \
    static inline TestPluginLastCallRecord& testPluginLastCallRecord_##suffix() \
    {                                                                           \
        return s_record_##suffix;                                               \
    }                                                                           \
    extern "C" const TestPluginLastCallRecord* getLastCallRecord_##suffix()     \
    {                                                                           \
        return &s_record_##suffix;                                              \
    }                                                                           \
    extern "C" void resetLastCallRecord_##suffix()                              \
    {                                                                           \
        s_record_##suffix = TestPluginLastCallRecord{};                         \
    }                                                                           \
    /* NOLINTEND(readability-identifier-naming) */

// Forward declarations of the per-plugin observation entry points emitted by
// the four RFC 0008 Phase 1 fake plugins. Tests that load these plugins call
// the suffixed accessor matching the plugin under test. The symbols are
// defined inside each plugin's `.so` (not linked into the test binary), so
// callers must resolve them via `dlsym` against the plugin's own dlopen
// handle — see `lookupLastCallRecordAccessor` / `lookupLastCallRecordResetter`
// below.
// NOLINTBEGIN(readability-identifier-naming) - C symbol convention requires
// the underscore-suffixed form so dlsym() lookups match the per-plugin
// suffix passed to DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE.
extern "C" const TestPluginLastCallRecord* getLastCallRecord_OverrideImplementing();
extern "C" void resetLastCallRecord_OverrideImplementing();
extern "C" const TestPluginLastCallRecord* getLastCallRecord_OverrideOmitting();
extern "C" void resetLastCallRecord_OverrideOmitting();
extern "C" const TestPluginLastCallRecord* getLastCallRecord_VersionLiar();
extern "C" void resetLastCallRecord_VersionLiar();
extern "C" const TestPluginLastCallRecord* getLastCallRecord_SecondOverride();
extern "C" void resetLastCallRecord_SecondOverride();
// NOLINTEND(readability-identifier-naming)

/// Resolve a fake plugin's `getLastCallRecord_<suffix>` /
/// `resetLastCallRecord_<suffix>` entry points through the dynamic loader.
///
/// The host engine framework loads plugin `.so` files with
/// `RTLD_NOW | RTLD_LOCAL` (see `backend/src/PlatformUtils.linux.cpp`), so
/// the symbols are NOT visible via `dlsym(RTLD_DEFAULT, ...)`. To reach
/// them tests must `dlopen` the plugin path themselves — this just bumps
/// the existing handle's refcount when the host has already loaded the
/// library — and pass the resulting handle to `dlsym`. The handle is
/// intentionally leaked: it stays valid for the lifetime of the test
/// process, mirroring `TestPluginKnobRecorder`'s pattern.
using TestPluginGetLastCallRecordFn = const TestPluginLastCallRecord* (*)();
using TestPluginResetLastCallRecordFn = void (*)();

namespace test_plugin_internal
{

/// Resolves a plugin path that may be relative against the directory
/// containing the loaded `libhipdnn_backend.so`. Mirrors the resolution
/// performed by `hipdnn_backend::plugin::SharedLibrary::load` (see
/// `backend/src/plugin/SharedLibrary.cpp`), which rebases relative paths
/// against `getCurrentModuleDirectory()` of the backend module. Anchoring
/// against the backend rather than the test executable matches how the
/// host actually loaded the plugin .so via `hipdnnSetEnginePluginPaths_ext`,
/// so the dlopen here returns a handle to the same library instance and
/// merely bumps its refcount.
///
/// The backend's `getCurrentModuleDirectory()` symbol is hidden, so this
/// helper recovers the backend's location without linking to it: it
/// resolves the address of a known exported backend C-API symbol via
/// `dlsym(RTLD_DEFAULT, ...)` and then asks `dladdr` for the containing
/// shared object's filename. Falls back to the unmodified path if the
/// backend cannot be located (e.g., statically linked test build), so the
/// caller still observes the original `dlopen` failure mode.
inline std::filesystem::path resolvePluginPathRelativeToBackend(const std::string& pluginPath)
{
    std::filesystem::path requestedPath{pluginPath};
    if(requestedPath.is_absolute())
    {
        return requestedPath;
    }

    void* backendSymbol = dlsym(RTLD_DEFAULT, "hipdnnCreate");
    if(backendSymbol == nullptr)
    {
        return requestedPath;
    }

    Dl_info info{};
    if(dladdr(backendSymbol, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0')
    {
        return requestedPath;
    }

    const auto backendDir = std::filesystem::path(info.dli_fname).parent_path();
    return std::filesystem::weakly_canonical(backendDir / requestedPath);
}

/// Returns (and caches) the handle for the plugin .so at `pluginPath`.
/// Returns `nullptr` if dlopen fails (e.g., the plugin was never loaded
/// and the path is invalid in the current test layout). Relative paths
/// are rebased against the backend module's directory (matching the
/// host's plugin loader) so the dlopen succeeds regardless of the test
/// process's current working directory — which under ctest typically
/// differs from the directory containing `test_plugins/`.
inline void* openPluginForSymbolLookup(const std::string& pluginPath)
{
    static thread_local std::unordered_map<std::string, void*> s_handles;
    auto iter = s_handles.find(pluginPath);
    if(iter != s_handles.end())
    {
        return iter->second;
    }
    const auto resolvedPath = resolvePluginPathRelativeToBackend(pluginPath);
    void* handle = dlopen(resolvedPath.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    s_handles.emplace(pluginPath, handle);
    return handle;
}

} // namespace test_plugin_internal

inline TestPluginGetLastCallRecordFn lookupLastCallRecordAccessor(const std::string& pluginPath,
                                                                  const std::string& suffix)
{
    void* handle = test_plugin_internal::openPluginForSymbolLookup(pluginPath);
    if(handle == nullptr)
    {
        return nullptr;
    }
    const std::string symbolName = "getLastCallRecord_" + suffix;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<TestPluginGetLastCallRecordFn>(dlsym(handle, symbolName.c_str()));
}

inline TestPluginResetLastCallRecordFn lookupLastCallRecordResetter(const std::string& pluginPath,
                                                                    const std::string& suffix)
{
    void* handle = test_plugin_internal::openPluginForSymbolLookup(pluginPath);
    if(handle == nullptr)
    {
        return nullptr;
    }
    const std::string symbolName = "resetLastCallRecord_" + suffix;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<TestPluginResetLastCallRecordFn>(dlsym(handle, symbolName.c_str()));
}

/// Convenience: call `resetLastCallRecord_<suffix>()` if the symbol is
/// resolvable, otherwise no-op. Tests use this in fixture SetUp to wipe
/// TLS state across all fake plugins regardless of which subset is loaded.
inline void resetLastCallRecordIfLoaded(const std::string& pluginPath, const std::string& suffix)
{
    auto* fn = lookupLastCallRecordResetter(pluginPath, suffix);
    if(fn != nullptr)
    {
        fn();
    }
}

/// Convenience: call `getLastCallRecord_<suffix>()` if the symbol is
/// resolvable, otherwise return `nullptr`. Tests use this to inspect a
/// specific plugin's TLS observation record.
inline const TestPluginLastCallRecord* getLastCallRecordIfLoaded(const std::string& pluginPath,
                                                                 const std::string& suffix)
{
    auto* fn = lookupLastCallRecordAccessor(pluginPath, suffix);
    return (fn != nullptr) ? fn() : nullptr;
}

// Base class for test plugins
class TestPluginBase
{
public:
    virtual ~TestPluginBase() = default;

    // Virtual methods to be overridden by derived classes
    virtual const char* getPluginName() const = 0;
    virtual const char* getPluginVersion() const = 0;
    virtual const char* getPluginApiVersion() const = 0;
    virtual int64_t getEngineId() const = 0;
    virtual uint32_t getNumEngines() const = 0;
    virtual uint32_t getNumApplicableEngines() const = 0;
    virtual bool supportsEngineOperations() const
    {
        return getNumApplicableEngines() > 0;
    }

    // Execute graph - derived classes override this for custom behavior
    virtual void executeGraph() const
    {
        HIPDNN_PLUGIN_LOG_INFO("executeGraph called");
    }

    /**
     * @brief Returns the calling thread's mutable last-call record for this
     *        plugin. Each fake plugin overrides this to return its own
     *        suffixed thread-local storage emitted by
     *        `DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(<suffix>)`. Plugins that
     *        do not opt into TLS observation can use the default no-op
     *        scratch instance.
     */
    virtual TestPluginLastCallRecord& lastCallRecord() const
    {
        thread_local TestPluginLastCallRecord s_unused;
        return s_unused;
    }

    /**
     * @brief Override-aware execute hook (RFC 0008 Phase 1).
     *
     * Default implementation captures every override selector into the
     * calling thread's per-plugin `TestPluginLastCallRecord` (resolved via
     * the virtual `lastCallRecord()`) so tests can assert the host
     * forwarded the variant-pack contents byte-for-byte. Plugins that want
     * bespoke behavior may override; they should still call this base so
     * that the LastCallRecord remains populated.
     */
    virtual void executeGraphWithOverrides(uint32_t numOverrides,
                                           const int64_t* overrideUniqueIds,
                                           const uint32_t* overrideLengths,
                                           const int64_t* const* overrideShapes,
                                           const int64_t* const* overrideStrides) const
    {
        HIPDNN_PLUGIN_LOG_INFO("executeGraphWithOverrides called numOverrides=" << numOverrides);

        auto& rec = lastCallRecord();
        rec = TestPluginLastCallRecord{};
        rec.whichEntry = TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES;
        rec.numOverrides = numOverrides;
        const auto cappedN = std::min(numOverrides, static_cast<uint32_t>(K_MAX_TEST_OVERRIDES));
        for(uint32_t i = 0; i < cappedN; ++i)
        {
            rec.capturedUniqueIds[i] = overrideUniqueIds[i];
            rec.capturedLengths[i] = overrideLengths[i];
            const auto cappedR
                = std::min(overrideLengths[i], static_cast<uint32_t>(K_MAX_TEST_OVERRIDE_RANK));
            for(uint32_t r = 0; r < cappedR; ++r)
            {
                rec.capturedShapes[i][r] = overrideShapes[i][r];
                rec.capturedStrides[i][r] = overrideStrides[i][r];
            }
        }
    }

    // Static instance management
    static void setInstance(std::unique_ptr<TestPluginBase> instance)
    {
        s_instance = std::move(instance);
    }

    static TestPluginBase* getInstance()
    {
        return s_instance.get();
    }

    // Common API implementations
    static hipdnnPluginStatus_t pluginGetName(const char** name)
    {
        LOG_API_ENTRY("namePtr=" << static_cast<const void*>(name));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(name);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            *name = getInstance()->getPluginName();

            LOG_API_SUCCESS(apiName, "pluginName=" << static_cast<const void*>(name));
        });
    }

    static hipdnnPluginStatus_t pluginGetVersion(const char** version)
    {
        LOG_API_ENTRY("versionPtr=" << static_cast<const void*>(version));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(version);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            *version = getInstance()->getPluginVersion();

            LOG_API_SUCCESS(apiName, "version=" << static_cast<const void*>(version));
        });
    }

    static hipdnnPluginStatus_t pluginGetApiVersion(const char** version)
    {
        LOG_API_ENTRY("versionPtr=" << static_cast<const void*>(version));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(version);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            *version = getInstance()->getPluginApiVersion();

            LOG_API_SUCCESS(apiName, "version=" << static_cast<const void*>(version));
        });
    }

    static hipdnnPluginStatus_t pluginGetType(hipdnnPluginType_t* type)
    {
        LOG_API_ENTRY("typePtr=" << static_cast<void*>(type));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(type);

            *type = HIPDNN_PLUGIN_TYPE_ENGINE;

            LOG_API_SUCCESS(apiName, "type=" << *type);
        });
    }

    static void pluginGetLastErrorString(const char** errorStr)
    {
        LOG_API_ENTRY("errorStrPtr=" << static_cast<const void*>(errorStr));

        hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(errorStr);

            *errorStr = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();

            LOG_API_SUCCESS(apiName, "errorStr=" << static_cast<const void*>(errorStr));
        });
    }

    static hipdnnPluginStatus_t pluginSetLoggingCallback(hipdnnCallback_t callback)
    {
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(callback);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            hipdnn_plugin_sdk::logging::initializeCallbackLogging(getInstance()->getPluginName(),
                                                                  callback);

            LOG_API_SUCCESS(apiName, "callback registered");
        });
    }

    static hipdnnPluginStatus_t pluginSetLogLevel(hipdnnSeverity_t level)
    {
        return hipdnn_plugin_sdk::tryCatch([&]() {
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            hipdnn_plugin_sdk::logging::setLogLevel(level);

            // Log at the level being set so tests can positively verify the call
            // and the level value for each severity
            switch(level)
            {
            case HIPDNN_SEV_INFO:
                HIPDNN_PLUGIN_LOG_INFO("TEST: pluginSetLogLevel level=" << level);
                break;
            case HIPDNN_SEV_WARN:
                HIPDNN_PLUGIN_LOG_WARN("TEST: pluginSetLogLevel level=" << level);
                break;
            case HIPDNN_SEV_ERROR: // Not used by tests
            case HIPDNN_SEV_FATAL: // Not used by tests
            case HIPDNN_SEV_OFF:
            default:
                break;
            }
        });
    }

    static hipdnnPluginStatus_t
        // NOLINTNEXTLINE(readability-non-const-parameter)
        enginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
    {
        LOG_API_ENTRY("engineIds=" << static_cast<void*>(engineIds) << ", maxEngines=" << maxEngines
                                   << ", numEngines=" << static_cast<void*>(numEngines));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            if(maxEngines != 0)
            {
                hipdnn_plugin_sdk::throwIfNull(engineIds);
            }
            hipdnn_plugin_sdk::throwIfNull(numEngines);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            *numEngines = getInstance()->getNumEngines();

            if(maxEngines >= 1 && *numEngines > 0)
            {
                assert(*numEngines == 1);
                engineIds[0] = getInstance()->getEngineId();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    static hipdnnPluginStatus_t enginePluginCreate(hipdnnEnginePluginHandle_t* handle)
    {
        LOG_API_ENTRY("handlePtr=" << static_cast<void*>(handle));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);

            *handle = new HipdnnEnginePluginHandle();

            LOG_API_SUCCESS(apiName, "createdHandle=" << static_cast<void*>(*handle));
        });
    }

    static hipdnnPluginStatus_t enginePluginDestroy(hipdnnEnginePluginHandle_t handle)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);

            delete handle;
            handle = nullptr;

            LOG_API_SUCCESS(apiName, "destroyed");
        });
    }

    static hipdnnPluginStatus_t enginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                      hipStream_t stream)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", streamId=" << static_cast<void*>(stream));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);

            LOG_API_SUCCESS(apiName, "stream set");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* opGraph,
                                           int64_t* engineIds,
                                           uint32_t maxEngines,
                                           uint32_t* numEngines)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", engineIds=" << static_cast<void*>(engineIds)
                                << ", maxEngines=" << maxEngines
                                << ", numEngines=" << static_cast<void*>(numEngines));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            if(maxEngines != 0)
            {
                hipdnn_plugin_sdk::throwIfNull(engineIds);
            }
            hipdnn_plugin_sdk::throwIfNull(numEngines);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            *numEngines = getInstance()->getNumApplicableEngines();

            if(maxEngines >= 1 && *numEngines > 0)
            {
                engineIds[0] = getInstance()->getEngineId();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    static hipdnnPluginStatus_t enginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                             int64_t engineId,
                                                             const hipdnnPluginConstData_t* opGraph,
                                                             hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", engineId=" << engineId
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", engineDetails=" << static_cast<void*>(engineDetails));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(engineDetails);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get engine details");
            }

            flatbuffers::FlatBufferBuilder builder;
            auto newEngineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
                builder, getInstance()->getEngineId());
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            auto* tempBuffer = new uint8_t[serializedDetails.size()];
            std::memcpy(tempBuffer, serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer;
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                         hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", engineDetails=" << static_cast<void*>(engineDetails));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineDetails);

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "No engine details to destroy");
            }

            hipdnn_plugin_sdk::throwIfNull(engineDetails->ptr);

            delete[] static_cast<const uint8_t*>(engineDetails->ptr);

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                     const hipdnnPluginConstData_t* engineConfig,
                                     const hipdnnPluginConstData_t* opGraph,
                                     size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", engineConfig=" << static_cast<const void*>(engineConfig)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", workspaceSize=" << static_cast<void*>(workspaceSize));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get workspace size");
            }

            *workspaceSize = 1024;

            LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", executionContext="
                                << static_cast<const void*>(executionContext)
                                << ", workspaceSize=" << static_cast<void*>(workspaceSize));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get workspace size");
            }

            *workspaceSize = 2048;

            LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* engineConfig,
                                           const hipdnnPluginConstData_t* opGraph,
                                           hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", engineConfig=" << static_cast<const void*>(engineConfig)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", executionContext=" << static_cast<void*>(executionContext));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot create execution context");
            }

            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(
                opGraph->ptr, opGraph->size);
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper
                engineConfigWrapper(engineConfig->ptr, engineConfig->size);

            *executionContext = new HipdnnEnginePluginExecutionContext();

            LOG_API_SUCCESS(apiName,
                            "createdExecutionContext=" << static_cast<void*>(*executionContext));
        });
    }

    static hipdnnPluginStatus_t
        enginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                            hipdnnEnginePluginExecutionContext_t executionContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", executionContext=" << static_cast<void*>(executionContext));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "No execution context to destroy");
            }

            delete executionContext;

            LOG_API_SUCCESS(apiName, "destroyed executionContext");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                   hipdnnEnginePluginExecutionContext_t executionContext,
                                   void* workspace,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   uint32_t numDeviceBuffers)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", executionContext=" << static_cast<void*>(executionContext)
                                << ", workspace=" << workspace
                                << ", deviceBuffers=" << static_cast<const void*>(deviceBuffers)
                                << ", numDeviceBuffers=" << numDeviceBuffers);

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(deviceBuffers);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot execute graph");
            }

            auto& rec = getInstance()->lastCallRecord();
            rec = TestPluginLastCallRecord{};
            rec.whichEntry = TestPluginExecuteEntry::OP_GRAPH;

            getInstance()->executeGraph();

            LOG_API_SUCCESS(apiName, "executed graph");
        });
    }

    /**
     * @brief Shared C-API implementation for the optional override-aware
     *        execute entry (RFC 0008 Phase 1). Plugins that opt in emit a
     *        forwarder via `REGISTER_TEST_PLUGIN_OVERRIDE_API()`; plugins
     *        that opt out simply do not emit the symbol so the host's
     *        `tryAssignSymbol` resolution treats it as unsupported.
     */
    static hipdnnPluginStatus_t enginePluginExecuteOpGraphWithOverrides(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t executionContext,
        void* workspace,
        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
        uint32_t numDeviceBuffers,
        uint32_t numOverrides,
        const int64_t* overrideUniqueIds,
        const uint32_t* overrideLengths,
        const int64_t* const* overrideShapes,
        const int64_t* const* overrideStrides)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", executionContext=" << static_cast<void*>(executionContext)
                                << ", workspace=" << workspace
                                << ", deviceBuffers=" << static_cast<const void*>(deviceBuffers)
                                << ", numDeviceBuffers=" << numDeviceBuffers
                                << ", numOverrides=" << numOverrides);

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(deviceBuffers);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(numOverrides > 0)
            {
                hipdnn_plugin_sdk::throwIfNull(overrideUniqueIds);
                hipdnn_plugin_sdk::throwIfNull(overrideLengths);
                hipdnn_plugin_sdk::throwIfNull(overrideShapes);
                hipdnn_plugin_sdk::throwIfNull(overrideStrides);
            }

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot execute graph with overrides");
            }

            getInstance()->executeGraphWithOverrides(
                numOverrides, overrideUniqueIds, overrideLengths, overrideShapes, overrideStrides);

            LOG_API_SUCCESS(apiName, "executed graph with overrides");
        });
    }

private:
    inline static std::unique_ptr<TestPluginBase> s_instance; //NOLINT
};

// Macro to register plugin API functions
#define REGISTER_TEST_PLUGIN_API()                                                                \
    extern "C" {                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                   \
    {                                                                                             \
        return TestPluginBase::pluginGetName(name);                                               \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                             \
    {                                                                                             \
        return TestPluginBase::pluginGetVersion(version);                                         \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)                          \
    {                                                                                             \
        return TestPluginBase::pluginGetApiVersion(version);                                      \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                            \
    {                                                                                             \
        return TestPluginBase::pluginGetType(type);                                               \
    }                                                                                             \
                                                                                                  \
    void hipdnnPluginGetLastErrorString(const char** errorStr)                                    \
    {                                                                                             \
        TestPluginBase::pluginGetLastErrorString(errorStr);                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                \
    {                                                                                             \
        return TestPluginBase::pluginSetLoggingCallback(callback);                                \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)                          \
    {                                                                                             \
        return TestPluginBase::pluginSetLogLevel(level);                                          \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds,                    \
                                                           uint32_t maxEngines,                   \
                                                           uint32_t* numEngines)                  \
    {                                                                                             \
        return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);    \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)             \
    {                                                                                             \
        return TestPluginBase::enginePluginCreate(handle);                                        \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)             \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroy(handle);                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,           \
                                                     hipStream_t stream)                          \
    {                                                                                             \
        return TestPluginBase::enginePluginSetStream(handle, stream);                             \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,               \
                                                 const hipdnnPluginConstData_t* opGraph,          \
                                                 int64_t* engineIds,                              \
                                                 uint32_t maxEngines,                             \
                                                 uint32_t* numEngines)                            \
    {                                                                                             \
        return TestPluginBase::enginePluginGetApplicableEngineIds(                                \
            handle, opGraph, engineIds, maxEngines, numEngines);                                  \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,                     \
                                           int64_t engineId,                                      \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           hipdnnPluginConstData_t* engineDetails)                \
    {                                                                                             \
        return TestPluginBase::enginePluginGetEngineDetails(                                      \
            handle, engineId, opGraph, engineDetails);                                            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,                 \
                                               hipdnnPluginConstData_t* engineDetails)            \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);           \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,                     \
                                           const hipdnnPluginConstData_t* engineConfig,           \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           size_t* workspaceSize)                                 \
    {                                                                                             \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                      \
            handle, engineConfig, opGraph, workspaceSize);                                        \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(                  \
        hipdnnEnginePluginHandle_t handle,                                                        \
        hipdnnEnginePluginExecutionContext_t executionContext,                                    \
        size_t* workspaceSize)                                                                    \
    {                                                                                             \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                      \
            handle, executionContext, workspaceSize);                                             \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(                                \
        hipdnnEnginePluginHandle_t handle,                                                        \
        const hipdnnPluginConstData_t* engineConfig,                                              \
        const hipdnnPluginConstData_t* opGraph,                                                   \
        hipdnnEnginePluginExecutionContext_t* executionContext)                                   \
    {                                                                                             \
        return TestPluginBase::enginePluginCreateExecutionContext(                                \
            handle, engineConfig, opGraph, executionContext);                                     \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(                               \
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext) \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);     \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,                       \
                                         hipdnnEnginePluginExecutionContext_t executionContext,   \
                                         void* workspace,                                         \
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,         \
                                         uint32_t numDeviceBuffers)                               \
    {                                                                                             \
        return TestPluginBase::enginePluginExecuteOpGraph(                                        \
            handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);                \
    }                                                                                             \
    } // extern "C"

/**
 * Companion to `REGISTER_TEST_PLUGIN_API()`: emits ONLY the optional
 * `hipdnnEnginePluginExecuteOpGraphWithOverrides` symbol (RFC 0008 Phase 1).
 * Plugins that should appear to opt out (Test #3 / #20: TestVersionLiarPlugin
 * and TestOverrideOmittingPlugin) simply do not invoke this macro, so the
 * symbol is absent from the resulting `.so` and the host's `tryAssignSymbol`
 * resolution leaves `_funcExecuteOpGraphWithOverrides` null.
 */
#define REGISTER_TEST_PLUGIN_OVERRIDE_API()                                               \
    extern "C" {                                                                          \
    hipdnnPluginStatus_t hipdnnEnginePluginExecuteOpGraphWithOverrides(                   \
        hipdnnEnginePluginHandle_t handle,                                                \
        hipdnnEnginePluginExecutionContext_t executionContext,                            \
        void* workspace,                                                                  \
        const hipdnnPluginDeviceBuffer_t* deviceBuffers,                                  \
        uint32_t numDeviceBuffers,                                                        \
        uint32_t numOverrides,                                                            \
        const int64_t* overrideUniqueIds,                                                 \
        const uint32_t* overrideLengths,                                                  \
        const int64_t* const* overrideShapes,                                             \
        const int64_t* const* overrideStrides)                                            \
    {                                                                                     \
        return TestPluginBase::enginePluginExecuteOpGraphWithOverrides(handle,            \
                                                                       executionContext,  \
                                                                       workspace,         \
                                                                       deviceBuffers,     \
                                                                       numDeviceBuffers,  \
                                                                       numOverrides,      \
                                                                       overrideUniqueIds, \
                                                                       overrideLengths,   \
                                                                       overrideShapes,    \
                                                                       overrideStrides);  \
    }                                                                                     \
    } // extern "C"

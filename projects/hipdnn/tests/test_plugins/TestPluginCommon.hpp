// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
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

/// Execute entry point most recently serviced by a fake plugin.
enum class TestPluginExecuteEntry : uint8_t
{
    NONE = 0,
    OP_GRAPH = 1, ///< hipdnnEnginePluginExecuteOpGraph was called.
    OP_GRAPH_WITH_OVERRIDES = 2, ///< hipdnnEnginePluginExecuteOpGraphWithOverrides was called.
};

inline constexpr std::size_t K_MAX_TEST_OVERRIDES = 8;

inline constexpr std::size_t K_MAX_TEST_OVERRIDE_RANK = 8;

using TestPluginSerializedContextPayload = std::array<uint8_t, 1>;

struct TestPluginFreeDeleter
{
    void operator()(const void* ptr) const
    {
        std::free(const_cast<void*>(ptr));
    }
};

using TestPluginMallocBuffer = std::unique_ptr<void, TestPluginFreeDeleter>;
using TestPluginConstMallocBuffer = std::unique_ptr<const void, TestPluginFreeDeleter>;

/// Captures the most recent execute call for fake-plugin assertions.
///
/// Kept trivially destructible because instances are `thread_local` inside
/// plugin libraries; non-trivial destructors can register thread-exit
/// callbacks that keep POSIX libraries mapped after unload.
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

#ifdef HIPDNN_TEST_PLUGIN_BUILD
#define HIPDNN_TEST_PLUGIN_EXPORT HIPDNN_PLUGIN_EXPORT
#else
#define HIPDNN_TEST_PLUGIN_EXPORT
#endif

/// Defines per-plugin test observation storage and exported accessors.
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
    extern "C" HIPDNN_TEST_PLUGIN_EXPORT const TestPluginLastCallRecord*        \
        getLastCallRecord_##suffix()                                            \
    {                                                                           \
        return &s_record_##suffix;                                              \
    }                                                                           \
    extern "C" HIPDNN_TEST_PLUGIN_EXPORT void resetLastCallRecord_##suffix()    \
    {                                                                           \
        s_record_##suffix = TestPluginLastCallRecord{};                         \
    }                                                                           \
    /* NOLINTEND(readability-identifier-naming) */

// Per-plugin observation entry points. Tests resolve these against the
// plugin's own dynamic-loader handle.
// NOLINTBEGIN(readability-identifier-naming) - C symbol convention requires
// the underscore-suffixed form so dynamic-loader lookups match the per-plugin
// suffix passed to DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE.
extern "C" HIPDNN_TEST_PLUGIN_EXPORT const TestPluginLastCallRecord*
    getLastCallRecord_OverrideImplementing();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT void resetLastCallRecord_OverrideImplementing();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT const TestPluginLastCallRecord*
    getLastCallRecord_OverrideOmitting();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT void resetLastCallRecord_OverrideOmitting();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT const TestPluginLastCallRecord*
    getLastCallRecord_VersionLiar();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT void resetLastCallRecord_VersionLiar();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT const TestPluginLastCallRecord*
    getLastCallRecord_SecondOverride();
extern "C" HIPDNN_TEST_PLUGIN_EXPORT void resetLastCallRecord_SecondOverride();
// NOLINTEND(readability-identifier-naming)

using TestPluginGetLastCallRecordFn = const TestPluginLastCallRecord* (*)();
using TestPluginResetLastCallRecordFn = void (*)();

namespace test_plugin_internal
{

/// Resolves relative plugin paths the same way the backend plugin loader does.
inline std::filesystem::path resolvePluginPathRelativeToBackend(const std::string& pluginPath)
{
    std::filesystem::path requestedPath{pluginPath};
    if(requestedPath.is_absolute())
    {
        return requestedPath;
    }

#ifdef _WIN32
    try
    {
        const auto backendDir = hipdnn_data_sdk::utilities::getLoadedLibraryDirectory(
            hipdnn_data_sdk::utilities::getLibraryName("hipdnn_backend").c_str());
        return std::filesystem::weakly_canonical(backendDir / requestedPath);
    }
    catch(const std::runtime_error&)
    {
        return requestedPath;
    }
#else
    try
    {
        const auto backendDir
            = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForSymbol("hipdnnCreate");
        return std::filesystem::weakly_canonical(backendDir / requestedPath);
    }
    catch(const std::runtime_error&)
    {
        return requestedPath;
    }
#endif
}

using TestPluginLibraryHandle = hipdnn_data_sdk::utilities::SharedLibraryHandle;

inline TestPluginLibraryHandle openPluginIfLoaded(const std::string& pluginPath)
{
    const auto resolvedPath = resolvePluginPathRelativeToBackend(pluginPath);
    return hipdnn_data_sdk::utilities::openLoadedLibrary(resolvedPath);
}

inline void* lookupPluginSymbol(TestPluginLibraryHandle handle, const std::string& symbolName)
{
    return hipdnn_data_sdk::utilities::getSymbol(handle, symbolName.c_str());
}

} // namespace test_plugin_internal

class ScopedTestPluginLibrary
{
public:
    explicit ScopedTestPluginLibrary(std::string pluginPath)
        : _pluginPath(std::move(pluginPath))
    {
        const auto resolvedPath
            = test_plugin_internal::resolvePluginPathRelativeToBackend(_pluginPath);
        _handle = hipdnn_data_sdk::utilities::openLibrary(resolvedPath);
    }

    ~ScopedTestPluginLibrary()
    {
        close();
    }

    ScopedTestPluginLibrary(const ScopedTestPluginLibrary&) = delete;
    ScopedTestPluginLibrary& operator=(const ScopedTestPluginLibrary&) = delete;

    ScopedTestPluginLibrary(ScopedTestPluginLibrary&& other) noexcept
        : _pluginPath(std::move(other._pluginPath))
        , _handle(std::exchange(other._handle, nullptr))
    {
    }

    ScopedTestPluginLibrary& operator=(ScopedTestPluginLibrary&& other) noexcept
    {
        if(this != &other)
        {
            close();
            _pluginPath = std::move(other._pluginPath);
            _handle = std::exchange(other._handle, nullptr);
        }
        return *this;
    }

    template <typename Fn>
    Fn lookup(const std::string& symbolName) const
    {
        if(_handle == nullptr)
        {
            return nullptr;
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<Fn>(test_plugin_internal::lookupPluginSymbol(_handle, symbolName));
    }

    const std::string& pluginPath() const
    {
        return _pluginPath;
    }

private:
    void close()
    {
        if(_handle != nullptr)
        {
            hipdnn_data_sdk::utilities::closeLibrary(_handle);
            _handle = nullptr;
        }
    }

    std::string _pluginPath;
    test_plugin_internal::TestPluginLibraryHandle _handle = nullptr;
};

inline TestPluginGetLastCallRecordFn
    lookupLastCallRecordAccessor(const ScopedTestPluginLibrary& pluginLibrary,
                                 const std::string& suffix)
{
    const std::string symbolName = "getLastCallRecord_" + suffix;
    return pluginLibrary.lookup<TestPluginGetLastCallRecordFn>(symbolName);
}

inline TestPluginResetLastCallRecordFn
    lookupLastCallRecordResetter(const ScopedTestPluginLibrary& pluginLibrary,
                                 const std::string& suffix)
{
    const std::string symbolName = "resetLastCallRecord_" + suffix;
    return pluginLibrary.lookup<TestPluginResetLastCallRecordFn>(symbolName);
}

inline void resetLastCallRecord(const ScopedTestPluginLibrary& pluginLibrary,
                                const std::string& suffix)
{
    auto* fn = lookupLastCallRecordResetter(pluginLibrary, suffix);
    if(fn != nullptr)
    {
        fn();
    }
}

inline const TestPluginLastCallRecord*
    getLastCallRecord(const ScopedTestPluginLibrary& pluginLibrary, const std::string& suffix)
{
    auto* fn = lookupLastCallRecordAccessor(pluginLibrary, suffix);
    return (fn != nullptr) ? fn() : nullptr;
}

/// Convenience: call `resetLastCallRecord_<suffix>()` if the symbol is
/// resolvable, otherwise no-op. Tests use this in fixture SetUp to wipe
/// TLS state across all fake plugins regardless of which subset is loaded.
inline void resetLastCallRecordIfLoaded(const std::string& pluginPath, const std::string& suffix)
{
    auto handle = test_plugin_internal::openPluginIfLoaded(pluginPath);
    if(handle == nullptr)
    {
        return;
    }

    const std::string symbolName = "resetLastCallRecord_" + suffix;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* fn = reinterpret_cast<TestPluginResetLastCallRecordFn>(
        test_plugin_internal::lookupPluginSymbol(handle, symbolName));
    if(fn != nullptr)
    {
        fn();
    }
    hipdnn_data_sdk::utilities::closeLibrary(handle);
}

/// Convenience: call `getLastCallRecord_<suffix>()` if the symbol is
/// resolvable, otherwise return `nullptr`. Tests use this to inspect a
/// specific plugin's TLS observation record.
inline const TestPluginLastCallRecord* getLastCallRecordIfLoaded(const std::string& pluginPath,
                                                                 const std::string& suffix)
{
    auto handle = test_plugin_internal::openPluginIfLoaded(pluginPath);
    if(handle == nullptr)
    {
        return nullptr;
    }

    const std::string symbolName = "getLastCallRecord_" + suffix;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* fn = reinterpret_cast<TestPluginGetLastCallRecordFn>(
        test_plugin_internal::lookupPluginSymbol(handle, symbolName));
    const TestPluginLastCallRecord* record = (fn != nullptr) ? fn() : nullptr;
    hipdnn_data_sdk::utilities::closeLibrary(handle);
    return record;
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

    /// Returns the calling thread's mutable last-call record for this plugin.
    virtual TestPluginLastCallRecord& lastCallRecord() const
    {
        thread_local TestPluginLastCallRecord s_unused;
        return s_unused;
    }

    /// Captures override-execute arguments for assertions.
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

            auto pluginHandle = std::make_unique<HipdnnEnginePluginHandle>();
            *handle = pluginHandle.release();

            LOG_API_SUCCESS(apiName, "createdHandle=" << static_cast<void*>(*handle));
        });
    }

    static hipdnnPluginStatus_t enginePluginDestroy(hipdnnEnginePluginHandle_t handle)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);

            const std::unique_ptr<HipdnnEnginePluginHandle> pluginHandle(handle);

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

            TestPluginMallocBuffer tempBuffer(std::malloc(serializedDetails.size()));
            hipdnn_plugin_sdk::throwIfNull(tempBuffer.get());
            std::memcpy(tempBuffer.get(), serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer.release();
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

            const TestPluginConstMallocBuffer engineDetailsBytes(engineDetails->ptr);
            engineDetails->ptr = nullptr;
            engineDetails->size = 0;

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

            auto context = std::make_unique<HipdnnEnginePluginExecutionContext>();
            *executionContext = context.release();

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

            const std::unique_ptr<HipdnnEnginePluginExecutionContext> context(executionContext);

            LOG_API_SUCCESS(apiName, "destroyed executionContext");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginSerializeExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext,
                                              hipdnnPluginConstData_t* serializedContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", executionContext=" << static_cast<void*>(executionContext)
                                << ", serializedContext=" << static_cast<void*>(serializedContext));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(serializedContext);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No execution context available to serialize");
            }

            auto payload = std::make_unique<TestPluginSerializedContextPayload>();
            (*payload)[0] = 0x42;

            LOG_API_SUCCESS(apiName, "serialized context");

            serializedContext->ptr = payload.release();
            serializedContext->size = sizeof(TestPluginSerializedContextPayload);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginDestroySerializedExecutionContext(hipdnnEnginePluginHandle_t handle,
                                                      hipdnnPluginConstData_t* serializedContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", serializedContext=" << static_cast<void*>(serializedContext));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(serializedContext);
            hipdnn_plugin_sdk::throwIfNull(serializedContext->ptr);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No serialized execution context to destroy");
            }

            const std::unique_ptr<const TestPluginSerializedContextPayload> payload(
                static_cast<const TestPluginSerializedContextPayload*>(serializedContext->ptr));
            serializedContext->ptr = nullptr;
            serializedContext->size = 0;

            LOG_API_SUCCESS(apiName, "destroyed serialized context");
        });
    }

    static hipdnnPluginStatus_t enginePluginCreateExecutionContextFromSerialized(
        hipdnnEnginePluginHandle_t handle,
        const hipdnnPluginConstData_t* serializedContext,
        hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", serializedContext="
                                << static_cast<const void*>(serializedContext)
                                << ", executionContext=" << static_cast<void*>(executionContext));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(serializedContext);
            hipdnn_plugin_sdk::throwIfNull(serializedContext->ptr);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot create execution context from serialized data");
            }

            const auto* payload
                = static_cast<const TestPluginSerializedContextPayload*>(serializedContext->ptr);
            if(serializedContext->size != sizeof(TestPluginSerializedContextPayload)
               || (*payload)[0] != 0x42)
            {
                const auto firstByte
                    = serializedContext->size > 0
                          ? static_cast<int>(static_cast<const uint8_t*>(serializedContext->ptr)[0])
                          : -1;
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    std::string("Serialized execution context payload is malformed for plugin ")
                        + getInstance()->getPluginName()
                        + ": size=" + std::to_string(serializedContext->size) + ", expected_size="
                        + std::to_string(sizeof(TestPluginSerializedContextPayload))
                        + ", first_byte=" + std::to_string(firstByte) + ", expected_first_byte=66");
            }

            auto context = std::make_unique<HipdnnEnginePluginExecutionContext>();

            LOG_API_SUCCESS(apiName,
                            "createdExecutionContext=" << static_cast<void*>(context.get()));

            *executionContext = context.release();
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

    /// Shared C-API implementation used by fake plugins that export override execute.
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
#define REGISTER_TEST_PLUGIN_API()                                                               \
    extern "C" {                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginGetName(const char** name)                                                   \
    {                                                                                            \
        return TestPluginBase::pluginGetName(name);                                              \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginGetVersion(const char** version)                                             \
    {                                                                                            \
        return TestPluginBase::pluginGetVersion(version);                                        \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginGetApiVersion(const char** version)                                          \
    {                                                                                            \
        return TestPluginBase::pluginGetApiVersion(version);                                     \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginGetType(hipdnnPluginType_t* type)                                            \
    {                                                                                            \
        return TestPluginBase::pluginGetType(type);                                              \
    }                                                                                            \
                                                                                                 \
    HIPDNN_TEST_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** errorStr)         \
    {                                                                                            \
        TestPluginBase::pluginGetLastErrorString(errorStr);                                      \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                                \
    {                                                                                            \
        return TestPluginBase::pluginSetLoggingCallback(callback);                               \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnPluginSetLogLevel(hipdnnSeverity_t level)                                          \
    {                                                                                            \
        return TestPluginBase::pluginSetLogLevel(level);                                         \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds,                                    \
                                          uint32_t maxEngines,                                   \
                                          uint32_t* numEngines)                                  \
    {                                                                                            \
        return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);   \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)                             \
    {                                                                                            \
        return TestPluginBase::enginePluginCreate(handle);                                       \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)                             \
    {                                                                                            \
        return TestPluginBase::enginePluginDestroy(handle);                                      \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream)       \
    {                                                                                            \
        return TestPluginBase::enginePluginSetStream(handle, stream);                            \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,              \
                                                 const hipdnnPluginConstData_t* opGraph,         \
                                                 int64_t* engineIds,                             \
                                                 uint32_t maxEngines,                            \
                                                 uint32_t* numEngines)                           \
    {                                                                                            \
        return TestPluginBase::enginePluginGetApplicableEngineIds(                               \
            handle, opGraph, engineIds, maxEngines, numEngines);                                 \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,                    \
                                           int64_t engineId,                                     \
                                           const hipdnnPluginConstData_t* opGraph,               \
                                           hipdnnPluginConstData_t* engineDetails)               \
    {                                                                                            \
        return TestPluginBase::enginePluginGetEngineDetails(                                     \
            handle, engineId, opGraph, engineDetails);                                           \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,                \
                                               hipdnnPluginConstData_t* engineDetails)           \
    {                                                                                            \
        return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);          \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,                    \
                                           const hipdnnPluginConstData_t* engineConfig,          \
                                           const hipdnnPluginConstData_t* opGraph,               \
                                           size_t* workspaceSize)                                \
    {                                                                                            \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                     \
            handle, engineConfig, opGraph, workspaceSize);                                       \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(                                  \
            hipdnnEnginePluginHandle_t handle,                                                   \
            hipdnnEnginePluginExecutionContext_t executionContext,                               \
            size_t* workspaceSize)                                                               \
    {                                                                                            \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                     \
            handle, executionContext, workspaceSize);                                            \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginCreateExecutionContext(                                                \
            hipdnnEnginePluginHandle_t handle,                                                   \
            const hipdnnPluginConstData_t* engineConfig,                                         \
            const hipdnnPluginConstData_t* opGraph,                                              \
            hipdnnEnginePluginExecutionContext_t* executionContext)                              \
    {                                                                                            \
        return TestPluginBase::enginePluginCreateExecutionContext(                               \
            handle, engineConfig, opGraph, executionContext);                                    \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginDestroyExecutionContext(                                               \
            hipdnnEnginePluginHandle_t handle,                                                   \
            hipdnnEnginePluginExecutionContext_t executionContext)                               \
    {                                                                                            \
        return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);    \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginSerializeExecutionContext(                                             \
            hipdnnEnginePluginHandle_t handle,                                                   \
            hipdnnEnginePluginExecutionContext_t executionContext,                               \
            hipdnnPluginConstData_t* serializedContext)                                          \
    {                                                                                            \
        return TestPluginBase::enginePluginSerializeExecutionContext(                            \
            handle, executionContext, serializedContext);                                        \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginDestroySerializedExecutionContext(                                     \
            hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* serializedContext)       \
    {                                                                                            \
        return TestPluginBase::enginePluginDestroySerializedExecutionContext(handle,             \
                                                                             serializedContext); \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginCreateExecutionContextFromSerialized(                                  \
            hipdnnEnginePluginHandle_t handle,                                                   \
            const hipdnnPluginConstData_t* serializedContext,                                    \
            hipdnnEnginePluginExecutionContext_t* executionContext)                              \
    {                                                                                            \
        return TestPluginBase::enginePluginCreateExecutionContextFromSerialized(                 \
            handle, serializedContext, executionContext);                                        \
    }                                                                                            \
                                                                                                 \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t                       \
        hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,                      \
                                         hipdnnEnginePluginExecutionContext_t executionContext,  \
                                         void* workspace,                                        \
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,        \
                                         uint32_t numDeviceBuffers)                              \
    {                                                                                            \
        return TestPluginBase::enginePluginExecuteOpGraph(                                       \
            handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);               \
    }                                                                                            \
    } // extern "C"

/// Emits only the optional override-execute C symbol for fake plugins.
#define REGISTER_TEST_PLUGIN_OVERRIDE_API()                                               \
    extern "C" {                                                                          \
    HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t                     \
        hipdnnEnginePluginExecuteOpGraphWithOverrides(                                    \
            hipdnnEnginePluginHandle_t handle,                                            \
            hipdnnEnginePluginExecutionContext_t executionContext,                        \
            void* workspace,                                                              \
            const hipdnnPluginDeviceBuffer_t* deviceBuffers,                              \
            uint32_t numDeviceBuffers,                                                    \
            uint32_t numOverrides,                                                        \
            const int64_t* overrideUniqueIds,                                             \
            const uint32_t* overrideLengths,                                              \
            const int64_t* const* overrideShapes,                                         \
            const int64_t* const* overrideStrides)                                        \
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

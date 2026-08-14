// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "hipdnn_backend.h"

// Forward declarations
namespace hipdnn_backend::plugin
{
class EnginePlugin;
class EnginePluginManager;
} // namespace hipdnn_backend::plugin

// Include base template (manager types will be complete when base methods are used in .cpp)
#include "PluginResourceManagerBase.hpp"

// Include complete manager type for template instantiation
#include "EnginePluginManager.hpp"

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
struct EngineDetails;
} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_backend
{
class GraphDescriptor;
}
namespace hipdnn_backend::plugin
{

class EngineDetailsWrapper;
class EngineExecutionContextWrapper;
class EnginePlugin;
class EnginePluginManager;

struct EngineInfo
{
    std::string engineName;
    std::string pluginName;
    int64_t engineId;
    std::string version;
    std::string type;
};

class EnginePluginResourceManager : public PluginResourceManagerBase<EnginePluginResourceManager,
                                                                     EnginePluginManager,
                                                                     EnginePlugin>
{
    // Allow base class to access private static accessors
    friend class PluginResourceManagerBase<EnginePluginResourceManager,
                                           EnginePluginManager,
                                           EnginePlugin>;

private:
    // Static accessors for CRTP base class
    static std::mutex& getMutex();
    static PluginLoadingConfig& getConfig();
    static std::weak_ptr<EnginePluginManager>& getWeakPtr();
    static std::shared_ptr<EnginePluginManager>& getPersistentPtr();
    static std::atomic<bool>& getShutdownFlag();
    static const char* getPluginTypeName();

protected:
    // Protected constructor for mock testing
    EnginePluginResourceManager();

public:
    // MT-safe static functions (inherited from base, re-declared for documentation)
    using PluginResourceManagerBase::getPluginPaths;
    using PluginResourceManagerBase::setPluginLogLevel;
    using PluginResourceManagerBase::setPluginPaths;
    using PluginResourceManagerBase::setPluginUnloadingMode;

    static std::shared_ptr<EnginePluginResourceManager> create();

    EnginePluginResourceManager(std::shared_ptr<EnginePluginManager> pm);
    ~EnginePluginResourceManager() override;

    // Prevent copying
    EnginePluginResourceManager(const EnginePluginResourceManager&) = delete;
    EnginePluginResourceManager& operator=(const EnginePluginResourceManager&) = delete;

    // Allow moving
    EnginePluginResourceManager(EnginePluginResourceManager&& other) noexcept;
    EnginePluginResourceManager& operator=(EnginePluginResourceManager&& other) noexcept;

    // MT-unsafe instance methods
    // virtual for gMock testing
    virtual void setStream(hipStream_t stream) const;
    virtual std::vector<int64_t> getApplicableEngineIds(const GraphDescriptor* graphDesc,
                                                        bool findFirst = false) const;
    virtual size_t getWorkspaceSize(int64_t engineId,
                                    const hipdnnPluginConstData_t* engineConfig,
                                    const GraphDescriptor* graphDesc) const;
    virtual size_t getWorkspaceSize(int64_t engineId,
                                    hipdnnEnginePluginExecutionContext_t executionContext) const;
    virtual void serializeExecutionContext(int64_t engineId,
                                           hipdnnEnginePluginExecutionContext_t executionContext,
                                           std::vector<uint8_t>& serializedContext) const;

    virtual void executeOpGraph(hipdnnBackendDescriptor_t executionPlan,
                                hipdnnBackendDescriptor_t variantPack) const;

    static std::shared_ptr<const EngineDetailsWrapper>
        getEngineDetails(const std::shared_ptr<EnginePluginResourceManager>& rm,
                         int64_t engineId,
                         const GraphDescriptor* graphDesc);
    static std::shared_ptr<const EngineExecutionContextWrapper>
        createExecutionContext(const std::shared_ptr<EnginePluginResourceManager>& rm,
                               int64_t engineId,
                               const hipdnnPluginConstData_t* engineConfig,
                               const GraphDescriptor* graphDesc);
    static std::shared_ptr<const EngineExecutionContextWrapper>
        createExecutionContextFromSerialized(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                             int64_t engineId,
                                             const hipdnnPluginConstData_t* serializedContext);

    virtual size_t getEngineCount() const;
    virtual std::vector<EngineInfo> getEngineInfos() const;

    /// @brief Resolves the name for an engine: plugin entry point -> static
    /// registry -> hex ID.
    /// Never throws; always returns a non-empty string.
    /// @param engineId    Engine ID as reported by the owning plugin.
    /// @param detailsName Name recorded in EngineDetails.name, or std::nullopt
    ///                    when there is no graph. Never a source for the result:
    ///                    load-time admission cannot see this field, so a name
    ///                    carried only here was never validated. It is compared
    ///                    against the entry point when one answers and reported
    ///                    as a plugin defect otherwise. The referenced storage
    ///                    must outlive this call.
    [[nodiscard]] virtual std::string
        resolveEngineName(int64_t engineId, std::optional<std::string_view> detailsName) const;

    /// @brief Resolves a name to the engine ID that carries it.
    ///
    /// The lookup is the exact inverse of the names getEngineInfos() reports, so
    /// a name from that enumeration always resolves, and to exactly one engine:
    /// names are unique across loaded engines because an engine whose name does
    /// not hash to its ID is dropped at load. Unlike engineNameToId(), which
    /// hashes any string, this also confirms the engine is loaded.
    ///
    /// An engine named only through the graph-scoped EngineDetails.name is
    /// indexed under its registry or hex name instead, the same limitation
    /// getEngineInfos() carries.
    ///
    /// @param engineName Name to resolve. An empty name never matches.
    /// @return The owning engine ID, or std::nullopt when no loaded engine
    ///         carries the name.
    [[nodiscard]] virtual std::optional<int64_t>
        findEngineIdByName(std::string_view engineName) const;

    // Inherited from base: getLoadedPluginFiles()
    using PluginResourceManagerBase::getLoadedPluginFiles;

    virtual std::string toString() const;

protected:
    // Note: _pm member is inherited from PluginResourceManagerBase

private:
    // MT-unsafe instance methods
    // virtual for gMock testing
    virtual void getEngineDetails(int64_t engineId,
                                  const GraphDescriptor* graphDesc,
                                  hipdnnPluginConstData_t* engineDetails) const;
    virtual void destroyEngineDetails(int64_t engineId,
                                      hipdnnPluginConstData_t* engineDetails) const;

    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t
        createExecutionContext(int64_t engineId,
                               const hipdnnPluginConstData_t* engineConfig,
                               const GraphDescriptor* graphDesc) const;
    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t createExecutionContextFromSerialized(
        int64_t engineId, const hipdnnPluginConstData_t* serializedContext) const;
    virtual void
        destroyExecutionContext(int64_t engineId,
                                hipdnnEnginePluginExecutionContext_t executionContext) const;

    void executeOpGraph(int64_t engineId,
                        hipdnnEnginePluginExecutionContext_t executionContext,
                        void* workspace,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    /// @brief Materializes and memoizes the engine enumeration together with its
    /// name -> ID reverse index, and returns the enumeration by reference.
    ///
    /// Both memos are filled in the same pass so they can never disagree about
    /// which engine carries which name.
    const std::vector<EngineInfo>& buildEngineIndex() const;

    std::unordered_map<hipdnnEnginePluginHandle_t, const EnginePlugin*> _handleToPlugin;
    std::unordered_map<int64_t, hipdnnEnginePluginHandle_t> _engineIdToHandle;
    mutable std::optional<std::vector<EngineInfo>> _cachedEngineInfos;
    mutable std::optional<std::unordered_map<std::string, int64_t>> _cachedEngineIdsByName;

    friend class EngineDetailsWrapper;
    friend class EngineExecutionContextWrapper;
};

// A class to manage engine details lifecycle
class EngineDetailsWrapper
{
public:
    EngineDetailsWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                         int64_t engineId,
                         const GraphDescriptor* graphDesc);
    ~EngineDetailsWrapper();

    // Prevent copying
    EngineDetailsWrapper(const EngineDetailsWrapper&) = delete;
    EngineDetailsWrapper& operator=(const EngineDetailsWrapper&) = delete;

    // Allow moving
    EngineDetailsWrapper(EngineDetailsWrapper&& other) noexcept;
    EngineDetailsWrapper& operator=(EngineDetailsWrapper&& other) noexcept;

    const hipdnn_flatbuffers_sdk::data_objects::EngineDetails* get() const;

private:
    std::shared_ptr<EnginePluginResourceManager> _rm;
    int64_t _engineId = 0;
    hipdnnPluginConstData_t _engineDetailsData{nullptr, 0};
};

// A class to manage engine execution context lifecycle
class EngineExecutionContextWrapper
{
public:
    EngineExecutionContextWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                  int64_t engineId,
                                  const hipdnnPluginConstData_t* engineConfig,
                                  const GraphDescriptor* graphDesc);
    EngineExecutionContextWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                  int64_t engineId,
                                  const hipdnnPluginConstData_t* serializedContext);
    ~EngineExecutionContextWrapper();

    // Prevent copying
    EngineExecutionContextWrapper(const EngineExecutionContextWrapper&) = delete;
    EngineExecutionContextWrapper& operator=(const EngineExecutionContextWrapper&) = delete;

    // Allow moving
    EngineExecutionContextWrapper(EngineExecutionContextWrapper&& other) noexcept;
    EngineExecutionContextWrapper& operator=(EngineExecutionContextWrapper&& other) noexcept;

    hipdnnEnginePluginExecutionContext_t get() const;

private:
    std::shared_ptr<EnginePluginResourceManager> _rm;
    int64_t _engineId;
    hipdnnEnginePluginExecutionContext_t _executionContext;
};

} // namespace hipdnn_backend::plugin

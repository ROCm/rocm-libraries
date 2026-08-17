// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "EnginePlugin.hpp"
#include "EnginePluginManager.hpp"
#include "EnginePluginResourceManager.hpp"
#include "HipdnnException.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include "logging/Logging.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <spdlog/fmt/ranges.h>

namespace hipdnn_backend
{
namespace plugin
{

namespace
{

// Static storage for engine plugin configuration
std::mutex gEngineMutex;
PluginLoadingConfig gEngineConfig;
std::weak_ptr<EnginePluginManager> gEngineWeakPtr;
std::shared_ptr<EnginePluginManager> gEnginePersistentPtr;
std::atomic<bool> gEngineShutdownFlag{false};

// Register atexit handler to set shutdown flag
struct EnginePluginShutdownRegistrar
{
    EnginePluginShutdownRegistrar()
    {
        std::atexit([]() { gEngineShutdownFlag.store(true, std::memory_order_release); });
    }
};

EnginePluginShutdownRegistrar gEngineShutdownRegistrar;

/// Missing override flag means false; other descriptor errors propagate.
bool readIsOverrideShapeEnabled(const GraphDescriptor& graphDesc)
{
    bool flag = false;
    int64_t elementCount = 0;
    try
    {
        graphDesc.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                               HIPDNN_TYPE_BOOLEAN,
                               1,
                               &elementCount,
                               &flag);
    }
    catch(const HipdnnException& ex)
    {
        // Only swallow "attribute not supported" — anything else is a real
        // error in the descriptor or its attribute machinery.
        if(ex.getStatus() == HIPDNN_STATUS_NOT_SUPPORTED)
        {
            return false;
        }
        throw;
    }
    if(elementCount < 1)
    {
        return false;
    }
    return flag;
}

} // namespace

// Static accessor implementations for CRTP base class
std::mutex& EnginePluginResourceManager::getMutex()
{
    return gEngineMutex;
}

PluginLoadingConfig& EnginePluginResourceManager::getConfig()
{
    return gEngineConfig;
}

std::weak_ptr<EnginePluginManager>& EnginePluginResourceManager::getWeakPtr()
{
    return gEngineWeakPtr;
}

std::shared_ptr<EnginePluginManager>& EnginePluginResourceManager::getPersistentPtr()
{
    return gEnginePersistentPtr;
}

std::atomic<bool>& EnginePluginResourceManager::getShutdownFlag()
{
    return gEngineShutdownFlag;
}

const char* EnginePluginResourceManager::getPluginTypeName()
{
    return "engine";
}

size_t EnginePluginResourceManager::getEngineCount() const
{
    return getEngineInfos().size();
}

std::vector<EngineInfo> EnginePluginResourceManager::getEngineInfos() const
{
    return buildEngineIndex();
}

std::optional<int64_t>
    EnginePluginResourceManager::findEngineIdByName(std::string_view engineName) const
{
    buildEngineIndex();

    const auto it = _cachedEngineIdsByName->find(std::string(engineName));
    if(it == _cachedEngineIdsByName->end())
    {
        return std::nullopt;
    }

    return it->second;
}

std::optional<std::string> EnginePluginResourceManager::findEngineNameById(int64_t engineId) const
{
    // Scanning rather than adding a third memo keeps the fill-both-together
    // invariant intact, and the engine count is small.
    const auto& infos = buildEngineIndex();

    const auto it = std::find_if(infos.begin(), infos.end(), [engineId](const EngineInfo& info) {
        return info.engineId == engineId;
    });

    if(it == infos.end())
    {
        return std::nullopt;
    }

    return it->engineName;
}

const std::vector<EngineInfo>& EnginePluginResourceManager::buildEngineIndex() const
{
    // Public entry points reach this concurrently, so the fill is
    // serialized. Neither memo is invalidated afterwards, which is what lets
    // callers hold the returned reference past the lock; a future reset path
    // would have to return copies instead.
    const std::lock_guard<std::mutex> lock(_engineIndexMutex);

    // Both memos are filled together on every path, so requiring both here keeps
    // findEngineIdByName() from ever seeing a half-built index.
    if(_cachedEngineInfos.has_value() && _cachedEngineIdsByName.has_value())
    {
        return *_cachedEngineInfos;
    }

    std::vector<EngineInfo> infos;
    if(!_pm)
    {
        _cachedEngineIdsByName.emplace();
        return _cachedEngineInfos.emplace(std::move(infos));
    }

    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        auto pluginVersion = std::string(plugin->version());
        auto pluginType = std::string(::toString(plugin->type()));
        auto pluginName = std::string(plugin->name());

        // The accepted set omits engines dropped at load time, so they reach
        // neither the enumeration nor the reverse index.
        for(const auto id : _pm->acceptedEngineIds(*plugin))
        {
            EngineInfo info;
            info.engineId = id;
            info.version = pluginVersion;
            info.type = pluginType;
            info.pluginName = pluginName;

            // No graph here, so no EngineDetails candidate.
            info.engineName = resolveEngineName(id, std::nullopt);

            infos.push_back(std::move(info));
        }
    }

    // Alphabetical by resolved name is the documented contract. The tie breakers
    // make the comparator a total order, keeping the order stable across runs.
    std::sort(infos.begin(), infos.end(), [](const EngineInfo& a, const EngineInfo& b) {
        return std::tie(a.engineName, a.engineId, a.pluginName)
               < std::tie(b.engineName, b.engineId, b.pluginName);
    });

    // Built from the sorted vector, so it agrees with the enumeration. Admission
    // ties each name to its own hash, so distinct engines cannot collide here.
    auto& idsByName = _cachedEngineIdsByName.emplace();
    idsByName.reserve(infos.size());
    for(const auto& info : infos)
    {
        idsByName.emplace(info.engineName, info.engineId);
    }

    return _cachedEngineInfos.emplace(std::move(infos));
}

std::string EnginePluginResourceManager::resolveEngineName(
    int64_t engineId, std::optional<std::string_view> detailsName) const
{
    const EnginePlugin* owningPlugin = _pm ? _pm->engineOwner(engineId) : nullptr;

    const std::string_view pluginName = owningPlugin != nullptr
                                            ? std::string_view(owningPlugin->cachedName())
                                            : std::string_view("<unknown>");

    // The entry point is the only channel a plugin can name an engine through,
    // because it is the only one load-time admission can reach.
    std::optional<std::string> entryPointName;
    if(owningPlugin != nullptr && owningPlugin->hasEngineName())
    {
        entryPointName = owningPlugin->getEngineName(engineId);
    }

    if(entryPointName.has_value())
    {
        // The entry point is authoritative; a disagreement is a plugin defect
        // worth reporting, not worth failing over.
        if(detailsName.has_value() && !detailsName->empty() && *detailsName != *entryPointName)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Plugin '{}' names engine {} '{}' through hipdnnEnginePluginGetEngineName but "
                "'{}' in EngineDetails.name; using '{}'",
                pluginName,
                hipdnn_data_sdk::utilities::formatEngineIdHex(engineId),
                *entryPointName,
                *detailsName,
                *entryPointName);
        }

        // Admission already established that this name hashes to engineId.
        return *entryPointName;
    }

    // EngineDetails.name records a name but never confers one. Admission is
    // graph-blind and cannot see this field, so honoring a candidate that
    // survived to here would surface a name no load-time gate ever checked.
    // Declaring a name here but not through the entry point is a plugin defect;
    // the name resolves from the registry or the hex ID instead.
    if(detailsName.has_value() && !detailsName->empty())
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Plugin '{}' names engine {} '{}' in EngineDetails.name but does not report that name "
            "through hipdnnEnginePluginGetEngineName. An engine name must be declared through the "
            "entry point so load-time admission can validate it; ignoring the reported name.",
            pluginName,
            hipdnn_data_sdk::utilities::formatEngineIdHex(engineId),
            *detailsName);
    }

    return hipdnn_data_sdk::utilities::engineNameOrHex(engineId);
}

std::shared_ptr<EnginePluginResourceManager> EnginePluginResourceManager::create()
{
    auto pm = getOrCreatePluginManager();
    return std::make_shared<EnginePluginResourceManager>(pm);
}

EnginePluginResourceManager::EnginePluginResourceManager()
    : PluginResourceManagerBase(std::make_shared<EnginePluginManager>())
{
}

EnginePluginResourceManager::EnginePluginResourceManager(std::shared_ptr<EnginePluginManager> pm)
    : PluginResourceManagerBase(std::move(pm))
{
    // Helper to safely destroy a handle during error cleanup, logging any failures
    auto safeDestroyHandle = [](const EnginePlugin* plugin, hipdnnEnginePluginHandle_t handle) {
        try
        {
            plugin->destroyHandle(handle);
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to destroy handle for plugin '{}' during cleanup: {}",
                                    plugin->name(),
                                    e.what());
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to destroy handle for plugin '{}' during cleanup: unknown error",
                plugin->name());
        }
    };

    // Create plugin handles
    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        hipdnnEnginePluginHandle_t handle = nullptr;

        try
        {
            handle = plugin->createHandle();
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_ERROR(
                "Failed to create handle for plugin '{}': {}", plugin->name(), e.what());
            continue;
        }

        if(handle == nullptr)
        {
            HIPDNN_BACKEND_LOG_ERROR("Plugin '{}' returned null handle", plugin->name());
            continue;
        }

        if(_handleToPlugin.find(handle) != _handleToPlugin.end())
        {
            safeDestroyHandle(plugin.get(), handle);
            HIPDNN_BACKEND_LOG_ERROR(
                "Plugin '{}' returned a handle that collides with another plugin. "
                "This may indicate a symbol collision between plugins. "
                "Ensure all plugins are built with -fvisibility=hidden.",
                plugin->name());
            continue;
        }

        _handleToPlugin[handle] = plugin.get();

        // Nested in the success path on purpose: a plugin that never got a handle
        // has nothing to route to. Routing is built from the accepted set, so a
        // dropped engine cannot be reached at all.
        for(const auto id : _pm->acceptedEngineIds(*plugin))
        {
            _engineIdToHandle[id] = handle;
        }
    }
}

EnginePluginResourceManager::~EnginePluginResourceManager()
{
    // Lambda to safely destroy a handle, catching all errors
    auto safeDestroyHandle = [](const EnginePlugin* plugin, hipdnnEnginePluginHandle_t handle) {
        try
        {
            plugin->destroyHandle(handle);
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to destroy handle for plugin '{}' during cleanup: {}",
                                    plugin->name(),
                                    e.what());
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to destroy handle for plugin '{}' during cleanup: unknown error",
                plugin->name());
        }
    };

    // Destroy plugin handles
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        safeDestroyHandle(plugin, handle);
    }
}

EnginePluginResourceManager::EnginePluginResourceManager(
    EnginePluginResourceManager&& other) noexcept
    : _handleToPlugin(std::move(other._handleToPlugin))
    , _engineIdToHandle(std::move(other._engineIdToHandle))
    , _cachedEngineInfos(std::move(other._cachedEngineInfos))
    , _cachedEngineIdsByName(std::move(other._cachedEngineIdsByName))
{
    // Move base class member explicitly
    _pm = std::move(other._pm);
}

EnginePluginResourceManager&
    EnginePluginResourceManager::operator=(EnginePluginResourceManager&& other) noexcept
{
    if(this != &other)
    {
        _handleToPlugin = std::move(other._handleToPlugin);
        _engineIdToHandle = std::move(other._engineIdToHandle);
        _cachedEngineInfos = std::move(other._cachedEngineInfos);
        _cachedEngineIdsByName = std::move(other._cachedEngineIdsByName);
        _pm = std::move(other._pm);
    }
    return *this;
}

void EnginePluginResourceManager::setStream(hipStream_t stream) const
{
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        plugin->setStream(handle, stream);
    }
}

std::vector<int64_t>
    EnginePluginResourceManager::getApplicableEngineIds(const GraphDescriptor* graphDesc,
                                                        bool findFirst) const
{
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");

    auto serializedGraphData = graphDesc->getSerializedGraph();

    // Applicability filter: all graphs require the baseline engine plugin API,
    // and graphs that opt in to overridable tensor shapes require the extended
    // override-execute SDK surface. Older explicit API versions are skipped.
    const bool isOverrideShapeEnabled = readIsOverrideShapeEnabled(*graphDesc);
    const bool isRuntimePBV = graphDesc->isRuntimePassByValueEnabled();
    const bool isRaggedTensorEnabled = graphDesc->hasRaggedTensors();
    const bool hasNonDefaultTensorAlignment = graphDesc->hasNonDefaultTensorAlignment();

    const auto& requiredVersion = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(
        isOverrideShapeEnabled, isRuntimePBV, isRaggedTensorEnabled, hasNonDefaultTensorAlignment);

    std::vector<int64_t> engineIds;

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        // Safe to deref: validateBeforeAdding rejected plugins with
        // unparseable versions at load time.
        const auto pluginVersion = *plugin->parsedApiVersion();
        if(pluginVersion < requiredVersion)
        {
            HIPDNN_BACKEND_LOG_INFO(
                "Skipping plugin '{}' (apiVersion={}) for graph requiring at least {}",
                plugin->cachedName(),
                pluginVersion.str(),
                requiredVersion.str());
            continue;
        }

        if(isOverrideShapeEnabled && !plugin->hasOverrideExecute())
        {
            HIPDNN_BACKEND_LOG_INFO(
                "Skipping plugin '{}' for override-enabled graph because it does not export "
                "hipdnnEnginePluginExecuteOpGraphWithOverrides",
                plugin->cachedName());
            continue;
        }

        const auto ids = plugin->getApplicableEngineIds(handle, &serializedGraphData);

        for(const auto& id : ids)
        {
            const auto handleIt = _engineIdToHandle.find(id);
            if(handleIt == _engineIdToHandle.end())
            {
                // A plugin is never told which of its engines were dropped, so it
                // keeps offering them; skipping beats failing the whole graph.
                HIPDNN_BACKEND_LOG_INFO(
                    "Skipping engine {} offered by plugin '{}': it was dropped at load time",
                    hipdnn_data_sdk::utilities::formatEngineIdHex(id),
                    plugin->cachedName());
                continue;
            }

            if(handleIt->second != handle)
            {
                throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                      "Engine ID " + std::to_string(id)
                                          + " is already associated with a different plugin");
            }

            engineIds.push_back(id);
        }

        if(findFirst && !engineIds.empty())
        {
            break;
        }
    }

    return engineIds;
}

void EnginePluginResourceManager::getEngineDetails(int64_t engineId,
                                                   const GraphDescriptor* graphDesc,
                                                   hipdnnPluginConstData_t* engineDetails) const
{
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");
    THROW_IF_NULL(engineDetails, HIPDNN_STATUS_INTERNAL_ERROR, "Engine details cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->getSerializedGraph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    plugin->getEngineDetails(handle, engineId, &serializedGraphData, engineDetails);

    if(engineDetails->ptr == nullptr || engineDetails->size == 0)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Engine details for engine ID " + std::to_string(engineId)
                                  + " are empty or null");
    }
}

void EnginePluginResourceManager::destroyEngineDetails(int64_t engineId,
                                                       hipdnnPluginConstData_t* engineDetails) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->destroyEngineDetails(handle, engineDetails);
}

std::shared_ptr<const EngineDetailsWrapper> EnginePluginResourceManager::getEngineDetails(
    const std::shared_ptr<EnginePluginResourceManager>& rm,
    int64_t engineId,
    const GraphDescriptor* graphDesc)
{
    return std::make_shared<EngineDetailsWrapper>(rm, engineId, graphDesc);
}

size_t EnginePluginResourceManager::getWorkspaceSize(int64_t engineId,
                                                     const hipdnnPluginConstData_t* engineConfig,
                                                     const GraphDescriptor* graphDesc) const
{
    THROW_IF_NULL(engineConfig, HIPDNN_STATUS_INTERNAL_ERROR, "Engine config cannot be null");
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->getSerializedGraph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->getWorkspaceSize(handle, engineConfig, &serializedGraphData);
}

// TODO: Pack engineConfig
// TODO: Get engineId from engineConfig
hipdnnEnginePluginExecutionContext_t
    EnginePluginResourceManager::createExecutionContext(int64_t engineId,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const GraphDescriptor* graphDesc) const
{
    THROW_IF_NULL(engineConfig, HIPDNN_STATUS_BAD_PARAM, "Engine config cannot be null");
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_BAD_PARAM, "Graph descriptor cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->getSerializedGraph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->createExecutionContext(handle, engineConfig, &serializedGraphData);
}

hipdnnEnginePluginExecutionContext_t
    EnginePluginResourceManager::createExecutionContextFromSerialized(
        int64_t engineId, const hipdnnPluginConstData_t* serializedContext) const
{
    THROW_IF_NULL(
        serializedContext, HIPDNN_STATUS_BAD_PARAM, "Serialized execution context cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->createExecutionContextFromSerialized(handle, serializedContext);
}

void EnginePluginResourceManager::destroyExecutionContext(
    int64_t engineId, hipdnnEnginePluginExecutionContext_t executionContext) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->destroyExecutionContext(handle, executionContext);
}

std::shared_ptr<const EngineExecutionContextWrapper>
    EnginePluginResourceManager::createExecutionContext(
        const std::shared_ptr<EnginePluginResourceManager>& rm,
        int64_t engineId,
        const hipdnnPluginConstData_t* engineConfig,
        const GraphDescriptor* graphDesc)
{
    return std::make_shared<EngineExecutionContextWrapper>(rm, engineId, engineConfig, graphDesc);
}

std::shared_ptr<const EngineExecutionContextWrapper>
    EnginePluginResourceManager::createExecutionContextFromSerialized(
        const std::shared_ptr<EnginePluginResourceManager>& rm,
        int64_t engineId,
        const hipdnnPluginConstData_t* serializedContext)
{
    return std::make_shared<EngineExecutionContextWrapper>(rm, engineId, serializedContext);
}

size_t EnginePluginResourceManager::getWorkspaceSize(
    int64_t engineId, hipdnnEnginePluginExecutionContext_t executionContext) const
{
    THROW_IF_NULL(
        executionContext, HIPDNN_STATUS_INTERNAL_ERROR, "Execution context cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->getWorkspaceSize(handle, executionContext);
}

void EnginePluginResourceManager::serializeExecutionContext(
    int64_t engineId,
    hipdnnEnginePluginExecutionContext_t executionContext,
    std::vector<uint8_t>& serializedContext) const
{
    THROW_IF_NULL(executionContext, HIPDNN_STATUS_BAD_PARAM, "Execution context cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid engine ID: " + std::to_string(engineId));
    }

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    hipdnnPluginConstData_t pluginData{nullptr, 0};
    plugin->serializeExecutionContext(handle, executionContext, &pluginData);

    try
    {
        THROW_IF_NULL(pluginData.ptr,
                      HIPDNN_STATUS_PLUGIN_ERROR,
                      "Serialized execution context payload is null");
        THROW_IF_TRUE(pluginData.size == 0,
                      HIPDNN_STATUS_PLUGIN_ERROR,
                      "Serialized execution context payload is empty");

        serializedContext.resize(pluginData.size);
        std::memcpy(serializedContext.data(), pluginData.ptr, pluginData.size);
    }
    catch(...)
    {
        plugin->destroySerializedExecutionContext(handle, &pluginData);
        throw;
    }

    plugin->destroySerializedExecutionContext(handle, &pluginData);
}

void EnginePluginResourceManager::executeOpGraph(
    int64_t engineId,
    hipdnnEnginePluginExecutionContext_t executionContext,
    void* workspace,
    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
    uint32_t numDeviceBuffers) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->executeOpGraph(handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}

void EnginePluginResourceManager::executeOpGraph(hipdnnBackendDescriptor_t executionPlan,
                                                 hipdnnBackendDescriptor_t variantPack) const
{
    auto executionPlanDesc = executionPlan->asDescriptor<ExecutionPlanDescriptor>();
    auto variantPackDesc = variantPack->asDescriptor<VariantDescriptor>();

    THROW_IF_FALSE(executionPlanDesc->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: executionPlanDesc "
                   "is not finalized");

    THROW_IF_FALSE(variantPackDesc->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: variantPackDesc is "
                   "not finalized");

    auto engineId = executionPlanDesc->getEngineId();
    void* workspace = variantPackDesc->getWorkspace();

    auto& tensorIds = variantPackDesc->getTensorIds();
    auto& tensorPointers = variantPackDesc->getDataPointers();

    THROW_IF_NE(tensorIds.size(),
                tensorPointers.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_plugin_resource_manager::execute_op_graph failed: "
                "tensorIds and tensorPointers must have the same size");

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
    deviceBuffers.reserve(tensorIds.size());
    for(size_t i = 0; i < tensorIds.size(); ++i)
    {
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = tensorIds[i];
        buffer.ptr = const_cast<void*>(tensorPointers[i]);
        deviceBuffers.push_back(buffer);
    }

    // Enforce each tensor's required device-pointer byte alignment. Alignments
    // are carried on the execution plan (populated from the graph at finalize and
    // preserved across plan serialization), keyed by tensor uid. Null pointers
    // (absent optional tensors) and uids without a recorded alignment are skipped.
    const auto& planTensorUids = executionPlanDesc->getTensorUids();
    const auto& planTensorAlignments = executionPlanDesc->getTensorAlignments();
    if(!planTensorAlignments.empty() && planTensorUids.size() == planTensorAlignments.size())
    {
        std::unordered_map<int64_t, int64_t> alignmentByUid;
        alignmentByUid.reserve(planTensorUids.size());
        for(size_t i = 0; i < planTensorUids.size(); ++i)
        {
            alignmentByUid.emplace(planTensorUids[i], planTensorAlignments[i]);
        }

        for(const auto& buffer : deviceBuffers)
        {
            if(buffer.ptr == nullptr)
            {
                continue;
            }

            const auto it = alignmentByUid.find(buffer.uid);
            if(it == alignmentByUid.end() || it->second <= 0)
            {
                continue;
            }

            const auto alignment = static_cast<uintptr_t>(it->second);
            const auto address = reinterpret_cast<uintptr_t>(buffer.ptr);
            THROW_IF_TRUE(address % alignment != 0,
                          HIPDNN_STATUS_BAD_PARAM,
                          "Tensor uid " + std::to_string(buffer.uid)
                              + " device pointer is not aligned to the required "
                              + std::to_string(it->second) + " bytes");
        }
    }

    const auto& overrideUniqueIds = variantPackDesc->getOverrideUniqueIds();
    const auto& overrideShapesFlat = variantPackDesc->getOverrideShapes();
    const auto& overrideStridesFlat = variantPackDesc->getOverrideStrides();
    const auto& overrideLengths64 = variantPackDesc->getOverrideLengths();

    const bool hasOverrides = !overrideUniqueIds.empty() || !overrideShapesFlat.empty()
                              || !overrideStridesFlat.empty() || !overrideLengths64.empty();

    if(hasOverrides)
    {
        THROW_IF_FALSE(executionPlanDesc->isOverrideShapeEnabled(),
                       HIPDNN_STATUS_NOT_SUPPORTED,
                       "Execution plan was not built with override shape support enabled, but the "
                       "variant pack carries override-tensor selectors.");

        THROW_IF_NE(overrideUniqueIds.size(),
                    overrideLengths64.size(),
                    HIPDNN_STATUS_BAD_PARAM,
                    "Override variant pack: OVERRIDE_UNIQUE_IDS and OVERRIDE_LENGTHS must have "
                    "the same size");

        auto handleIt = _engineIdToHandle.find(engineId);
        THROW_IF_FALSE(
            handleIt != _engineIdToHandle.end(),
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Engine_plugin_resource_manager::execute_op_graph failed: unknown engine ID");
        const auto* plugin = _handleToPlugin.at(handleIt->second);

        // Never silently fall back to legacy execute when override metadata exists.
        THROW_IF_FALSE(plugin->hasOverrideExecute(),
                       HIPDNN_STATUS_NOT_SUPPORTED,
                       "Selected plugin does not export "
                       "hipdnnEnginePluginExecuteOpGraphWithOverrides although the variant pack "
                       "carries override-tensor selectors.");

        // Defense-in-depth recheck of the override-execute floor. Override-shape
        // support is hipDNN-owned routing metadata carried in the execution plan
        // envelope, so it is cheap and correct to re-verify here that the selected
        // plugin still meets the override API floor. Pass-by-value is intentionally
        // false: per RFC 0009, once a serialized plan resolves to an engine id the
        // plugin owns its own payload versioning/compatibility, so hipDNN does not
        // re-gate feature floors (like pbv) that live in the plugin payload rather
        // than the envelope.
        const auto pluginApiVersion = plugin->parsedApiVersion();
        THROW_IF_FALSE(pluginApiVersion.has_value()
                           && *pluginApiVersion
                                  >= hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(
                                      true,
                                      /*isRuntimePassByValue=*/false,
                                      /*isRaggedTensorEnabled=*/false,
                                      /*hasNonDefaultTensorAlignment=*/false),
                       HIPDNN_STATUS_NOT_SUPPORTED,
                       "Selected plugin API version does not support "
                       "hipdnnEnginePluginExecuteOpGraphWithOverrides.");

        // Validate before narrowing variant-pack int64 lengths to the SDK uint32 surface.
        const auto numOverridesSize = overrideUniqueIds.size();
        if constexpr(sizeof(size_t) > sizeof(uint32_t))
        {
            THROW_IF_TRUE(numOverridesSize > std::numeric_limits<uint32_t>::max(),
                          HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                          "Override variant pack: number of overrides exceeds uint32 max");
        }

        std::vector<uint32_t> overrideLengthsU32;
        overrideLengthsU32.reserve(numOverridesSize);
        for(size_t i = 0; i < overrideLengths64.size(); ++i)
        {
            const auto value = overrideLengths64[i];
            THROW_IF_TRUE(value <= 0,
                          HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                          "Override variant pack: OVERRIDE_LENGTHS for unique-id "
                              + std::to_string(overrideUniqueIds[i]) + " must be positive ("
                              + std::to_string(value) + ")");
            THROW_IF_TRUE(static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max(),
                          HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                          "Override variant pack: OVERRIDE_LENGTHS for unique-id "
                              + std::to_string(overrideUniqueIds[i]) + " exceeds uint32 max ("
                              + std::to_string(value) + ")");
            overrideLengthsU32.push_back(static_cast<uint32_t>(value));
        }

        // Reconstruct per-UID pointers into the flat shape/stride buffers.
        std::vector<const int64_t*> overrideShapesPerUid;
        overrideShapesPerUid.reserve(numOverridesSize);
        std::vector<const int64_t*> overrideStridesPerUid;
        overrideStridesPerUid.reserve(numOverridesSize);

        size_t expectedFlatTotal = 0;
        for(const auto rank : overrideLengthsU32)
        {
            const auto rankSize = static_cast<size_t>(rank);
            THROW_IF_TRUE(expectedFlatTotal > std::numeric_limits<size_t>::max() - rankSize,
                          HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                          "Override variant pack: OVERRIDE_LENGTHS sum overflow");
            expectedFlatTotal += rankSize;
        }
        THROW_IF_NE(overrideShapesFlat.size(),
                    expectedFlatTotal,
                    HIPDNN_STATUS_BAD_PARAM,
                    "Override variant pack: OVERRIDE_SHAPES total length does not match the "
                    "sum of OVERRIDE_LENGTHS");
        THROW_IF_NE(overrideStridesFlat.size(),
                    expectedFlatTotal,
                    HIPDNN_STATUS_BAD_PARAM,
                    "Override variant pack: OVERRIDE_STRIDES total length does not match the "
                    "sum of OVERRIDE_LENGTHS");

        size_t offset = 0;
        for(size_t i = 0; i < numOverridesSize; ++i)
        {
            overrideShapesPerUid.push_back(overrideShapesFlat.data() + offset);
            overrideStridesPerUid.push_back(overrideStridesFlat.data() + offset);
            offset += static_cast<size_t>(overrideLengthsU32[i]);
        }

        plugin->executeOpGraphWithOverrides(handleIt->second,
                                            executionPlanDesc->getExecutionContext(),
                                            workspace,
                                            deviceBuffers.data(),
                                            static_cast<uint32_t>(tensorIds.size()),
                                            static_cast<uint32_t>(numOverridesSize),
                                            overrideUniqueIds.data(),
                                            overrideLengthsU32.data(),
                                            overrideShapesPerUid.data(),
                                            overrideStridesPerUid.data());
        return;
    }

    executeOpGraph(engineId,
                   executionPlanDesc->getExecutionContext(),
                   workspace,
                   deviceBuffers.data(),
                   static_cast<uint32_t>(tensorIds.size()));
}

EngineDetailsWrapper::EngineDetailsWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                           int64_t engineId,
                                           const GraphDescriptor* graphDesc)
    : _rm(rm)
    , _engineId(engineId)
{
    hipdnnPluginConstData_t engineDetailsData{nullptr, 0};
    _rm->getEngineDetails(engineId, graphDesc, &engineDetailsData);

    try
    {
        flatbuffers::Verifier verifier(static_cast<const uint8_t*>(engineDetailsData.ptr),
                                       engineDetailsData.size);
        if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::EngineDetails>())
        {
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                                  "EngineDetailsWrapper: unable to verify the flatbuffer schema.");
        }
    }
    catch(...)
    {
        if(engineDetailsData.ptr != nullptr)
        {
            try
            {
                _rm->destroyEngineDetails(engineId, &engineDetailsData);
            }
            catch(const HipdnnException& e)
            {
                HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
            }
        }
        throw;
    }

    _engineDetailsData = engineDetailsData;
}

EngineDetailsWrapper::~EngineDetailsWrapper()
{
    if(_engineDetailsData.ptr == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroyEngineDetails(_engineId, &_engineDetailsData);
    }
    catch(const HipdnnException& e)
    {
        HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
    }
}

EngineDetailsWrapper::EngineDetailsWrapper(EngineDetailsWrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engineId(other._engineId)
    , _engineDetailsData(other._engineDetailsData)
{
    other._rm = nullptr;
    other._engineId = 0;
    other._engineDetailsData.ptr = nullptr;
    other._engineDetailsData.size = 0;
}

EngineDetailsWrapper& EngineDetailsWrapper::operator=(EngineDetailsWrapper&& other) noexcept
{
    if(this != &other)
    {
        if(_engineDetailsData.ptr != nullptr && _rm != nullptr)
        {
            try
            {
                _rm->destroyEngineDetails(_engineId, &_engineDetailsData);
            }
            catch(const HipdnnException& e)
            {
                HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
            }
        }

        _rm = std::move(other._rm);
        _engineId = other._engineId;
        _engineDetailsData = other._engineDetailsData;

        other._rm = nullptr;
        other._engineId = 0;
        other._engineDetailsData.ptr = nullptr;
        other._engineDetailsData.size = 0;
    }
    return *this;
}

const hipdnn_flatbuffers_sdk::data_objects::EngineDetails* EngineDetailsWrapper::get() const
{
    if(_engineDetailsData.ptr == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "EngineDetailsWrapper: wrong usage: "
                              "get() called on an empty object");
    }

    return hipdnn_flatbuffers_sdk::data_objects::GetEngineDetails(_engineDetailsData.ptr);
}

// TODO: Use engineId from engineConfig
EngineExecutionContextWrapper::EngineExecutionContextWrapper(
    const std::shared_ptr<EnginePluginResourceManager>& rm,
    int64_t engineId,
    const hipdnnPluginConstData_t* engineConfig,
    const GraphDescriptor* graphDesc)
    : _rm(rm)
    , _engineId(engineId)
{
    _executionContext = _rm->createExecutionContext(engineId, engineConfig, graphDesc);
}

EngineExecutionContextWrapper::EngineExecutionContextWrapper(
    const std::shared_ptr<EnginePluginResourceManager>& rm,
    int64_t engineId,
    const hipdnnPluginConstData_t* serializedContext)
    : _rm(rm)
    , _engineId(engineId)
{
    _executionContext = _rm->createExecutionContextFromSerialized(engineId, serializedContext);
}

EngineExecutionContextWrapper::~EngineExecutionContextWrapper()
{
    if(_executionContext == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroyExecutionContext(_engineId, _executionContext);
    }
    catch(const HipdnnException& e)
    {
        HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
    }
}

EngineExecutionContextWrapper::EngineExecutionContextWrapper(
    EngineExecutionContextWrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engineId(other._engineId)
    , _executionContext(other._executionContext)
{
    other._rm = nullptr;
    other._engineId = 0;
    other._executionContext = nullptr;
}

EngineExecutionContextWrapper&
    EngineExecutionContextWrapper::operator=(EngineExecutionContextWrapper&& other) noexcept
{
    if(this != &other)
    {
        if(_executionContext != nullptr && _rm != nullptr)
        {
            try
            {
                _rm->destroyExecutionContext(_engineId, _executionContext);
            }
            catch(const HipdnnException& e)
            {
                HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
            }
        }

        _rm = std::move(other._rm);
        _engineId = other._engineId;
        _executionContext = other._executionContext;

        other._rm = nullptr;
        other._engineId = 0;
        other._executionContext = nullptr;
    }
    return *this;
}

hipdnnEnginePluginExecutionContext_t EngineExecutionContextWrapper::get() const
{
    if(_executionContext == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "EngineExecutionContextWrapper: wrong usage: "
                              "get() called on an empty object");
    }

    return _executionContext;
}

std::string EnginePluginResourceManager::toString() const
{
    if(!_pm)
    {
        return "EnginePluginResourceManager: {loadedPlugins=0}";
    }

    auto loadedPlugins = _pm->getLoadedPluginFiles();

    std::vector<std::string> pluginPathStrings;
    pluginPathStrings.reserve(loadedPlugins.size());
    for(const auto& path : loadedPlugins)
    {
        pluginPathStrings.push_back(path.string());
    }

    return fmt::format("EnginePluginResourceManager: {{loadedPlugins={}, loadedPluginPaths=[{}]}}",
                       loadedPlugins.size(),
                       fmt::join(pluginPathStrings, ", "));
}

} // namespace plugin
} // namespace hipdnn_backend

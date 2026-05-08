// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstring>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <mutex>
#include <vector>

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
    if(_cachedEngineInfos.has_value())
    {
        return *_cachedEngineInfos;
    }

    std::vector<EngineInfo> infos;
    if(!_pm)
    {
        _cachedEngineInfos = infos;
        return infos;
    }

    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        auto pluginVersion = std::string(plugin->version());
        auto pluginType = std::string(::toString(plugin->type()));
        auto pluginName = std::string(plugin->name());

        auto engineIds = plugin->getAllEngineIds();
        for(const auto id : engineIds)
        {
            EngineInfo info;
            info.engineId = id;
            info.version = pluginVersion;
            info.type = pluginType;
            info.pluginName = pluginName;

            try
            {
                info.engineName = hipdnn_data_sdk::utilities::getEngineNameFromId(id);
            }
            catch(const std::out_of_range&)
            {
                info.engineName = hipdnn_data_sdk::utilities::formatEngineIdHex(id);
            }

            infos.push_back(std::move(info));
        }
    }

    std::sort(infos.begin(), infos.end(), [](const EngineInfo& a, const EngineInfo& b) {
        return a.engineName < b.engineName;
    });

    _cachedEngineInfos = infos;
    return infos;
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

        std::vector<int64_t> engineIds;
        try
        {
            engineIds = plugin->getAllEngineIds();
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_ERROR(
                "Failed to get engine IDs for plugin '{}': {}", plugin->name(), e.what());
            safeDestroyHandle(plugin.get(), handle);
            _handleToPlugin.erase(handle);
            continue;
        }

        for(const auto id : engineIds)
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

    std::vector<int64_t> engineIds;

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        auto ids = plugin->getApplicableEngineIds(handle, &serializedGraphData);
        engineIds.insert(engineIds.end(), ids.begin(), ids.end());

        for(const auto& id : ids)
        {
            if(_engineIdToHandle.find(id) == _engineIdToHandle.end())
            {
                throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Unknown engine ID");
            }

            auto existingHandle = _engineIdToHandle.at(id);
            if(existingHandle != handle)
            {
                throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                      "Engine ID " + std::to_string(id)
                                          + " is already associated with a different plugin");
            }
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
        int64_t engineId,
        const hipdnnPluginConstData_t* engineConfig,
        const hipdnnPluginConstData_t* serializedContext) const
{
    THROW_IF_NULL(engineConfig, HIPDNN_STATUS_BAD_PARAM, "Engine config cannot be null");
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

    return plugin->createExecutionContextFromSerialized(handle, engineConfig, serializedContext);
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
        const hipdnnPluginConstData_t* engineConfig,
        const hipdnnPluginConstData_t* serializedContext)
{
    return std::make_shared<EngineExecutionContextWrapper>(
        rm, engineId, engineConfig, serializedContext);
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
{
    _rm->getEngineDetails(engineId, graphDesc, &_engineDetailsData);
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(_engineDetailsData.ptr),
                                   _engineDetailsData.size);
    if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::EngineDetails>())
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "EngineDetailsWrapper: unable to verify the flatbuffer schema.");
    }
}

EngineDetailsWrapper::~EngineDetailsWrapper()
{
    if(_engineDetailsData.ptr == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroyEngineDetails(get()->engine_id(), &_engineDetailsData);
    }
    catch(const HipdnnException& e)
    {
        HIPDNN_BACKEND_LOG_ERROR(e.getMessage());
    }
}

EngineDetailsWrapper::EngineDetailsWrapper(EngineDetailsWrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engineDetailsData(other._engineDetailsData)
{
    other._rm = nullptr;
    other._engineDetailsData.ptr = nullptr;
}

EngineDetailsWrapper& EngineDetailsWrapper::operator=(EngineDetailsWrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engineDetailsData = other._engineDetailsData;

        other._rm = nullptr;
        other._engineDetailsData.ptr = nullptr;
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
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* serializedContext)
    : _rm(rm)
    , _engineId(engineId)
{
    _executionContext
        = _rm->createExecutionContextFromSerialized(engineId, engineConfig, serializedContext);
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
    other._executionContext = nullptr;
}

EngineExecutionContextWrapper&
    EngineExecutionContextWrapper::operator=(EngineExecutionContextWrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engineId = other._engineId;
        _executionContext = other._executionContext;

        other._rm = nullptr;
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

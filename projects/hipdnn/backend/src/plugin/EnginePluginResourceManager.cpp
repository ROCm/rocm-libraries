// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <string_view>
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

/// Reads the boolean override-flag attribute from a graph descriptor via the
/// public C-API getAttribute(). Defaults to false on any failure (the flag is
/// optional; older graphs do not set it).
bool readIsDynamicShapeEnabled(const GraphDescriptor& graphDesc)
{
    bool flag = false;
    int64_t elementCount = 0;
    try
    {
        graphDesc.getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
                               HIPDNN_TYPE_BOOLEAN,
                               1,
                               &elementCount,
                               &flag);
    }
    catch(const HipdnnException&)
    {
        // The optional flag was never set; treat as the default (false).
        return false;
    }
    if(elementCount < 1)
    {
        return false;
    }
    return flag;
}

/// Computes the minimum plugin SDK API version required to serve a given
/// graph. Graphs that opt in to RFC 0008 Phase 1 dynamic shapes require
/// `K_PHASE1_OVERRIDE_MIN_VERSION`; all other graphs accept any plugin at or
/// above the post-PR1 baseline `"1.0.0"`.
///
/// Per RFC 0008 §4.5, the `"1.1.0"` constant is a placeholder that may be
/// renamed before any non-fake provider implements override execute. The
/// constant is centralized in `<hipdnn_plugin_sdk/PluginVersionConstants.hpp>`
/// so the version filter, fake plugins, and tests share a single source of
/// truth.
///
/// Both candidate `Version` values are parsed exactly once at first call
/// (function-local `static const`) so the dispatch hot path does not re-parse
/// constant strings per plugin per call.
const hipdnn_data_sdk::utilities::Version&
    computeMinimumPluginApiVersion(const GraphDescriptor& graphDesc)
{
    static const hipdnn_data_sdk::utilities::Version sBaselineVersion{std::string_view{"1.0.0"}};
    static const hipdnn_data_sdk::utilities::Version sPhase1OverrideMinVersion{
        hipdnn_plugin_sdk::K_PHASE1_OVERRIDE_MIN_VERSION};

    if(readIsDynamicShapeEnabled(graphDesc))
    {
        return sPhase1OverrideMinVersion;
    }
    return sBaselineVersion;
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

    // RFC 0008 Phase 1 applicability filter: graphs that opt in to overridable
    // tensor shapes require plugins exporting at least the override-execute
    // SDK surface. Older plugins are silently skipped here so they remain
    // usable for non-override graphs.
    const auto& requiredVersion = computeMinimumPluginApiVersion(*graphDesc);

    std::vector<int64_t> engineIds;

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        // Safe to deref: validateBeforeAdding rejected plugins with
        // unparseable versions at load time, and the value is parsed once
        // and cached on the plugin instance so the dispatch hot path does
        // not re-parse the version string per plugin per call.
        const auto& pluginVersion = *plugin->parsedApiVersion();
        if(pluginVersion < requiredVersion)
        {
            HIPDNN_BACKEND_LOG_INFO(
                "Skipping plugin '{}' (apiVersion={}) for graph requiring at least {}",
                plugin->cachedName(),
                pluginVersion.str(),
                requiredVersion.str());
            continue;
        }

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

    // RFC 0008 Phase 1 dispatch switch: if the variant pack carries override
    // tensors, route through the override-aware execute entry; otherwise
    // preserve the existing dispatch path.
    const auto& overrideUniqueIds = variantPackDesc->getOverrideUniqueIds();
    const auto& overrideShapesFlat = variantPackDesc->getOverrideShapes();
    const auto& overrideStridesFlat = variantPackDesc->getOverrideStrides();
    const auto& overrideLengths64 = variantPackDesc->getOverrideLengths();

    const bool hasOverrides = !overrideUniqueIds.empty();

    if(hasOverrides)
    {
        // Sanity-check the variant-pack invariant before doing any narrowing.
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

        // RFC 0008 §4.6 safety net: a plugin that reported a recent enough API
        // version but failed to export the override symbol is rejected here
        // rather than silently falling back to the existing entry (which would
        // ignore the user's override request).
        THROW_IF_FALSE(plugin->hasOverrideExecute(),
                       HIPDNN_STATUS_NOT_SUPPORTED,
                       "Selected plugin does not export "
                       "hipdnnEnginePluginExecuteOpGraphWithOverrides although the variant pack "
                       "carries override-tensor selectors (RFC 0008 §4.6).");

        // D1 narrowing: variant-pack stores int64; SDK surface is uint32.
        // Each rank is bounded by structural tensor constants and cannot
        // legitimately exceed uint32 limits, but -Wconversion requires the
        // explicit cast.
        const auto numOverridesSize = overrideUniqueIds.size();
        std::vector<uint32_t> overrideLengthsU32;
        overrideLengthsU32.reserve(numOverridesSize);
        for(const auto value : overrideLengths64)
        {
            assert(value >= 0
                   && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max());
            overrideLengthsU32.push_back(static_cast<uint32_t>(value));
        }

        // D2 ptr-array reconstruction: build per-UID pointers into the
        // contiguous flat shape/stride buffers using a prefix-sum walk over
        // the per-UID rank. Both arrays live on the dispatch stack frame for
        // the duration of the call; their pointee data is owned by the
        // variant pack.
        std::vector<const int64_t*> overrideShapesPerUid;
        overrideShapesPerUid.reserve(numOverridesSize);
        std::vector<const int64_t*> overrideStridesPerUid;
        overrideStridesPerUid.reserve(numOverridesSize);

        size_t expectedFlatTotal = 0;
        for(const auto rank : overrideLengthsU32)
        {
            expectedFlatTotal += static_cast<size_t>(rank);
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

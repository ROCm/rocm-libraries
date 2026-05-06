// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendEnumStringUtils.hpp"
#include "Helpers.hpp"
#include "HipdnnException.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/DescriptorFactory.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include "handle/Handle.hpp"
#include "handle/HandleFactory.hpp"
#include "hipdnn_backend.h"
#include "logging/Logging.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <hipdnn_backend/version.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/FunctionNameMacro.hpp>

#include <cstring>

using namespace hipdnn_backend;

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_BACKEND_LOG_INFO("API called: [{}] " format, __func__, __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_BACKEND_LOG_INFO("API success: [{}] " format, func_name, __VA_ARGS__)

namespace
{
void throwIfInvalidDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    if(descriptor == nullptr)
    {
        throw hipdnn_backend::HipdnnException(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                                              "hipdnnBackendDescriptor_t is nullptr");
    }

    if(!descriptor->isValid())
    {
        throw hipdnn_backend::HipdnnException(
            HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
            "hipdnnBackendDescriptor_t private_descriptor is nullptr");
    }
}

template <typename T>
void throwIfNull(T* value)
{
    if(value == nullptr)
    {
        throw hipdnn_backend::HipdnnException(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                                              "Null pointer provided to "
                                                  + std::string(HIPDNN_FUNCTION_NAME));
    }
}
} // namespace

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        hipdnn_backend::HandleFactory::createHandle(handle);

        LOG_API_SUCCESS(apiName, "createHandle={}", logPtr(*handle));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    LOG_API_ENTRY("handle={}", logPtr(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        delete handle;

        LOG_API_SUCCESS(apiName, "handle destroyed", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    LOG_API_ENTRY("handle={}, streamId={:p}", logPtr(handle), static_cast<void*>(streamId));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        handle->setStream(streamId);
        hipdnn_backend::logging::logHipDeviceInfo(streamId);

        LOG_API_SUCCESS(apiName, "stream={:p}", static_cast<void*>(streamId));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    LOG_API_ENTRY("handle={}, streamId_ptr={:p}", logPtr(handle), static_cast<void*>(streamId));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(streamId);

        *streamId = handle->getStream();

        LOG_API_SUCCESS(apiName, "retrieved_stream={:p}", static_cast<void*>(*streamId));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptorType, hipdnnBackendDescriptor_t* descriptor)
{
    LOG_API_ENTRY(
        "descriptorType={}, descriptor_ptr={:p}", descriptorType, static_cast<void*>(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::create(descriptorType, descriptor);

        LOG_API_SUCCESS(apiName, "created_descriptor={}", logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={}", logPtr(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        hipdnn_backend::DescriptorFactory::destroy(descriptor);

        LOG_API_SUCCESS(apiName, "descriptor destroyed", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t executionPlan,
                                                          hipdnnBackendDescriptor_t variantPack)
{
    LOG_API_ENTRY("handle={}, executionPlan={}, variantPack={}",
                  logPtr(handle),
                  logPtr(executionPlan),
                  logPtr(variantPack));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfInvalidDescriptor(executionPlan);
        throwIfInvalidDescriptor(variantPack);

        handle->getPluginResourceManager()->executeOpGraph(executionPlan, variantPack);

        LOG_API_SUCCESS(apiName, "executionPlan={}", logPtr(executionPlan));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={}", logPtr(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->finalize();

        LOG_API_SUCCESS(apiName, "descriptor={}", logPtr(descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements)
{
    LOG_API_ENTRY("descriptor={}, attributeName={}, attributeType={}, "
                  "requestedElementCount={}, elementCount_ptr={:p}, arrayOfElements_ptr={:p}",
                  logPtr(descriptor),
                  attributeName,
                  attributeType,
                  requestedElementCount,
                  static_cast<void*>(elementCount),
                  arrayOfElements);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->getAttribute(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

        if(elementCount == nullptr)
        {
            LOG_API_SUCCESS(apiName, "status={}, elementCount_ptr=nullptr", HIPDNN_STATUS_SUCCESS);
        }
        else
        {
            LOG_API_SUCCESS(apiName,
                            "status={}, retrieved_elementCount={}",
                            HIPDNN_STATUS_SUCCESS,
                            *elementCount);
        }
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements)
{
    LOG_API_ENTRY("descriptor={}, attributeName={}, attributeType={}, "
                  "elementCount={}, arrayOfElements_ptr={:p}",
                  logPtr(descriptor),
                  attributeName,
                  attributeType,
                  elementCount,
                  arrayOfElements);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->setAttribute(attributeName, attributeType, elementCount, arrayOfElements);

        LOG_API_SUCCESS(apiName, "status={}", HIPDNN_STATUS_SUCCESS);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph, size_t graphByteSize)
{
    LOG_API_ENTRY("descriptor_ptr={:p}, serializedGraph_ptr={:p}, graphByteSize={}",
                  static_cast<void*>(descriptor),
                  static_cast<const void*>(serializedGraph),
                  graphByteSize);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::createGraphExt(
            descriptor, serializedGraph, graphByteSize);

        LOG_API_SUCCESS(apiName, "created_descriptor={}", logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedBinaryGraph_ext(hipdnnBackendDescriptor_t descriptor,
                                              size_t requestedByteSize,
                                              size_t* graphByteSize,
                                              uint8_t* serializedGraph)
{
    LOG_API_ENTRY("descriptor={}, requestedByteSize={}, graphByteSize_ptr={:p}, "
                  "serializedGraph_ptr={:p}",
                  logPtr(descriptor),
                  requestedByteSize,
                  static_cast<void*>(graphByteSize),
                  static_cast<void*>(serializedGraph));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);
        throwIfNull(graphByteSize);

        auto graphDesc = descriptor->asDescriptor<hipdnn_backend::GraphDescriptor>();
        graphDesc->buildSerializedGraph();
        auto data = graphDesc->getSerializedGraph();

        *graphByteSize = data.size;

        if(serializedGraph != nullptr)
        {
            THROW_IF_LT(requestedByteSize,
                        data.size,
                        HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
                        "Requested buffer size (" + std::to_string(requestedByteSize)
                            + ") is smaller than the serialized graph size ("
                            + std::to_string(data.size) + ")");
            std::memcpy(serializedGraph, data.ptr, data.size);
        }

        LOG_API_SUCCESS(apiName, "graphByteSize={}", *graphByteSize);
    });
}

// NOTE: The JSON serialization path is slower than the binary path. Each call
// reconstructs the JSON from the cached binary buffer (binary -> unpack -> JSON).
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedJsonGraph_ext(hipdnnBackendDescriptor_t descriptor,
                                            size_t requestedByteSize,
                                            size_t* graphByteSize,
                                            char* serializedJsonGraph)
{
    LOG_API_ENTRY("descriptor={}, requestedByteSize={}, graphByteSize_ptr={:p}, "
                  "serializedJsonGraph_ptr={:p}",
                  logPtr(descriptor),
                  requestedByteSize,
                  static_cast<void*>(graphByteSize),
                  static_cast<void*>(serializedJsonGraph));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);
        throwIfNull(graphByteSize);

        auto graphDesc = descriptor->asDescriptor<hipdnn_backend::GraphDescriptor>();
        // buildSerializedGraph() populates the cached binary buffer (no-op if already cached).
        // getSerializedJsonGraph() then converts binary -> JSON on each call (not cached).
        graphDesc->buildSerializedGraph();
        auto jsonStr = graphDesc->getSerializedJsonGraph();

        *graphByteSize = jsonStr.size() + 1;

        if(serializedJsonGraph != nullptr)
        {
            THROW_IF_LT(requestedByteSize,
                        jsonStr.size() + 1,
                        HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
                        "Requested buffer size (" + std::to_string(requestedByteSize)
                            + ") is smaller than the JSON graph size ("
                            + std::to_string(jsonStr.size() + 1) + ")");
            std::memcpy(serializedJsonGraph, jsonStr.c_str(), jsonStr.size() + 1);
        }

        LOG_API_SUCCESS(apiName, "graphByteSize={}", *graphByteSize);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedExecutionPlan_ext(hipdnnBackendDescriptor_t descriptor,
                                                size_t requestedByteSize,
                                                size_t* planByteSize,
                                                uint8_t* serializedPlan)
{
    LOG_API_ENTRY("descriptor={}, requestedByteSize={}, planByteSize_ptr={:p}, "
                  "serializedPlan_ptr={:p}",
                  logPtr(descriptor),
                  requestedByteSize,
                  static_cast<void*>(planByteSize),
                  static_cast<void*>(serializedPlan));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        auto executionPlanDesc
            = descriptor->asDescriptor<hipdnn_backend::ExecutionPlanDescriptor>();
        executionPlanDesc->serializeBackendPlan(requestedByteSize, planByteSize, serializedPlan);

        LOG_API_SUCCESS(apiName, "planByteSize={}", *planByteSize);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendCreateAndDeserializeExecutionPlan_ext(hipdnnHandle_t handle,
                                                       hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedPlan,
                                                       size_t planByteSize)
{
    LOG_API_ENTRY("handle={}, descriptor_ptr={:p}, serializedPlan_ptr={:p}, planByteSize={}",
                  logPtr(handle),
                  static_cast<void*>(descriptor),
                  static_cast<const void*>(serializedPlan),
                  planByteSize);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(descriptor);

        auto executionPlanDesc = std::make_shared<hipdnn_backend::ExecutionPlanDescriptor>();
        executionPlanDesc->deserializeBackendPlan(
            handle->getPluginResourceManager(), serializedPlan, planByteSize);
        *descriptor = HipdnnBackendDescriptor::packDescriptor(executionPlanDesc);

        LOG_API_SUCCESS(apiName, "created_descriptor={}", logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeJsonGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const char* jsonGraph, size_t jsonByteSize)
{
    LOG_API_ENTRY("descriptor_ptr={:p}, jsonGraph_ptr={:p}, jsonByteSize={}",
                  static_cast<void*>(descriptor),
                  static_cast<const void*>(jsonGraph),
                  jsonByteSize);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::createGraphFromJsonExt(
            descriptor, jsonGraph, jsonByteSize);

        LOG_API_SUCCESS(apiName, "created_descriptor={}", logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT const char* hipdnnGetErrorString(hipdnnStatus_t status)
{
    LOG_API_ENTRY("status={}", status);

    return hipdnn_backend::hipdnnGetStatusString(status);
}

HIPDNN_BACKEND_EXPORT void hipdnnGetLastErrorString(char* message, size_t maxSize)
{
    LOG_API_ENTRY("message_ptr={:p}, maxSize={}", static_cast<void*>(message), maxSize);
    // Ignore status since API doesn't return it.
    // We still want to catch and log if the user provides incorrect parameters.
    auto _ = hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(message);

        if(maxSize == 0)
        {
            throw hipdnn_backend::HipdnnException(HIPDNN_STATUS_BAD_PARAM, "maxSize is 0");
        }

        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            message, hipdnn_backend::LastErrorManager::getLastError(), maxSize);

        // Clear the error after retrieval
        hipdnn_backend::LastErrorManager::clearLastError();

        LOG_API_SUCCESS(apiName, "set_error_message={:p}", static_cast<void*>(message));
    });
}

HIPDNN_BACKEND_EXPORT void hipdnnPeekLastErrorString_ext(char* message, size_t maxSize)
{
    LOG_API_ENTRY("message_ptr={:p}, maxSize={}", static_cast<void*>(message), maxSize);
    // Ignore status since API doesn't return it.
    // We still want to catch and log if the user provides incorrect parameters.
    auto _ = hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(message);

        if(maxSize == 0)
        {
            throw hipdnn_backend::HipdnnException(HIPDNN_STATUS_BAD_PARAM, "maxSize is 0");
        }

        // Get the error without clearing it (preserves pre-existing behavior)
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            message, hipdnn_backend::LastErrorManager::getLastError(), maxSize);

        LOG_API_SUCCESS(apiName, "set_error_message={:p}", static_cast<void*>(message));
    });
}

HIPDNN_BACKEND_EXPORT void hipdnnLoggingCallback_ext(hipdnnSeverity_t severity, const char* msg)
{
    hipdnn_backend::logging::backendLoggingCallback(severity, msg);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t numPaths, const char* const* pluginPaths, hipdnnPluginLoadingMode_ext_t loadingMode)
{
    LOG_API_ENTRY("numPaths={}, pluginPaths_ptr={:p}, loadingMode={}",
                  numPaths,
                  static_cast<const void*>(pluginPaths),
                  loadingMode);

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        if(numPaths > 0)
        {
            throwIfNull(pluginPaths);
        }

        std::vector<std::filesystem::path> pathsVec;
        pathsVec.reserve(numPaths);

        for(size_t i = 0; i < numPaths; ++i)
        {
            throwIfNull(pluginPaths[i]);
            pathsVec.emplace_back(pluginPaths[i]);
        }

        hipdnn_backend::plugin::EnginePluginResourceManager::setPluginPaths(pathsVec, loadingMode);
        LOG_API_SUCCESS(apiName, "set_plugin_paths={}", loadingMode);
        return HIPDNN_STATUS_SUCCESS;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnSetPluginUnloadMode_ext(hipdnnPluginUnloadingMode_ext_t unloadingMode)
{
    LOG_API_ENTRY("unloadingMode={}", unloadingMode);

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        hipdnn_backend::plugin::EnginePluginResourceManager::setPluginUnloadingMode(unloadingMode);
        LOG_API_SUCCESS(apiName, "set_plugin_unloading_mode={}", unloadingMode);
        return HIPDNN_STATUS_SUCCESS;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(hipdnnHandle_t handle,
                                                                          size_t* numPluginPaths,
                                                                          char** pluginPaths,
                                                                          size_t* maxStringLen)
{
    LOG_API_ENTRY(
        "handle={:p}, numPluginPaths_ptr={:p}, pluginPaths_ptr={:p}, maxStringLen_ptr={:p}",
        static_cast<void*>(handle),
        static_cast<void*>(numPluginPaths),
        static_cast<const void*>(pluginPaths),
        static_cast<void*>(maxStringLen));

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(handle);
        throwIfNull(numPluginPaths);
        throwIfNull(maxStringLen);

        handle->getPluginResourceManager()->getLoadedPluginFiles(
            numPluginPaths, pluginPaths, maxStringLen);

        LOG_API_SUCCESS(apiName,
                        "retrieved_numPluginPaths={}, retrieved_maxStringLen={}",
                        *numPluginPaths,
                        *maxStringLen);
    });
}

// NOLINTBEGIN(readability-identifier-naming) - C API function
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnSetUserLogCallback_ext(hipdnnUserLogCallback_t callback,
                                 hipdnnSeverity_t minLevel,
                                 hipdnnLogCallbackMode_t mode,
                                 hipdnnUserLogCallbackHandle_t userHandle)
// NOLINTEND(readability-identifier-naming)
{
    return hipdnn_backend::logging::setUserLogCallback(callback, minLevel, mode, userHandle);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendSetGlobalLogLevel_ext(hipdnnSeverity_t level)
{
    // Validate log level
    if(level != HIPDNN_SEV_INFO && level != HIPDNN_SEV_WARN && level != HIPDNN_SEV_ERROR
       && level != HIPDNN_SEV_FATAL && level != HIPDNN_SEV_OFF)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    return hipdnn_backend::logging::setGlobalLogLevel(level);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendGetGlobalLogLevel_ext(hipdnnSeverity_t* level)
{
    if(level == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    return hipdnn_backend::logging::getGlobalLogLevel(*level);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetEngineCount_ext(hipdnnHandle_t handle,
                                                              size_t* numEngines)
{
    LOG_API_ENTRY("handle={:p}, numEngines_ptr={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(numEngines));

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(handle);
        throwIfNull(numEngines);

        *numEngines = handle->getEngineCount();

        LOG_API_SUCCESS(apiName, "retrieved_numEngines={}", *numEngines);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetEngineInfo_ext(hipdnnHandle_t handle,
                                                             size_t engineIndex,
                                                             int64_t* engineId,
                                                             char* engineName,
                                                             size_t* engineNameLen,
                                                             char* pluginName,
                                                             size_t* pluginNameLen,
                                                             char* version,
                                                             size_t* versionLen,
                                                             char* type,
                                                             size_t* typeLen)
{
    LOG_API_ENTRY("handle={:p}, engineIndex={}, engineId_ptr={:p}, engineName_ptr={:p}, "
                  "pluginName_ptr={:p}, version_ptr={:p}, type_ptr={:p}",
                  static_cast<void*>(handle),
                  engineIndex,
                  static_cast<void*>(engineId),
                  static_cast<void*>(engineName),
                  static_cast<void*>(pluginName),
                  static_cast<void*>(version),
                  static_cast<void*>(type));

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(handle);
        throwIfNull(engineNameLen);
        throwIfNull(pluginNameLen);
        throwIfNull(versionLen);
        throwIfNull(typeLen);

        auto infos = handle->getEngineInfos();
        if(engineIndex >= infos.size())
        {
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                                  "Engine index " + std::to_string(engineIndex) + " out of range ("
                                      + std::to_string(infos.size()) + " engines loaded).");
        }

        const auto& info = infos[engineIndex];

        if(engineId != nullptr)
        {
            *engineId = info.engineId;
        }

        const size_t requiredEngineNameLen = info.engineName.size() + 1;
        const size_t requiredPluginNameLen = info.pluginName.size() + 1;
        const size_t requiredVersionLen = info.version.size() + 1;
        const size_t requiredTypeLen = info.type.size() + 1;

        if(engineName == nullptr || pluginName == nullptr || version == nullptr || type == nullptr)
        {
            *engineNameLen = requiredEngineNameLen;
            *pluginNameLen = requiredPluginNameLen;
            *versionLen = requiredVersionLen;
            *typeLen = requiredTypeLen;
            return;
        }

        if(*engineNameLen < requiredEngineNameLen || *pluginNameLen < requiredPluginNameLen
           || *versionLen < requiredVersionLen || *typeLen < requiredTypeLen)
        {
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Insufficient buffer space provided.");
        }

        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            engineName, info.engineName.c_str(), *engineNameLen);
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            pluginName, info.pluginName.c_str(), *pluginNameLen);
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            version, info.version.c_str(), *versionLen);
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            type, info.type.c_str(), *typeLen);

        LOG_API_SUCCESS(apiName,
                        "engine[{}]: engineId={}, engineName={}, pluginName={}, version={}, "
                        "type={}",
                        engineIndex,
                        info.engineId,
                        info.engineName,
                        info.pluginName,
                        info.version,
                        info.type);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetVersion_ext(const char** version)
{
    return hipdnn_backend::tryCatch([&]() {
        throwIfNull(version);
        *version = HIPDNN_BACKEND_VERSION_STRING;
    });
}

HIPDNN_BACKEND_EXPORT const char* hipdnnVersionString_ext()
{
    return HIPDNN_BACKEND_VERSION_STRING;
}

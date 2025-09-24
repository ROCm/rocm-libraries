// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendEnumStringUtils.hpp"
#include "Helpers.hpp"
#include "HipdnnException.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/DescriptorFactory.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include "handle/Handle.hpp"
#include "handle/HandleFactory.hpp"
#include "hipdnn_backend.h"
#include "logging/Logging.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/utilities/StringUtil.hpp>

using namespace hipdnn_backend;

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__, __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name, __VA_ARGS__)

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
                                              std::string(typeid(T).name()) + " is nullptr");
    }
}
} // namespace

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        hipdnn_backend::HandleFactory::createHandle(handle);

        LOG_API_SUCCESS(apiName, "createHandle={:p}", static_cast<void*>(*handle));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        delete handle;

        LOG_API_SUCCESS(apiName, "", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    LOG_API_ENTRY(
        "handle={:p}, streamId={:p}", static_cast<void*>(handle), static_cast<void*>(streamId));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        handle->setStream(streamId);

        LOG_API_SUCCESS(apiName, "", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    LOG_API_ENTRY(
        "handle={:p}, streamId_ptr={:p}", static_cast<void*>(handle), static_cast<void*>(streamId));

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
    LOG_API_ENTRY("descriptorType={}, descriptor_ptr={:p}",
                  hipdnn_backend::hipdnnGetBackendDescriptorTypeName(descriptorType),
                  static_cast<void*>(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::create(descriptorType, descriptor);

        LOG_API_SUCCESS(apiName, "created_descriptor={:p}", static_cast<void*>(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={:p}", static_cast<void*>(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        hipdnn_backend::DescriptorFactory::destroy(descriptor);

        LOG_API_SUCCESS(apiName, "", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t executionPlan,
                                                          hipdnnBackendDescriptor_t variantPack)
{
    LOG_API_ENTRY("handle={:p}, executionPlan={:p}, variantPack={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionPlan),
                  static_cast<void*>(variantPack));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfInvalidDescriptor(executionPlan);
        throwIfInvalidDescriptor(variantPack);

        handle->getPluginResourceManager()->executeOpGraph(executionPlan, variantPack);

        LOG_API_SUCCESS(apiName, "", "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={:p}", static_cast<void*>(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->finalize();

        LOG_API_SUCCESS(apiName, "", "");
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
    LOG_API_ENTRY("descriptor={:p}, attributeName={}, attributeType={}, "
                  "requestedElementCount={}, elementCount_ptr={:p}, arrayOfElements_ptr={:p}",
                  static_cast<void*>(descriptor),
                  hipdnn_backend::hipdnnGetAttributeNameString(attributeName),
                  hipdnn_backend::hipdnnGetAttributeTypeString(attributeType),
                  requestedElementCount,
                  static_cast<void*>(elementCount),
                  arrayOfElements);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->getAttribute(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

        if(elementCount == nullptr)
        {
            LOG_API_SUCCESS(apiName,
                            "status={}, elementCount_ptr=nullptr",
                            hipdnn_backend::hipdnnGetStatusString(HIPDNN_STATUS_SUCCESS));
        }
        else
        {
            LOG_API_SUCCESS(apiName,
                            "status={}, retrieved_elementCount={}",
                            hipdnn_backend::hipdnnGetStatusString(HIPDNN_STATUS_SUCCESS),
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
    LOG_API_ENTRY("descriptor={:p}, attributeName={}, attributeType={}, "
                  "elementCount={}, arrayOfElements_ptr={:p}",
                  static_cast<void*>(descriptor),
                  hipdnn_backend::hipdnnGetAttributeNameString(attributeName),
                  hipdnn_backend::hipdnnGetAttributeTypeString(attributeType),
                  elementCount,
                  arrayOfElements);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->setAttribute(attributeName, attributeType, elementCount, arrayOfElements);

        LOG_API_SUCCESS(
            apiName, "status={}", hipdnn_backend::hipdnnGetStatusString(HIPDNN_STATUS_SUCCESS));
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

        LOG_API_SUCCESS(apiName, "created_descriptor={:p}", static_cast<void*>(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT const char* hipdnnGetErrorString(hipdnnStatus_t status)
{
    LOG_API_ENTRY("status={}", hipdnn_backend::hipdnnGetStatusString(status));

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

        hipdnn_sdk::utilities::copyMaxSizeWithNullTerminator(
            message, hipdnn_backend::LastErrorManager::getLastError(), maxSize);

        LOG_API_SUCCESS(apiName, "set_error_message={:p}", static_cast<void*>(message));
    });
}

HIPDNN_BACKEND_EXPORT void hipdnnLoggingCallback_ext(hipdnnSeverity_t severity, const char* msg)
{
    hipdnn_backend::logging::hipdnnLoggingCallback(severity, msg);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t numPaths, const char* const* pluginPaths, hipdnnPluginLoadingMode_ext_t loadingMode)
{
    LOG_API_ENTRY("numPaths={}, pluginPaths_ptr={:p}, loadingMode={}",
                  numPaths,
                  static_cast<const void*>(pluginPaths),
                  hipdnn_backend::hipdnnGetPluginLoadingModeString(loadingMode));

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
        // TODO: automatic formatting loading mode to string
        LOG_API_SUCCESS(apiName,
                        "set_plugin_paths={}",
                        hipdnn_backend::hipdnnGetPluginLoadingModeString(loadingMode));
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

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

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include "utilities/PointerToString.hpp"

using namespace hipdnn_backend;

#define LOG_API_ENTRY(msg) HIPDNN_BACKEND_LOG_INFO("API called: [" << __func__ << "] " << msg)

#define LOG_API_SUCCESS(func_name, msg) \
    HIPDNN_BACKEND_LOG_INFO("API success: [" << func_name << "] " << msg)

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
    LOG_API_ENTRY("handle_ptr=" << ptrToString(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        hipdnn_backend::HandleFactory::createHandle(handle);

        LOG_API_SUCCESS(apiName, "createHandle=" << logPtr(*handle));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    LOG_API_ENTRY("handle=" << logPtr(handle));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        delete handle;

        LOG_API_SUCCESS(apiName, "handle destroyed");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    LOG_API_ENTRY("handle=" << logPtr(handle) << ", streamId=" << ptrToString(streamId));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        handle->setStream(streamId);
        hipdnn_backend::logging::logHipDeviceInfo(streamId);

        LOG_API_SUCCESS(apiName, "stream=" << ptrToString(streamId));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    LOG_API_ENTRY("handle=" << logPtr(handle) << ", streamId_ptr=" << ptrToString(streamId));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(streamId);

        *streamId = handle->getStream();

        LOG_API_SUCCESS(apiName, "retrieved_stream=" << ptrToString(*streamId));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptorType, hipdnnBackendDescriptor_t* descriptor)
{
    LOG_API_ENTRY("descriptorType=" << descriptorType
                                    << ", descriptor_ptr=" << ptrToString(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::create(descriptorType, descriptor);

        LOG_API_SUCCESS(apiName, "created_descriptor=" << logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor=" << logPtr(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        hipdnn_backend::DescriptorFactory::destroy(descriptor);

        LOG_API_SUCCESS(apiName, "descriptor destroyed");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t executionPlan,
                                                          hipdnnBackendDescriptor_t variantPack)
{
    LOG_API_ENTRY("handle=" << logPtr(handle) << ", executionPlan=" << logPtr(executionPlan)
                            << ", variantPack=" << logPtr(variantPack));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfInvalidDescriptor(executionPlan);
        throwIfInvalidDescriptor(variantPack);

        handle->getPluginResourceManager()->executeOpGraph(executionPlan, variantPack);

        LOG_API_SUCCESS(apiName, "executionPlan=" << logPtr(executionPlan));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor=" << logPtr(descriptor));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->finalize();

        LOG_API_SUCCESS(apiName, "descriptor=" << logPtr(descriptor));
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
    LOG_API_ENTRY("descriptor=" << logPtr(descriptor) << ", attributeName=" << attributeName
                                << ", attributeType=" << attributeType
                                << ", requestedElementCount=" << requestedElementCount
                                << ", elementCount_ptr=" << ptrToString(elementCount)
                                << ", arrayOfElements_ptr=" << ptrToString(arrayOfElements));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->getAttribute(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

        if(elementCount == nullptr)
        {
            LOG_API_SUCCESS(apiName,
                            "status=" << HIPDNN_STATUS_SUCCESS << ", elementCount_ptr=nullptr");
        }
        else
        {
            LOG_API_SUCCESS(apiName,
                            "status=" << HIPDNN_STATUS_SUCCESS
                                      << ", retrieved_elementCount=" << *elementCount);
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
    LOG_API_ENTRY("descriptor=" << logPtr(descriptor) << ", attributeName=" << attributeName
                                << ", attributeType=" << attributeType
                                << ", elementCount=" << elementCount
                                << ", arrayOfElements_ptr=" << ptrToString(arrayOfElements));

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        throwIfInvalidDescriptor(descriptor);

        descriptor->setAttribute(attributeName, attributeType, elementCount, arrayOfElements);

        LOG_API_SUCCESS(apiName, "status=" << HIPDNN_STATUS_SUCCESS);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph, size_t graphByteSize)
{
    LOG_API_ENTRY("descriptor_ptr=" << ptrToString(descriptor)
                                    << ", serializedGraph_ptr=" << ptrToString(serializedGraph)
                                    << ", graphByteSize=" << graphByteSize);

    return hipdnn_backend::tryCatch([&, apiName = __func__]() {
        hipdnn_backend::DescriptorFactory::createGraphExt(
            descriptor, serializedGraph, graphByteSize);

        LOG_API_SUCCESS(apiName, "created_descriptor=" << logPtr(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT const char* hipdnnGetErrorString(hipdnnStatus_t status)
{
    LOG_API_ENTRY("status=" << status);

    return hipdnn_backend::hipdnnGetStatusString(status);
}

HIPDNN_BACKEND_EXPORT void hipdnnGetLastErrorString(char* message, size_t maxSize)
{
    LOG_API_ENTRY("message_ptr=" << ptrToString(message) << ", maxSize=" << maxSize);
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

        LOG_API_SUCCESS(apiName, "set_error_message=" << ptrToString(message));
    });
}

HIPDNN_BACKEND_EXPORT void hipdnnPeekLastErrorString_ext(char* message, size_t maxSize)
{
    LOG_API_ENTRY("message_ptr=" << ptrToString(message) << ", maxSize=" << maxSize);
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

        LOG_API_SUCCESS(apiName, "set_error_message=" << ptrToString(message));
    });
}

HIPDNN_BACKEND_EXPORT void hipdnnLoggingCallback_ext(hipdnnSeverity_t severity, const char* msg)
{
    hipdnn_backend::logging::hipdnnLoggingCallback(severity, msg);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t numPaths, const char* const* pluginPaths, hipdnnPluginLoadingMode_ext_t loadingMode)
{
    LOG_API_ENTRY("numPaths=" << numPaths << ", pluginPaths_ptr=" << ptrToString(pluginPaths)
                              << ", loadingMode=" << loadingMode);

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
        LOG_API_SUCCESS(apiName, "set_plugin_paths=" << loadingMode);
        return HIPDNN_STATUS_SUCCESS;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(hipdnnHandle_t handle,
                                                                          size_t* numPluginPaths,
                                                                          char** pluginPaths,
                                                                          size_t* maxStringLen)
{
    LOG_API_ENTRY("handle=" << ptrToString(handle)
                            << ", numPluginPaths_ptr=" << ptrToString(numPluginPaths)
                            << ", pluginPaths_ptr=" << ptrToString(pluginPaths)
                            << ", maxStringLen_ptr=" << ptrToString(maxStringLen));

    return hipdnn_backend::tryCatch([&, apiName = __func__] {
        throwIfNull(handle);
        throwIfNull(numPluginPaths);
        throwIfNull(maxStringLen);

        handle->getPluginResourceManager()->getLoadedPluginFiles(
            numPluginPaths, pluginPaths, maxStringLen);

        LOG_API_SUCCESS(apiName,
                        "retrieved_numPluginPaths="
                            << *numPluginPaths << ", retrieved_maxStringLen=" << *maxStringLen);
    });
}

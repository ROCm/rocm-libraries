// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file HipdnnDynamicBackendWrapper.hpp
 * @brief IHipdnnBackend implementation that resolves the backend at runtime.
 *
 * Counterpart to @ref HipdnnDirectBackendWrapper: instead of calling the backend
 * C API directly (which would link `libhipdnn_backend.so`), each entry point is
 * resolved via `dlopen`/`dlsym` at construction time and cached as a typed
 * function pointer. Selected by the frontend when built with
 * @ref HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND.
 *
 * `decltype(&symbol)` is unevaluated — it does not create a link-time reference.
 */

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

#ifdef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

namespace hipdnn_frontend::detail
{

using SymbolResolver = void* (*)(const char*);

class HipdnnDynamicBackendWrapper : public IHipdnnBackend
{
public:
    explicit HipdnnDynamicBackendWrapper(hipdnn_data_sdk::utilities::Version version,
                                         SymbolResolver symbolResolver = resolveSymbol)
        : _version(version)
        , _symbolResolver(symbolResolver == nullptr ? resolveSymbol : symbolResolver)
    {
        resolveAllSymbols();
    }

    hipdnnStatus_t create(hipdnnHandle_t* handle) override
    {
        return _create != nullptr ? _create(handle) : missingSymbolStatus();
    }

    hipdnnStatus_t destroy(hipdnnHandle_t handle) override
    {
        return _destroy != nullptr ? _destroy(handle) : missingSymbolStatus();
    }

    hipdnnStatus_t setStream(hipdnnHandle_t handle, hipStream_t streamId) override
    {
        return _setStream != nullptr ? _setStream(handle, streamId) : missingSymbolStatus();
    }

    hipdnnStatus_t getStream(hipdnnHandle_t handle, hipStream_t* streamId) override
    {
        return _getStream != nullptr ? _getStream(handle, streamId) : missingSymbolStatus();
    }

    hipdnnStatus_t backendCreateDescriptor(hipdnnBackendDescriptorType_t descriptorType,
                                           hipdnnBackendDescriptor_t* descriptor) override
    {
        return _backendCreateDescriptor != nullptr
                   ? _backendCreateDescriptor(descriptorType, descriptor)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor) override
    {
        return _backendDestroyDescriptor != nullptr ? _backendDestroyDescriptor(descriptor)
                                                    : missingSymbolStatus();
    }

    hipdnnStatus_t backendExecute(hipdnnHandle_t handle,
                                  hipdnnBackendDescriptor_t executionPlan,
                                  hipdnnBackendDescriptor_t variantPack) override
    {
        return _backendExecute != nullptr ? _backendExecute(handle, executionPlan, variantPack)
                                          : missingSymbolStatus();
    }

    hipdnnStatus_t backendFinalize(hipdnnBackendDescriptor_t descriptor) override
    {
        return _backendFinalize != nullptr ? _backendFinalize(descriptor) : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) override
    {
        return _backendGetAttribute != nullptr ? _backendGetAttribute(descriptor,
                                                                      attributeName,
                                                                      attributeType,
                                                                      requestedElementCount,
                                                                      elementCount,
                                                                      arrayOfElements)
                                               : missingSymbolStatus();
    }

    hipdnnStatus_t backendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements) override
    {
        return _backendSetAttribute != nullptr
                   ? _backendSetAttribute(
                         descriptor, attributeName, attributeType, elementCount, arrayOfElements)
                   : missingSymbolStatus();
    }

    const char* getErrorString(hipdnnStatus_t status) override
    {
        if(status == HIPDNN_STATUS_VERSION_MISMATCH)
        {
            return "HIPDNN_STATUS_VERSION_MISMATCH";
        }
        return _getErrorString != nullptr ? _getErrorString(status) : "";
    }

    void getLastErrorString(char* message, size_t maxSize) override
    {
        if(_getLastErrorString != nullptr)
        {
            _getLastErrorString(message, maxSize);
        }
        else if(message != nullptr && maxSize > 0)
        {
            message[0] = '\0';
        }
    }

    hipdnn_data_sdk::utilities::Version version() override
    {
        return _version;
    }

    const char* versionString() override
    {
        return _versionString != nullptr ? _versionString() : "";
    }

    hipdnnStatus_t backendCreateAndDeserializeGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedGraph,
                                                       size_t graphByteSize) override
    {
        return _backendCreateAndDeserializeGraphExt != nullptr
                   ? _backendCreateAndDeserializeGraphExt(
                         descriptor, serializedGraph, graphByteSize)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetSerializedBinaryGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                      size_t requestedByteSize,
                                                      size_t* graphByteSize,
                                                      uint8_t* serializedGraph) override
    {
        return _backendGetSerializedBinaryGraphExt != nullptr
                   ? _backendGetSerializedBinaryGraphExt(
                         descriptor, requestedByteSize, graphByteSize, serializedGraph)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetSerializedJsonGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                    size_t requestedByteSize,
                                                    size_t* graphByteSize,
                                                    char* serializedJsonGraph) override
    {
        return _backendGetSerializedJsonGraphExt != nullptr
                   ? _backendGetSerializedJsonGraphExt(
                         descriptor, requestedByteSize, graphByteSize, serializedJsonGraph)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendCreateAndDeserializeJsonGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                           const char* jsonGraph,
                                                           size_t jsonByteSize) override
    {
        return _backendCreateAndDeserializeJsonGraphExt != nullptr
                   ? _backendCreateAndDeserializeJsonGraphExt(descriptor, jsonGraph, jsonByteSize)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetSerializedExecutionPlanExt(hipdnnBackendDescriptor_t descriptor,
                                                        size_t requestedByteSize,
                                                        size_t* planByteSize,
                                                        uint8_t* serializedPlan) override
    {
        return _backendGetSerializedExecutionPlanExt != nullptr
                   ? _backendGetSerializedExecutionPlanExt(
                         descriptor, requestedByteSize, planByteSize, serializedPlan)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t
        backendCreateAndDeserializeExecutionPlanExt(hipdnnHandle_t handle,
                                                    hipdnnBackendDescriptor_t* descriptor,
                                                    const uint8_t* serializedPlan,
                                                    size_t planByteSize) override
    {
        return _backendCreateAndDeserializeExecutionPlanExt != nullptr
                   ? _backendCreateAndDeserializeExecutionPlanExt(
                         handle, descriptor, serializedPlan, planByteSize)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t
        backendGetSerializedBinaryGraphAndPlanExt(hipdnnBackendDescriptor_t graphDescriptor,
                                                  hipdnnBackendDescriptor_t executionPlanDescriptor,
                                                  size_t requestedByteSize,
                                                  size_t* blobByteSize,
                                                  uint8_t* serializedBlob) override
    {
        return _backendGetSerializedBinaryGraphAndPlanExt != nullptr
                   ? _backendGetSerializedBinaryGraphAndPlanExt(graphDescriptor,
                                                                executionPlanDescriptor,
                                                                requestedByteSize,
                                                                blobByteSize,
                                                                serializedBlob)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetSerializedBinaryContentsExt(const uint8_t* serializedBlob,
                                                         size_t blobByteSize,
                                                         int* contentFlags) override
    {
        return _backendGetSerializedBinaryContentsExt != nullptr
                   ? _backendGetSerializedBinaryContentsExt(
                         serializedBlob, blobByteSize, contentFlags)
                   : missingSymbolStatus();
    }

    void loggingCallbackExt(hipdnnSeverity_t severity, const char* msg) override
    {
        if(_loggingCallbackExt != nullptr)
        {
            _loggingCallbackExt(severity, msg);
        }
    }

    hipdnnStatus_t setEnginePluginPathsExt(size_t numPaths,
                                           const char* const* pluginPaths,
                                           hipdnnPluginLoadingMode_ext_t mode) override
    {
        return _setEnginePluginPathsExt != nullptr
                   ? _setEnginePluginPathsExt(numPaths, pluginPaths, mode)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t setHeuristicPluginPathsExt(size_t numPaths,
                                              const char* const* pluginPaths,
                                              hipdnnPluginLoadingMode_ext_t mode) override
    {
        return _setHeuristicPluginPathsExt != nullptr
                   ? _setHeuristicPluginPathsExt(numPaths, pluginPaths, mode)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t getLoadedEnginePluginPathsExt(hipdnnHandle_t handle,
                                                 size_t* numPluginPaths,
                                                 char** pluginPaths,
                                                 size_t* maxStringLen) override
    {
        return _getLoadedEnginePluginPathsExt != nullptr
                   ? _getLoadedEnginePluginPathsExt(
                         handle, numPluginPaths, pluginPaths, maxStringLen)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t getEngineNameByIdExt(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        char* engineName,
                                        size_t* engineNameLen) override
    {
        return _getEngineNameByIdExt != nullptr
                   ? _getEngineNameByIdExt(handle, engineId, engineName, engineNameLen)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t getHeuristicPolicyCount(hipdnnHandle_t handle, size_t* numPolicies) override
    {
        return _getHeuristicPolicyCount != nullptr ? _getHeuristicPolicyCount(handle, numPolicies)
                                                   : missingSymbolStatus();
    }

    hipdnnStatus_t getHeuristicPolicyInfo(hipdnnHandle_t handle,
                                          size_t policyIndex,
                                          int64_t* policyId,
                                          char* policyName,
                                          size_t* policyNameLen,
                                          char* pluginName,
                                          size_t* pluginNameLen,
                                          char* pluginVersion,
                                          size_t* pluginVersionLen,
                                          char* apiVersion,
                                          size_t* apiVersionLen) override
    {
        return _getHeuristicPolicyInfo != nullptr ? _getHeuristicPolicyInfo(handle,
                                                                            policyIndex,
                                                                            policyId,
                                                                            policyName,
                                                                            policyNameLen,
                                                                            pluginName,
                                                                            pluginNameLen,
                                                                            pluginVersion,
                                                                            pluginVersionLen,
                                                                            apiVersion,
                                                                            apiVersionLen)
                                                  : missingSymbolStatus();
    }

    hipdnnStatus_t setUserLogCallbackExt(hipdnnUserLogCallback_t callback,
                                         hipdnnSeverity_t minLevel,
                                         hipdnnLogCallbackMode_t mode,
                                         hipdnnUserLogCallbackHandle_t userHandle) override
    {
        return _setUserLogCallbackExt != nullptr
                   ? _setUserLogCallbackExt(callback, minLevel, mode, userHandle)
                   : missingSymbolStatus();
    }

    hipdnnStatus_t backendSetGlobalLogLevelExt(hipdnnSeverity_t level) override
    {
        return _backendSetGlobalLogLevelExt != nullptr ? _backendSetGlobalLogLevelExt(level)
                                                       : missingSymbolStatus();
    }

    hipdnnStatus_t backendGetGlobalLogLevelExt(hipdnnSeverity_t* level) override
    {
        return _backendGetGlobalLogLevelExt != nullptr ? _backendGetGlobalLogLevelExt(level)
                                                       : missingSymbolStatus();
    }

    hipdnnStatus_t writeEngineRankingResultsExt(hipdnnHandle_t handle,
                                                hipdnnBackendDescriptor_t graphDescriptor,
                                                const int64_t* engineIdsInRankOrder,
                                                size_t engineIdCount,
                                                hipdnnAutotuneCacheWriteOutcome_ext_t* outcome
                                                = nullptr) override
    {
        return _writeEngineRankingResultsExt != nullptr
                   ? _writeEngineRankingResultsExt(
                         handle, graphDescriptor, engineIdsInRankOrder, engineIdCount, outcome)
                   : missingSymbolStatus();
    }

private:
    static constexpr hipdnnStatus_t missingSymbolStatus()
    {
        return HIPDNN_STATUS_VERSION_MISMATCH;
    }

    template <typename Fn>
    Fn resolve(const char* name)
    {
        return reinterpret_cast<Fn>(_symbolResolver(name));
    }

    void resolveAllSymbols()
    {
        _create = resolve<decltype(&hipdnnCreate)>("hipdnnCreate");
        _destroy = resolve<decltype(&hipdnnDestroy)>("hipdnnDestroy");
        _setStream = resolve<decltype(&hipdnnSetStream)>("hipdnnSetStream");
        _getStream = resolve<decltype(&hipdnnGetStream)>("hipdnnGetStream");
        _backendCreateDescriptor
            = resolve<decltype(&hipdnnBackendCreateDescriptor)>("hipdnnBackendCreateDescriptor");
        _backendDestroyDescriptor
            = resolve<decltype(&hipdnnBackendDestroyDescriptor)>("hipdnnBackendDestroyDescriptor");
        _backendExecute = resolve<decltype(&hipdnnBackendExecute)>("hipdnnBackendExecute");
        _backendFinalize = resolve<decltype(&hipdnnBackendFinalize)>("hipdnnBackendFinalize");
        _backendGetAttribute
            = resolve<decltype(&hipdnnBackendGetAttribute)>("hipdnnBackendGetAttribute");
        _backendSetAttribute
            = resolve<decltype(&hipdnnBackendSetAttribute)>("hipdnnBackendSetAttribute");
        _getErrorString = resolve<decltype(&hipdnnGetErrorString)>("hipdnnGetErrorString");
        _getLastErrorString
            = resolve<decltype(&hipdnnGetLastErrorString)>("hipdnnGetLastErrorString");
        _versionString = resolve<decltype(&hipdnnVersionString_ext)>("hipdnnVersionString_ext");
        _backendCreateAndDeserializeGraphExt
            = resolve<decltype(&hipdnnBackendCreateAndDeserializeGraph_ext)>(
                "hipdnnBackendCreateAndDeserializeGraph_ext");
        _backendGetSerializedBinaryGraphExt
            = resolve<decltype(&hipdnnBackendGetSerializedBinaryGraph_ext)>(
                "hipdnnBackendGetSerializedBinaryGraph_ext");
        _backendGetSerializedJsonGraphExt
            = resolve<decltype(&hipdnnBackendGetSerializedJsonGraph_ext)>(
                "hipdnnBackendGetSerializedJsonGraph_ext");
        _backendCreateAndDeserializeJsonGraphExt
            = resolve<decltype(&hipdnnBackendCreateAndDeserializeJsonGraph_ext)>(
                "hipdnnBackendCreateAndDeserializeJsonGraph_ext");
        _backendGetSerializedExecutionPlanExt
            = resolve<decltype(&hipdnnBackendGetSerializedExecutionPlan_ext)>(
                "hipdnnBackendGetSerializedExecutionPlan_ext");
        _backendCreateAndDeserializeExecutionPlanExt
            = resolve<decltype(&hipdnnBackendCreateAndDeserializeExecutionPlan_ext)>(
                "hipdnnBackendCreateAndDeserializeExecutionPlan_ext");
        _backendGetSerializedBinaryGraphAndPlanExt
            = resolve<decltype(&hipdnnBackendGetSerializedBinaryGraphAndPlan_ext)>(
                "hipdnnBackendGetSerializedBinaryGraphAndPlan_ext");
        _backendGetSerializedBinaryContentsExt
            = resolve<decltype(&hipdnnBackendGetSerializedBinaryContents_ext)>(
                "hipdnnBackendGetSerializedBinaryContents_ext");
        _loggingCallbackExt
            = resolve<decltype(&hipdnnLoggingCallback_ext)>("hipdnnLoggingCallback_ext");
        _setEnginePluginPathsExt
            = resolve<decltype(&hipdnnSetEnginePluginPaths_ext)>("hipdnnSetEnginePluginPaths_ext");
        _setHeuristicPluginPathsExt = resolve<decltype(&hipdnnSetHeuristicPluginPaths_ext)>(
            "hipdnnSetHeuristicPluginPaths_ext");
        _getLoadedEnginePluginPathsExt = resolve<decltype(&hipdnnGetLoadedEnginePluginPaths_ext)>(
            "hipdnnGetLoadedEnginePluginPaths_ext");
        _getEngineNameByIdExt
            = resolve<decltype(&hipdnnGetEngineNameById_ext)>("hipdnnGetEngineNameById_ext");
        _getHeuristicPolicyCount = resolve<decltype(&hipdnnGetHeuristicPolicyCount_ext)>(
            "hipdnnGetHeuristicPolicyCount_ext");
        _getHeuristicPolicyInfo = resolve<decltype(&hipdnnGetHeuristicPolicyInfo_ext)>(
            "hipdnnGetHeuristicPolicyInfo_ext");
        _setUserLogCallbackExt
            = resolve<decltype(&hipdnnSetUserLogCallback_ext)>("hipdnnSetUserLogCallback_ext");
        _backendSetGlobalLogLevelExt = resolve<decltype(&hipdnnBackendSetGlobalLogLevel_ext)>(
            "hipdnnBackendSetGlobalLogLevel_ext");
        _backendGetGlobalLogLevelExt = resolve<decltype(&hipdnnBackendGetGlobalLogLevel_ext)>(
            "hipdnnBackendGetGlobalLogLevel_ext");
        _writeEngineRankingResultsExt
            = resolve<decltype(&hipdnnBackendWriteEngineRankingResults_ext)>(
                "hipdnnBackendWriteEngineRankingResults_ext");
    }

    hipdnn_data_sdk::utilities::Version _version;
    SymbolResolver _symbolResolver;

    decltype(&hipdnnCreate) _create = nullptr;
    decltype(&hipdnnDestroy) _destroy = nullptr;
    decltype(&hipdnnSetStream) _setStream = nullptr;
    decltype(&hipdnnGetStream) _getStream = nullptr;
    decltype(&hipdnnBackendCreateDescriptor) _backendCreateDescriptor = nullptr;
    decltype(&hipdnnBackendDestroyDescriptor) _backendDestroyDescriptor = nullptr;
    decltype(&hipdnnBackendExecute) _backendExecute = nullptr;
    decltype(&hipdnnBackendFinalize) _backendFinalize = nullptr;
    decltype(&hipdnnBackendGetAttribute) _backendGetAttribute = nullptr;
    decltype(&hipdnnBackendSetAttribute) _backendSetAttribute = nullptr;
    decltype(&hipdnnGetErrorString) _getErrorString = nullptr;
    decltype(&hipdnnGetLastErrorString) _getLastErrorString = nullptr;
    decltype(&hipdnnVersionString_ext) _versionString = nullptr;
    decltype(&hipdnnBackendCreateAndDeserializeGraph_ext) _backendCreateAndDeserializeGraphExt
        = nullptr;
    decltype(&hipdnnBackendGetSerializedBinaryGraph_ext) _backendGetSerializedBinaryGraphExt
        = nullptr;
    decltype(&hipdnnBackendGetSerializedJsonGraph_ext) _backendGetSerializedJsonGraphExt = nullptr;
    decltype(&hipdnnBackendCreateAndDeserializeJsonGraph_ext)
        _backendCreateAndDeserializeJsonGraphExt
        = nullptr;
    decltype(&hipdnnBackendGetSerializedExecutionPlan_ext) _backendGetSerializedExecutionPlanExt
        = nullptr;
    decltype(&hipdnnBackendCreateAndDeserializeExecutionPlan_ext)
        _backendCreateAndDeserializeExecutionPlanExt
        = nullptr;
    decltype(&hipdnnBackendGetSerializedBinaryGraphAndPlan_ext)
        _backendGetSerializedBinaryGraphAndPlanExt
        = nullptr;
    decltype(&hipdnnBackendGetSerializedBinaryContents_ext) _backendGetSerializedBinaryContentsExt
        = nullptr;
    decltype(&hipdnnLoggingCallback_ext) _loggingCallbackExt = nullptr;
    decltype(&hipdnnSetEnginePluginPaths_ext) _setEnginePluginPathsExt = nullptr;
    decltype(&hipdnnSetHeuristicPluginPaths_ext) _setHeuristicPluginPathsExt = nullptr;
    decltype(&hipdnnGetLoadedEnginePluginPaths_ext) _getLoadedEnginePluginPathsExt = nullptr;
    decltype(&hipdnnGetEngineNameById_ext) _getEngineNameByIdExt = nullptr;
    decltype(&hipdnnGetHeuristicPolicyCount_ext) _getHeuristicPolicyCount = nullptr;
    decltype(&hipdnnGetHeuristicPolicyInfo_ext) _getHeuristicPolicyInfo = nullptr;
    decltype(&hipdnnSetUserLogCallback_ext) _setUserLogCallbackExt = nullptr;
    decltype(&hipdnnBackendSetGlobalLogLevel_ext) _backendSetGlobalLogLevelExt = nullptr;
    decltype(&hipdnnBackendGetGlobalLogLevel_ext) _backendGetGlobalLogLevelExt = nullptr;
    decltype(&hipdnnBackendWriteEngineRankingResults_ext) _writeEngineRankingResultsExt = nullptr;
};

} // namespace hipdnn_frontend::detail

#endif // not HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

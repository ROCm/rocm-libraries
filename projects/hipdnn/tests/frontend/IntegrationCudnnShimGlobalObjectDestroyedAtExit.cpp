// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle from the cuDNN-compatibility shim's create_cudnn_handle(), owned by a
// namespace-scope object and destroyed during process exit. See
// GlobalObjectDestroyedAtExitFixture.hpp for how the child process is checked.
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in tests/frontend/CMakeLists.txt.

#include <atomic>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

#include "GlobalObjectDestroyedAtExitFixture.hpp"

namespace
{

using hipdnn_frontend::detail::IHipdnnBackend;

// True once DestroyObservingBackend::destroy() runs while IHipdnnBackend::isShutdown() is true:
// cudnnDestroy's own exit guard let the call reach the backend instead of short-circuiting.
// Namespace scope, so it is still readable from a destructor running during exit.
std::atomic<bool> gDestroyCalledAfterShutdown{false};

// Forwards every call to the real backend unchanged, except that destroy() first records
// whether it was reached while isShutdown() is true. Installed (via IHipdnnBackend::setInstance)
// only after the real backend already exists, so hipdnnBackend() returns this wrapper for the
// rest of the test: cudnnDestroy's own guard, when present, returns before ever calling
// hipdnnBackend()->destroy(), so this wrapper's destroy() is not reached; removing or defeating
// that guard routes the real call through here instead, which DestroyGuardProbe below observes.
class DestroyObservingBackend : public IHipdnnBackend
{
public:
    explicit DestroyObservingBackend(std::shared_ptr<IHipdnnBackend> real)
        : _real(std::move(real))
    {
    }

    hipdnnStatus_t create(hipdnnHandle_t* handle) override
    {
        return _real->create(handle);
    }

    hipdnnStatus_t destroy(hipdnnHandle_t handle) override
    {
        if(IHipdnnBackend::isShutdown())
        {
            gDestroyCalledAfterShutdown.store(true, std::memory_order_release);
        }
        return _real->destroy(handle);
    }

    hipdnnStatus_t setStream(hipdnnHandle_t handle, hipStream_t streamId) override
    {
        return _real->setStream(handle, streamId);
    }

    hipdnnStatus_t getStream(hipdnnHandle_t handle, hipStream_t* streamId) override
    {
        return _real->getStream(handle, streamId);
    }

    hipdnnStatus_t backendCreateDescriptor(hipdnnBackendDescriptorType_t descriptorType,
                                           hipdnnBackendDescriptor_t* descriptor) override
    {
        return _real->backendCreateDescriptor(descriptorType, descriptor);
    }

    hipdnnStatus_t backendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor) override
    {
        return _real->backendDestroyDescriptor(descriptor);
    }

    hipdnnStatus_t backendExecute(hipdnnHandle_t handle,
                                  hipdnnBackendDescriptor_t executionPlan,
                                  hipdnnBackendDescriptor_t variantPack) override
    {
        return _real->backendExecute(handle, executionPlan, variantPack);
    }

    hipdnnStatus_t backendFinalize(hipdnnBackendDescriptor_t descriptor) override
    {
        return _real->backendFinalize(descriptor);
    }

    hipdnnStatus_t backendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) override
    {
        return _real->backendGetAttribute(descriptor,
                                          attributeName,
                                          attributeType,
                                          requestedElementCount,
                                          elementCount,
                                          arrayOfElements);
    }

    hipdnnStatus_t backendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements) override
    {
        return _real->backendSetAttribute(
            descriptor, attributeName, attributeType, elementCount, arrayOfElements);
    }

    const char* getErrorString(hipdnnStatus_t status) override
    {
        return _real->getErrorString(status);
    }

    void getLastErrorString(char* message, size_t maxSize) override
    {
        _real->getLastErrorString(message, maxSize);
    }

    hipdnn_data_sdk::utilities::Version version() override
    {
        return _real->version();
    }

    const char* versionString() override
    {
        return _real->versionString();
    }

    hipdnnStatus_t backendCreateAndDeserializeGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedGraph,
                                                       size_t graphByteSize) override
    {
        return _real->backendCreateAndDeserializeGraphExt(
            descriptor, serializedGraph, graphByteSize);
    }

    hipdnnStatus_t backendGetSerializedBinaryGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                      size_t requestedByteSize,
                                                      size_t* graphByteSize,
                                                      uint8_t* serializedGraph) override
    {
        return _real->backendGetSerializedBinaryGraphExt(
            descriptor, requestedByteSize, graphByteSize, serializedGraph);
    }

    hipdnnStatus_t backendGetSerializedJsonGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                    size_t requestedByteSize,
                                                    size_t* graphByteSize,
                                                    char* serializedJsonGraph) override
    {
        return _real->backendGetSerializedJsonGraphExt(
            descriptor, requestedByteSize, graphByteSize, serializedJsonGraph);
    }

    hipdnnStatus_t backendCreateAndDeserializeJsonGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                           const char* jsonGraph,
                                                           size_t jsonByteSize) override
    {
        return _real->backendCreateAndDeserializeJsonGraphExt(descriptor, jsonGraph, jsonByteSize);
    }

    hipdnnStatus_t backendGetSerializedExecutionPlanExt(hipdnnBackendDescriptor_t descriptor,
                                                        size_t requestedByteSize,
                                                        size_t* planByteSize,
                                                        uint8_t* serializedPlan) override
    {
        return _real->backendGetSerializedExecutionPlanExt(
            descriptor, requestedByteSize, planByteSize, serializedPlan);
    }

    hipdnnStatus_t
        backendCreateAndDeserializeExecutionPlanExt(hipdnnHandle_t handle,
                                                    hipdnnBackendDescriptor_t* descriptor,
                                                    const uint8_t* serializedPlan,
                                                    size_t planByteSize) override
    {
        return _real->backendCreateAndDeserializeExecutionPlanExt(
            handle, descriptor, serializedPlan, planByteSize);
    }

    hipdnnStatus_t
        backendGetSerializedBinaryGraphAndPlanExt(hipdnnBackendDescriptor_t graphDescriptor,
                                                  hipdnnBackendDescriptor_t executionPlanDescriptor,
                                                  size_t requestedByteSize,
                                                  size_t* blobByteSize,
                                                  uint8_t* serializedBlob) override
    {
        return _real->backendGetSerializedBinaryGraphAndPlanExt(graphDescriptor,
                                                                executionPlanDescriptor,
                                                                requestedByteSize,
                                                                blobByteSize,
                                                                serializedBlob);
    }

    hipdnnStatus_t backendGetSerializedBinaryContentsExt(const uint8_t* serializedBlob,
                                                         size_t blobByteSize,
                                                         int* contentFlags) override
    {
        return _real->backendGetSerializedBinaryContentsExt(
            serializedBlob, blobByteSize, contentFlags);
    }

    void loggingCallbackExt(hipdnnSeverity_t severity, const char* msg) override
    {
        _real->loggingCallbackExt(severity, msg);
    }

    hipdnnStatus_t setEnginePluginPathsExt(size_t numPaths,
                                           const char* const* pluginPaths,
                                           hipdnnPluginLoadingMode_ext_t mode) override
    {
        return _real->setEnginePluginPathsExt(numPaths, pluginPaths, mode);
    }

    hipdnnStatus_t setHeuristicPluginPathsExt(size_t numPaths,
                                              const char* const* pluginPaths,
                                              hipdnnPluginLoadingMode_ext_t mode) override
    {
        return _real->setHeuristicPluginPathsExt(numPaths, pluginPaths, mode);
    }

    hipdnnStatus_t getLoadedEnginePluginPathsExt(hipdnnHandle_t handle,
                                                 size_t* numPluginPaths,
                                                 char** pluginPaths,
                                                 size_t* maxStringLen) override
    {
        return _real->getLoadedEnginePluginPathsExt(
            handle, numPluginPaths, pluginPaths, maxStringLen);
    }

    hipdnnStatus_t getEngineNameByIdExt(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        char* engineName,
                                        size_t* engineNameLen) override
    {
        return _real->getEngineNameByIdExt(handle, engineId, engineName, engineNameLen);
    }

    hipdnnStatus_t getHeuristicPolicyCount(hipdnnHandle_t handle, size_t* numPolicies) override
    {
        return _real->getHeuristicPolicyCount(handle, numPolicies);
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
        return _real->getHeuristicPolicyInfo(handle,
                                             policyIndex,
                                             policyId,
                                             policyName,
                                             policyNameLen,
                                             pluginName,
                                             pluginNameLen,
                                             pluginVersion,
                                             pluginVersionLen,
                                             apiVersion,
                                             apiVersionLen);
    }

    hipdnnStatus_t setUserLogCallbackExt(hipdnnUserLogCallback_t callback,
                                         hipdnnSeverity_t minLevel,
                                         hipdnnLogCallbackMode_t mode,
                                         hipdnnUserLogCallbackHandle_t userHandle) override
    {
        return _real->setUserLogCallbackExt(callback, minLevel, mode, userHandle);
    }

    hipdnnStatus_t backendSetGlobalLogLevelExt(hipdnnSeverity_t level) override
    {
        return _real->backendSetGlobalLogLevelExt(level);
    }

    hipdnnStatus_t backendGetGlobalLogLevelExt(hipdnnSeverity_t* level) override
    {
        return _real->backendGetGlobalLogLevelExt(level);
    }

    hipdnnStatus_t
        writeEngineRankingResultsExt(hipdnnHandle_t handle,
                                     hipdnnBackendDescriptor_t graphDescriptor,
                                     const int64_t* engineIdsInRankOrder,
                                     size_t engineIdCount,
                                     hipdnnAutotuneCacheWriteOutcome_ext_t* outcome) override
    {
        return _real->writeEngineRankingResultsExt(
            handle, graphDescriptor, engineIdsInRankOrder, engineIdCount, outcome);
    }

private:
    std::shared_ptr<IHipdnnBackend> _real;
};

// Destroyed after gCudnnHandle (declared below it): fails the child if cudnnDestroy's exit
// guard let the call reach DestroyObservingBackend::destroy() while isShutdown() was true.
class DestroyGuardProbe
{
public:
    ~DestroyGuardProbe()
    {
        if(gDestroyCalledAfterShutdown.load(std::memory_order_acquire))
        {
            std::_Exit(hipdnn_tests::DESTROY_FAILED_AT_EXIT);
        }
    }
};

DestroyGuardProbe gDestroyGuardProbe;
hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;
std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter> gCudnnHandle;

class IntegrationCudnnShimGlobalObjectDestroyedAtExit
    : public hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit
{
};

TEST_F(IntegrationCudnnShimGlobalObjectDestroyedAtExit, HandleExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    gCudnnHandle = create_cudnn_handle();
    ASSERT_NE(gCudnnHandle, nullptr);
    ASSERT_TRUE(isTestGoodPluginLoaded());

    // Install the observer only after the real backend exists: hipdnnBackend() keeps returning
    // the existing instance (now this wrapper) for the rest of the process, including at exit.
    auto real = IHipdnnBackend::getInstance();
    ASSERT_NE(real, nullptr);
    IHipdnnBackend::setInstance(std::make_shared<DestroyObservingBackend>(std::move(real)));
}

} // namespace

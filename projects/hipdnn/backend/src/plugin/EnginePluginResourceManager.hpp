// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

#include "hipdnn_backend.h"

namespace hipdnn_sdk
{
namespace data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
struct EngineDetails;
}
}

namespace hipdnn_backend
{

class GraphDescriptor;

namespace plugin
{

class EngineDetailsWrapper;
class EngineExecutionContextWrapper;
class EnginePlugin;
class EnginePluginManager;

class EnginePluginResourceManager
{
protected:
    // Protected constructor for mock testing
    EnginePluginResourceManager();

public:
    // MT-safe static functions
    // Load plugins from a specific path, for testing purposes
    static void setPluginPaths(const std::vector<std::filesystem::path>& pluginPaths,
                               hipdnnPluginLoadingMode_ext_t loadingMode);
    static std::set<std::filesystem::path> getPluginPaths();

    static std::shared_ptr<EnginePluginResourceManager> create();

    EnginePluginResourceManager(std::shared_ptr<EnginePluginManager> pm);
    virtual ~EnginePluginResourceManager();

    // Prevent copying
    EnginePluginResourceManager(const EnginePluginResourceManager&) = delete;
    EnginePluginResourceManager& operator=(const EnginePluginResourceManager&) = delete;

    // Allow moving
    EnginePluginResourceManager(EnginePluginResourceManager&& other) noexcept;
    EnginePluginResourceManager& operator=(EnginePluginResourceManager&& other) noexcept;

    // MT-unsafe instance methods
    // virtual for gMock testing
    virtual void setStream(hipStream_t stream) const;
    virtual std::vector<int64_t> getApplicableEngineIds(const GraphDescriptor* graphDesc) const;
    virtual size_t getWorkspaceSize(int64_t engineId,
                                    const hipdnnPluginConstData_t* engineConfig,
                                    const GraphDescriptor* graphDesc) const;
    virtual size_t getWorkspaceSize(int64_t engineId,
                                    hipdnnEnginePluginExecutionContext_t executionContext) const;

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

    virtual void
        getLoadedPluginFiles(size_t* numPlugins, char** pluginPaths, size_t* maxStringLen) const;

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
    virtual void
        destroyExecutionContext(int64_t engineId,
                                hipdnnEnginePluginExecutionContext_t executionContext) const;

    void executeOpGraph(int64_t engineId,
                        hipdnnEnginePluginExecutionContext_t executionContext,
                        void* workspace,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    std::shared_ptr<EnginePluginManager> _pm;
    std::unordered_map<hipdnnEnginePluginHandle_t, const EnginePlugin*> _handleToPlugin;
    std::unordered_map<int64_t, hipdnnEnginePluginHandle_t> _engineIdToHandle;

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

    const hipdnn_sdk::data_objects::EngineDetails* get() const;

private:
    std::shared_ptr<EnginePluginResourceManager> _rm;
    hipdnnPluginConstData_t _engineDetailsData;
};

// A class to manage engine execution context lifecycle
class EngineExecutionContextWrapper
{
public:
    EngineExecutionContextWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                  int64_t engineId,
                                  const hipdnnPluginConstData_t* engineConfig,
                                  const GraphDescriptor* graphDesc);
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

} // namespace plugin
} // hipdnn_backend

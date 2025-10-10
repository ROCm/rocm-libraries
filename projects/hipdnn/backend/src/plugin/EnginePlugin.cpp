// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <limits>

#include "EnginePlugin.hpp"

namespace hipdnn_backend
{
namespace plugin
{

EnginePlugin::EnginePlugin(SharedLibrary&& lib)
    : PluginBase(std::move(lib))
{
    resolveSymbols();
}

EnginePlugin::EnginePlugin()
{
    // This constructor is used for mocking purposes in tests.
#ifndef NDEBUG
    _initialized = true;
#endif
}

void EnginePlugin::resolveSymbols()
{
    if(type() != HIPDNN_PLUGIN_TYPE_ENGINE)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Wrong plugin type");
    }

    const auto funcNameGetAllEngineIds = "hipdnnEnginePluginGetAllEngineIds";
    _funcGetAllEngineIds = _lib.getSymbol<decltype(_funcGetAllEngineIds)>(funcNameGetAllEngineIds);

    const auto funcNameCreateHandle = "hipdnnEnginePluginCreate";
    _funcCreateHandle = _lib.getSymbol<decltype(_funcCreateHandle)>(funcNameCreateHandle);

    const auto funcNameDestroyHandle = "hipdnnEnginePluginDestroy";
    _funcDestroyHandle = _lib.getSymbol<decltype(_funcDestroyHandle)>(funcNameDestroyHandle);

    const auto funcNameSetStream = "hipdnnEnginePluginSetStream";
    _funcSetStream = _lib.getSymbol<decltype(_funcSetStream)>(funcNameSetStream);

    const auto funcNameGetApplicableEngineIds = "hipdnnEnginePluginGetApplicableEngineIds";
    _funcGetApplicableEngineIds
        = _lib.getSymbol<decltype(_funcGetApplicableEngineIds)>(funcNameGetApplicableEngineIds);

    const auto funcNameGetEngineDetails = "hipdnnEnginePluginGetEngineDetails";
    _funcGetEngineDetails
        = _lib.getSymbol<decltype(_funcGetEngineDetails)>(funcNameGetEngineDetails);

    const auto funcNameDestroyEngineDetails = "hipdnnEnginePluginDestroyEngineDetails";
    _funcDestroyEngineDetails
        = _lib.getSymbol<decltype(_funcDestroyEngineDetails)>(funcNameDestroyEngineDetails);

    const auto funcNameGetWorkspaceSize = "hipdnnEnginePluginGetWorkspaceSize";
    _funcGetWorkspaceSize
        = _lib.getSymbol<decltype(_funcGetWorkspaceSize)>(funcNameGetWorkspaceSize);

    const auto funcNameCreateExecutionContext = "hipdnnEnginePluginCreateExecutionContext";
    _funcCreateExecutionContext
        = _lib.getSymbol<decltype(_funcCreateExecutionContext)>(funcNameCreateExecutionContext);

    const auto funcNameDestroyExecutionContext = "hipdnnEnginePluginDestroyExecutionContext";
    _funcDestroyExecutionContext
        = _lib.getSymbol<decltype(_funcDestroyExecutionContext)>(funcNameDestroyExecutionContext);

    const auto funcNameGetWorkspaceSizeFromExecutionContext
        = "hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext";
    _funcGetWorkspaceSizeFromExecutionContext
        = _lib.getSymbol<decltype(_funcGetWorkspaceSizeFromExecutionContext)>(
            funcNameGetWorkspaceSizeFromExecutionContext);

    const auto funcNameExecuteOpGraph = "hipdnnEnginePluginExecuteOpGraph";
    _funcExecuteOpGraph = _lib.getSymbol<decltype(_funcExecuteOpGraph)>(funcNameExecuteOpGraph);

#ifndef NDEBUG
    _initialized = true;
#endif
}

std::vector<int64_t> EnginePlugin::getAllEngineIds() const
{
    assert(_initialized);

    if(!_allEngineIds.empty())
    {
        return _allEngineIds;
    }

    uint32_t numEngines = 0;
    invokePluginFunction("get number of engines", _funcGetAllEngineIds, nullptr, 0u, &numEngines);

    THROW_IF_EQ(numEngines, 0, HIPDNN_STATUS_PLUGIN_ERROR, "No engines found in the plugin");

    const uint32_t maxEngines = numEngines;
    std::vector<int64_t> engineIds(maxEngines);

    invokePluginFunction(
        "get all engine IDs", _funcGetAllEngineIds, engineIds.data(), maxEngines, &numEngines);

    THROW_IF_NE(numEngines,
                maxEngines,
                HIPDNN_STATUS_PLUGIN_ERROR,
                "Number of engines returned does not match expected count");

    std::sort(engineIds.begin(), engineIds.end());
    if(std::adjacent_find(engineIds.begin(), engineIds.end()) != engineIds.end())
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Duplicate engine IDs found");
    }
    _allEngineIds = engineIds;

    return engineIds;
}

hipdnnEnginePluginHandle_t EnginePlugin::createHandle() const
{
    assert(_initialized);
    hipdnnEnginePluginHandle_t handle;
    invokePluginFunction("create engine plugin handle", _funcCreateHandle, &handle);
    return handle;
}

void EnginePlugin::destroyHandle(hipdnnEnginePluginHandle_t handle) const
{
    assert(_initialized);
    invokePluginFunction("destroy engine plugin handle", _funcDestroyHandle, handle);
}

void EnginePlugin::setStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream) const
{
    assert(_initialized);
    invokePluginFunction("set stream for engine plugin handle", _funcSetStream, handle, stream);
}

std::vector<int64_t>
    EnginePlugin::getApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                         const hipdnnPluginConstData_t* opGraph) const
{
    assert(_initialized);

    if(_allEngineIds.empty())
    {
        getAllEngineIds();
    }

    const auto maxEngines = static_cast<uint32_t>(_allEngineIds.size());
    std::vector<int64_t> engineIds(maxEngines);
    uint32_t numEngines = 0;

    invokePluginFunction("get applicable engine IDs",
                         _funcGetApplicableEngineIds,
                         handle,
                         opGraph,
                         engineIds.data(),
                         maxEngines,
                         &numEngines);

    if(numEngines == 0)
    {
        return {}; // No applicable engines found
    }

    THROW_IF_LT(maxEngines,
                numEngines,
                HIPDNN_STATUS_PLUGIN_ERROR,
                "More applicable engines than expected");

    engineIds.resize(numEngines);

    std::sort(engineIds.begin(), engineIds.end());
    if(std::adjacent_find(engineIds.begin(), engineIds.end()) != engineIds.end())
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Duplicate engine IDs found");
    }

    for(const auto engineId : engineIds)
    {
        if(std::find(_allEngineIds.begin(), _allEngineIds.end(), engineId) == _allEngineIds.end())
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "Engine ID not found in the plugin's known IDs");
        }
    }

    return engineIds;
}

void EnginePlugin::getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                    int64_t engineId,
                                    const hipdnnPluginConstData_t* opGraph,
                                    hipdnnPluginConstData_t* engineDetails) const
{
    assert(_initialized);
    invokePluginFunction(
        "get engine details", _funcGetEngineDetails, handle, engineId, opGraph, engineDetails);
}

void EnginePlugin::destroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                        hipdnnPluginConstData_t* engineDetails) const

{
    assert(_initialized);
    invokePluginFunction(
        "destroy engine details", _funcDestroyEngineDetails, handle, engineDetails);
}

size_t EnginePlugin::getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                      const hipdnnPluginConstData_t* engineConfig,
                                      const hipdnnPluginConstData_t* opGraph) const
{
    assert(_initialized);
    size_t workspaceSize = 0;
    invokePluginFunction(
        "get workspace size", _funcGetWorkspaceSize, handle, engineConfig, opGraph, &workspaceSize);
    return workspaceSize;
}

size_t EnginePlugin::getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                      hipdnnEnginePluginExecutionContext_t executionContext) const
{
    assert(_initialized);
    size_t workspaceSize = 0;
    invokePluginFunction("get workspace size from execution context",
                         _funcGetWorkspaceSizeFromExecutionContext,
                         handle,
                         executionContext,
                         &workspaceSize);
    return workspaceSize;
}

hipdnnEnginePluginExecutionContext_t
    EnginePlugin::createExecutionContext(hipdnnEnginePluginHandle_t handle,
                                         const hipdnnPluginConstData_t* engineConfig,
                                         const hipdnnPluginConstData_t* opGraph) const
{
    assert(_initialized);
    hipdnnEnginePluginExecutionContext_t execContext;
    invokePluginFunction("create execution context",
                         _funcCreateExecutionContext,
                         handle,
                         engineConfig,
                         opGraph,
                         &execContext);
    return execContext;
}

void EnginePlugin::destroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext) const
{
    assert(_initialized);
    invokePluginFunction(
        "destroy execution context", _funcDestroyExecutionContext, handle, executionContext);
}

void EnginePlugin::executeOpGraph(hipdnnEnginePluginHandle_t handle,
                                  hipdnnEnginePluginExecutionContext_t executionContext,
                                  void* workspace,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers) const
{
    assert(_initialized);
    invokePluginFunction("execute op graph",
                         _funcExecuteOpGraph,
                         handle,
                         executionContext,
                         workspace,
                         deviceBuffers,
                         numDeviceBuffers);
}

} // namespace plugin
} // hipdnn_backend

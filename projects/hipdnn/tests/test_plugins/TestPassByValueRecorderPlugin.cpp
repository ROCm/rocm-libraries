// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin that reports the runtime pass-by-value minimum API version
// (K_PASS_BY_VALUE_MIN_API_VERSION, "1.2.0") and RECORDS the scalar value it
// resolves for every runtime pass-by-value tensor. Unlike TestPassByValuePlugin
// (which only clears the version floor so the version-filter tests can admit a
// runtime-pbv graph), this plugin exercises the full host-scalar delivery path:
//
//   - At createExecutionContext it walks the op-graph tensor map and, using the
//     shared plugin SDK helper makeScalarOperand(), builds a ScalarOperand for
//     every runtime pass-by-value tensor (classification deferred exactly as a
//     real provider does).
//   - At executeOpGraph it calls the shared helper resolveScalarOperand() with
//     the caller's device_buffers, i.e. the same code path MIOpen / hip-kernel
//     use, and records the resulting (uid, value) so a frontend test can assert
//     the EXACT host scalar reached the plugin boundary.
//
// Using the real SDK helpers here is deliberate: it verifies the helpers work
// inside an independent plugin, not just inside the in-tree providers.
//
// The recorded values are exposed via four exported C symbols the test resolves
// through the plugin's own dynamic-loader handle:
//   hipdnnTestPbvPluginGetReceivedCount() -> uint32_t
//   hipdnnTestPbvPluginGetReceivedUidAt(uint32_t) -> int64_t
//   hipdnnTestPbvPluginGetReceivedValueAt(uint32_t) -> double
//   hipdnnTestPbvPluginReset() -> void

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

#include <cstdint>
#include <utility>
#include <vector>

// Runtime pass-by-value scalar operands built at createExecutionContext from the
// op-graph tensor map, and the (uid, resolved-value) pairs recorded at execute.
// thread_local mirrors TestKnobsPlugin's recording storage. Tests reset before
// each execute so a single graph's deliveries are read back deterministically.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local std::vector<hipdnn_plugin_sdk::ScalarOperand> gScalarOperands;
static thread_local std::vector<std::pair<int64_t, double>> gRecordedScalars;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class PassByValueRecorderPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_PassByValueRecorderPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports the minimum plugin SDK version that advertises runtime pass-by-value support.
    const char* getPluginApiVersion() const override
    {
        return hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION.data();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<PassByValueRecorderPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    /// Builds a ScalarOperand (via the shared SDK helper) for every runtime
    /// pass-by-value tensor in the op graph, deferring the value read to execute.
    static hipdnnPluginStatus_t
        createExecutionContext(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* engineConfig,
                               const hipdnnPluginConstData_t* opGraph,
                               hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        return hipdnn_plugin_sdk::tryCatch([&]() {
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(opGraph->ptr);

            gScalarOperands.clear();

            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
                opGraph->ptr, opGraph->size);
            const auto& tensorMap = graphWrapper.getTensorMap();
            for(const auto& [uid, attr] : tensorMap)
            {
                if(attr != nullptr && attr->is_runtime_pass_by_value())
                {
                    gScalarOperands.push_back(
                        hipdnn_plugin_sdk::makeScalarOperand(tensorMap, uid, "pbv_scalar"));
                }
            }

            // Delegate context object creation to the base implementation.
            const hipdnnPluginStatus_t status = TestPluginBase::enginePluginCreateExecutionContext(
                handle, engineConfig, opGraph, executionContext);
            if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    status, "base createExecutionContext failed in recorder plugin");
            }
        });
    }

    /// Resolves every recorded ScalarOperand from the caller's device_buffers
    /// (the same SDK path a real provider uses) and records (uid, value).
    static hipdnnPluginStatus_t
        executeOpGraph(hipdnnEnginePluginHandle_t handle,
                       hipdnnEnginePluginExecutionContext_t executionContext,
                       void* workspace,
                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                       uint32_t numDeviceBuffers)
    {
        return hipdnn_plugin_sdk::tryCatch([&]() {
            hipdnn_plugin_sdk::throwIfNull(deviceBuffers);

            for(const auto& op : gScalarOperands)
            {
                const double value = hipdnn_plugin_sdk::toDouble(
                    hipdnn_plugin_sdk::resolveScalarOperand(op, deviceBuffers, numDeviceBuffers));
                gRecordedScalars.emplace_back(op.uid, value);
            }

            // Delegate the (no-op) execute bookkeeping to the base implementation.
            const hipdnnPluginStatus_t status = TestPluginBase::enginePluginExecuteOpGraph(
                handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
            if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    status, "base executeOpGraph failed in recorder plugin");
            }
        });
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<PassByValueRecorderPlugin>());
}

// Custom API registration: standard symbols delegate to the base, while
// createExecutionContext / executeOpGraph route through the recorder overrides,
// plus the four recorder accessor symbols.
extern "C" {
HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return TestPluginBase::pluginGetName(name);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return TestPluginBase::pluginGetVersion(version);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    return TestPluginBase::pluginGetApiVersion(version);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return TestPluginBase::pluginGetType(type);
}

HIPDNN_TEST_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    TestPluginBase::pluginGetLastErrorString(errorStr);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return TestPluginBase::pluginSetLoggingCallback(callback);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return TestPluginBase::enginePluginCreate(handle);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return TestPluginBase::enginePluginDestroy(handle);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream)
{
    return TestPluginBase::enginePluginSetStream(handle, stream);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    return TestPluginBase::enginePluginGetApplicableEngineIds(
        handle, opGraph, engineIds, maxEngines, numEngines);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engineId,
                                       const hipdnnPluginConstData_t* opGraph,
                                       hipdnnPluginConstData_t* engineDetails)
{
    return TestPluginBase::enginePluginGetEngineDetails(handle, engineId, opGraph, engineDetails);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(
    hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engineDetails)
{
    return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engineConfig,
                                       const hipdnnPluginConstData_t* opGraph,
                                       size_t* workspaceSize)
{
    return TestPluginBase::enginePluginGetWorkspaceSize(
        handle, engineConfig, opGraph, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t executionContext,
        size_t* workspaceSize)
{
    return TestPluginBase::enginePluginGetWorkspaceSize(handle, executionContext, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return PassByValueRecorderPlugin::createExecutionContext(
        handle, engineConfig, opGraph, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)
{
    return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    return PassByValueRecorderPlugin::executeOpGraph(
        handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}

// --- Recorder accessor symbols (resolved by the test via the plugin handle) ---

HIPDNN_PLUGIN_EXPORT uint32_t hipdnnTestPbvPluginGetReceivedCount()
{
    return static_cast<uint32_t>(gRecordedScalars.size());
}

HIPDNN_PLUGIN_EXPORT int64_t hipdnnTestPbvPluginGetReceivedUidAt(uint32_t index)
{
    if(index >= gRecordedScalars.size())
    {
        return 0;
    }
    return gRecordedScalars[index].first;
}

HIPDNN_PLUGIN_EXPORT double hipdnnTestPbvPluginGetReceivedValueAt(uint32_t index)
{
    if(index >= gRecordedScalars.size())
    {
        return 0.0;
    }
    return gRecordedScalars[index].second;
}

HIPDNN_PLUGIN_EXPORT void hipdnnTestPbvPluginReset()
{
    gScalarOperands.clear();
    gRecordedScalars.clear();
}
} // extern "C"

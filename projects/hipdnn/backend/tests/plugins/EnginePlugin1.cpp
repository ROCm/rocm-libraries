// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple engine plugin.
// It is used to test the plugin subsystem.

#include <tuple>

#include "hipdnn_sdk/plugin/PluginException.hpp"
#include <hip/hip_runtime.h>

#include "EnginePluginApiImpl.hpp"
#include "PluginApiImpl.hpp"

using namespace hipdnn_plugin;

const char* const PLUGIN_NAME = "EnginePlugin1";
const char* const PLUGIN_VERSION = "1.0";
const hipdnnPluginType_t PLUGIN_TYPE = HIPDNN_PLUGIN_TYPE_ENGINE;

struct HipdnnEnginePluginExecutionContext
{
    uint64_t dummy; // Placeholder
};

namespace
{

const uint32_t PLUGIN_NUM_ENGINES = 3;
const int64_t PLUGIN_FIRST_ENGINE_ID = 100;

// TODO Use op_graph instead of hardcoded value
const unsigned GPU_DATA_SIZE = 512;

// TODO Use HIP RTC to compile the kernel at runtime
__global__ void engineKernel(const uint32_t* input, uint32_t* output, uint32_t size)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size)
    {
        output[tid] = input[tid];
    }
}

// Run the kernel
void runEngine(const uint32_t* input, uint32_t* output, uint32_t size)
{
    const auto blockSize = 256U;
    const auto gridSize = (size + blockSize - 1) / blockSize;

    // Launch the kernel on the default stream.
    engineKernel<<<dim3(gridSize), dim3(blockSize), 0, hipStreamDefault>>>(input, output, size);

    // Check if the kernel launch was successful.
    hipError_t error = hipGetLastError();
    if(error != hipSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "kernel launch failed, error: "
                                        + std::string(hipGetErrorString(error)));
    }
}

} // namespace

void getAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    for(uint32_t i = 0; i < maxEngines && i < PLUGIN_NUM_ENGINES; ++i)
    {
        engineIds[i] = PLUGIN_FIRST_ENGINE_ID + i;
    }
    *numEngines = PLUGIN_NUM_ENGINES;
}

void getApplicableEngineIds([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                            [[maybe_unused]] const hipdnnPluginConstData_t* opGraph,
                            int64_t* engineIds,
                            uint32_t maxEngines,
                            uint32_t* numEngines)
{
    // TODO Implement actual logic to determine applicable engine IDs.
    // Now we just return a fixed set of engine IDs.
    for(uint32_t i = 0; i < maxEngines && i < PLUGIN_NUM_ENGINES; ++i)
    {
        engineIds[i] = PLUGIN_FIRST_ENGINE_ID + i;
    }
    *numEngines = PLUGIN_NUM_ENGINES;
}

void checkEngineIdValidity(int64_t engineId)
{
    // Check if the engineId is within the valid range.
    if(engineId < PLUGIN_FIRST_ENGINE_ID || engineId >= PLUGIN_FIRST_ENGINE_ID + PLUGIN_NUM_ENGINES)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "invalid engine_id");
    }
}

void getEngineDetails([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                      [[maybe_unused]] int64_t engineId,
                      [[maybe_unused]] const hipdnnPluginConstData_t* opGraph,
                      hipdnnPluginConstData_t* engineDetails)
{
    // TODO Implement actual logic
    // For now, we just allocate some memory for engine details.
    size_t size = 1024;
    engineDetails->ptr = new uint8_t[size];
    engineDetails->size = size;
}

void destroyEngineDetails([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                          hipdnnPluginConstData_t* engineDetails)
{
    delete[] static_cast<const uint8_t*>(engineDetails->ptr);
    engineDetails->ptr = nullptr;
    engineDetails->size = 0;
}

size_t getWorkspaceSize([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                        [[maybe_unused]] const hipdnnPluginConstData_t* engineConfig,
                        [[maybe_unused]] const hipdnnPluginConstData_t* opGraph)
{
    // TODO Implement actual logic
    // For now, we just return a fixed workspace size.
    return 4096;
}

size_t getWorkspaceSize([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                        [[maybe_unused]] hipdnnEnginePluginExecutionContext_t executionContext)
{
    // TODO Implement actual logic
    // For now, we just return a fixed workspace size.
    return 2048;
}

hipdnnEnginePluginExecutionContext_t
    createExecutionContext([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                           [[maybe_unused]] const hipdnnPluginConstData_t* engineConfig,
                           [[maybe_unused]] const hipdnnPluginConstData_t* opGraph)
{
    auto executionContext = new HipdnnEnginePluginExecutionContext{0};
    return executionContext;
}

void destroyExecutionContext([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                             hipdnnEnginePluginExecutionContext_t executionContext)
{
    // Free the memory allocated for the execution context.
    delete executionContext;
}

void executeOpGraph([[maybe_unused]] hipdnnEnginePluginHandle_t handle,
                    [[maybe_unused]] hipdnnEnginePluginExecutionContext_t executionContext,
                    [[maybe_unused]] void* workspace,
                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                    uint32_t numDeviceBuffers)
{
    if(numDeviceBuffers != 2)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "expected 2 device buffers, got "
                                        + std::to_string(numDeviceBuffers));
    }

    runEngine(static_cast<const uint32_t*>(deviceBuffers[0].ptr),
              static_cast<uint32_t*>(deviceBuffers[1].ptr),
              GPU_DATA_SIZE);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <unordered_map>

#include "ExampleProviderContext.hpp"
#include "ExampleProviderSettings.hpp"

namespace example_provider
{
class ExampleProviderContainer;
}

// TEMPLATE ADAPTATION: Copy this file and rename the class. The stream management and FlatBuffer
// detached buffer lifetime methods are framework plumbing; no operation-specific changes are needed.
// If your plugin requires additional per-session state, add members here.

/// Handle for the plugin.
///
/// Inherits from HipdnnEnginePluginHandle for opaque pointer compatibility.
/// Manages the HIP stream, plugin container, and detached FlatBuffers buffers.
struct ExampleProviderHandle : HipdnnEnginePluginHandle
{
public:
    ExampleProviderHandle() = default;

    ~ExampleProviderHandle() override = default;

    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<example_provider::ExampleProviderContainer> container;

    // Defined in ExampleProviderHandle.cpp to avoid circular dependency
    hipdnn_plugin_sdk::
        EngineManager<ExampleProviderHandle, ExampleProviderSettings, ExampleProviderContext>&
        getEngineManager();

    // When requested by the user, engine details are serialized as a
    // FlatBuffer (ExampleProviderEngine::getDetails()) and a pointer to the heap
    // memory for the completed FlatBuffer is returned to the backend. The
    // Flatbuffer memory needs to remain allocated until the user releases the
    // engine details, at which point the FlatBuffer object can be destroyed.
    //
    // The storeEngineDetailsDetachedBuffer() and removeEngineDetailsDetachedBuffer()
    // functions below are provided to assist with managing the lifetime of these
    // FlatBuffer engine detail objects. These functions can be copied/used as-is.
    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_PLUGIN_LOG_INFO("Storing detached buffer at address: " << ptr);
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        HIPDNN_PLUGIN_LOG_INFO("Removing detached buffer at address: " << ptr);

        auto it = _engineDetailsBuffers.find(ptr);
        if(it != _engineDetailsBuffers.end())
        {
            _engineDetailsBuffers.erase(it);
        }
        else
        {
            HIPDNN_PLUGIN_LOG_WARN(
                "No detached buffer found at address: "
                << ptr
                << ". Could not remove engine "
                   "details. Ensure you "
                   "are using the same hipdnn handle you used for engine details creation");
        }
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

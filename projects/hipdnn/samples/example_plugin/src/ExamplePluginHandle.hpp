// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <unordered_map>

#include "ExamplePluginContext.hpp"
#include "ExamplePluginSettings.hpp"

namespace example_plugin
{
class ExamplePluginContainer;
}

/// Handle for the example plugin.
///
/// Inherits from HipdnnEnginePluginHandle for opaque pointer compatibility.
/// Manages the HIP stream, plugin container, and detached FlatBuffers buffers.
// NOLINTBEGIN
struct ExamplePluginHandle : HipdnnEnginePluginHandle
{
public:
    ExamplePluginHandle() = default;

    ~ExamplePluginHandle() override = default;

    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<example_plugin::ExamplePluginContainer> container;

    // Defined in ExamplePluginHandle.cpp to avoid circular dependency
    hipdnn_plugin_sdk::
        EngineManager<ExamplePluginHandle, ExamplePluginSettings, ExamplePluginContext>&
        getEngineManager();

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

// NOLINTEND

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltContainer.hpp"
#include "HipblasltUtils.hpp"

// NOLINTBEGIN
struct HipdnnEnginePluginHandle
{
public:
    virtual ~HipdnnEnginePluginHandle() = default;

    hipblasLtHandle_t hipblasltHandle = nullptr;

    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<hipblaslt_plugin::HipblasltContainer> hipblasltContainer;
    hipblaslt_plugin::EngineManager& getEngineManager()
    {
        return hipblasltContainer->getEngineManager();
    }

    // The engine-details map is shared by every thread querying engine details through this
    // handle (get_knobs_for_engine takes no handle, so threads that each own a handle still
    // reach the handle of the graph), so every access takes _engineDetailsBuffersMutex.
    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_PLUGIN_LOG_INFO("Storing detached buffer at address: " << ptr);
        const std::lock_guard<std::mutex> lock(_engineDetailsBuffersMutex);
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        HIPDNN_PLUGIN_LOG_INFO("Removing detached buffer at address: " << ptr);

        bool removed = false;
        {
            const std::lock_guard<std::mutex> lock(_engineDetailsBuffersMutex);
            removed = _engineDetailsBuffers.erase(ptr) != 0;
        }

        if(!removed)
        {
            HIPDNN_PLUGIN_LOG_WARN(
                "No detached buffer found at address: "
                << ptr
                << ". Could not remove engine "
                   "details. Ensure you "
                   "are using the same hipdnn handle you used for engine details creation");
        }
    }

    size_t engineDetailsDetachedBufferCount() const
    {
        const std::lock_guard<std::mutex> lock(_engineDetailsBuffersMutex);
        return _engineDetailsBuffers.size();
    }

private:
    hipStream_t _stream = nullptr;
    mutable std::mutex _engineDetailsBuffersMutex;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

// NOLINTEND

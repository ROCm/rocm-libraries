// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <unordered_map>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

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

    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_LOG_INFO("Storing detached buffer at address: {:p}", ptr);
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        HIPDNN_LOG_INFO("Removing detached buffer at address: {:p}", ptr);

        auto it = _engineDetailsBuffers.find(ptr);
        if(it != _engineDetailsBuffers.end())
        {
            _engineDetailsBuffers.erase(it);
        }
        else
        {
            HIPDNN_LOG_WARN("No detached buffer found at address: {:p}. Could not remove engine "
                            "details. Ensure you "
                            "are using the same hipdnn handle you used for engine details creation",
                            ptr);
        }
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

// NOLINTEND

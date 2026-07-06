// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <miopen/miopen.h>
#include <unordered_map>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{
class MiopenContainer;
}

/**
 * @brief MIOpen plugin handle.
 *
 * Inherits from HipdnnEnginePluginHandle for opaque pointer compatibility.
 * Manages the MIOpen library handle, HIP stream, and plugin container.
 */
// NOLINTBEGIN
struct HipdnnMiopenHandle : HipdnnEnginePluginHandle
{
public:
    HipdnnMiopenHandle(const HipdnnMiopenHandle&) = delete;
    HipdnnMiopenHandle& operator=(const HipdnnMiopenHandle&) = delete;
    HipdnnMiopenHandle(HipdnnMiopenHandle&&) = delete;
    HipdnnMiopenHandle& operator=(HipdnnMiopenHandle&&) = delete;

    HipdnnMiopenHandle()
    {
        miopenStatus_t status = miopenCreate(&miopenHandle);
        if(status != miopenStatusSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "Failed to create MIOpen handle");
        }
    }

    ~HipdnnMiopenHandle() override
    {
        if(miopenHandle != nullptr)
        {
            miopenStatus_t status = miopenDestroy(miopenHandle);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR("Failed to destroy MIOpen handle");
            }
        }
    }

    miopenHandle_t miopenHandle = nullptr;

    void setStream(hipStream_t stream)
    {
        THROW_ON_MIOPEN_FAILURE(miopenSetStream(miopenHandle, stream));
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<miopen_plugin::MiopenContainer> container;

    // Defined in HipdnnMiopenHandle.cpp to avoid circular dependency
    hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>&
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

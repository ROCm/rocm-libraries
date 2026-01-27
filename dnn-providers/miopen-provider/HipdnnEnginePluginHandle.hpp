// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <miopen/miopen.h>
#include <unordered_map>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenContainer.hpp"
#include "MiopenUtils.hpp"

// NOLINTBEGIN
struct HipdnnEnginePluginHandle
{
public:
    HipdnnEnginePluginHandle()
    {
        miopenStatus_t status = miopenCreate(&miopenHandle);
        if(status != miopenStatusSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "Failed to create MIOpen handle");
        }
    }

    virtual ~HipdnnEnginePluginHandle()
    {
        if(miopenHandle != nullptr)
        {
            miopenStatus_t status = miopenDestroy(miopenHandle);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("Failed to destroy MIOpen handle");
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

    std::shared_ptr<miopen_legacy_plugin::MiopenContainer> container;
    hipdnn_plugin_sdk::EngineManager& getEngineManager()
    {
        return container->getEngineManager();
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

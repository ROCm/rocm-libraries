// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <unordered_map>
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
#include <mutex>
#include <rocblas/rocblas.h>
#endif

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltContainer.hpp"
#include "HipblasltUtils.hpp"

// NOLINTBEGIN
struct HipdnnEnginePluginHandle
{
public:
    virtual ~HipdnnEnginePluginHandle()
    {
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
        if(_rocblasHandle != nullptr)
        {
            rocblas_destroy_handle(_rocblasHandle);
        }
#endif
    }

    hipblasLtHandle_t hipblasltHandle = nullptr;

#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    /// The rocBLAS handle, created on first use: creating one loads rocBLAS's kernel library,
    /// a cost a handle that only ever serves hipBLASLt graphs should not pay.
    rocblas_handle rocblasHandle() const
    {
        std::call_once(_rocblasOnce, [this] {
            if(rocblas_create_handle(&_rocblasHandle) != rocblas_status_success)
            {
                _rocblasHandle = nullptr;
            }
        });
        if(_rocblasHandle == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "Failed to create rocBLAS handle");
        }
        return _rocblasHandle;
    }
#endif

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
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    mutable std::once_flag _rocblasOnce;
    mutable rocblas_handle _rocblasHandle = nullptr;
#endif
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

// NOLINTEND

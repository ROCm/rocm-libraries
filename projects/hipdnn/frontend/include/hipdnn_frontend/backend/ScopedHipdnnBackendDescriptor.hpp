// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/backend/BackendLoggingHelpers.hpp>
#include <hipdnn_frontend/backend/BackendWrapper.hpp>

namespace hipdnn_frontend
{

class ScopedHipdnnBackendDescriptor
{
private:
    hipdnnBackendDescriptor_t _descriptor;
    bool _valid;

    static void logBackendError(const std::string& errorString, const hipdnnStatus_t status)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg;
        hipdnn_frontend::hipdnnBackend()->getLastErrorString(backendErrMsg.data(),
                                                             backendErrMsg.size());
        HIPDNN_FE_LOG_ERROR(
            "{}: {}. Backend error string: {}", errorString, status, backendErrMsg.data());
    }

public:
    ScopedHipdnnBackendDescriptor()
        : _descriptor(nullptr)
        , _valid(false)
    {
    }

    ScopedHipdnnBackendDescriptor(hipdnnBackendDescriptor_t descriptor)
        : _descriptor(descriptor)
        , _valid(true)
    {
    }

    explicit ScopedHipdnnBackendDescriptor(hipdnnBackendDescriptorType_t descriptorType)
    {
        auto status = hipdnnBackend()->backendCreateDescriptor(descriptorType, &_descriptor);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            _descriptor = nullptr;
            logBackendError("Failed to create backend descriptor", status);
        }
    }

    explicit ScopedHipdnnBackendDescriptor(const uint8_t* serializedGraph, size_t graphByteSize)
    {
        auto status = hipdnnBackend()->backendCreateAndDeserializeGraphExt(
            &_descriptor, serializedGraph, graphByteSize);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            _descriptor = nullptr;
            logBackendError("Failed to create and deserialize graph", status);
        }
    }

    ~ScopedHipdnnBackendDescriptor()
    {
        if(_valid && _descriptor != nullptr)
        {
            auto status = hipdnnBackend()->backendDestroyDescriptor(_descriptor);

            if(status != HIPDNN_STATUS_SUCCESS)
            {
                logBackendError("Failed to destroy backend descriptor", status);
            }
        }
    }

    ScopedHipdnnBackendDescriptor(const ScopedHipdnnBackendDescriptor&) = delete;
    ScopedHipdnnBackendDescriptor& operator=(const ScopedHipdnnBackendDescriptor&) = delete;

    ScopedHipdnnBackendDescriptor(ScopedHipdnnBackendDescriptor&& other) noexcept
        : _descriptor(other._descriptor)
        , _valid(other._valid)
    {
        other._descriptor = nullptr;
        other._valid = false;
    }

    ScopedHipdnnBackendDescriptor& operator=(ScopedHipdnnBackendDescriptor&& other) noexcept
    {
        if(this != &other)
        {
            if(_valid && _descriptor != nullptr)
            {
                auto status = hipdnnBackend()->backendDestroyDescriptor(_descriptor);
                if(status != HIPDNN_STATUS_SUCCESS)
                {
                    logBackendError("Failed to destroy backend descriptor during move assignment",
                                    status);
                }
            }
            _descriptor = other._descriptor;
            _valid = other._valid;
            other._descriptor = nullptr;
            other._valid = false;
        }
        return *this;
    }

    bool valid() const
    {
        return _valid;
    }

    const hipdnnBackendDescriptor_t& get() const
    {
        return _descriptor;
    }
};

}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstring>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

class DeviceBuffer
{
private:
    void* _devicePtr = nullptr;
    size_t _sizeBytes = 0;

public:
    DeviceBuffer(size_t sizeBytes)
        : _sizeBytes(sizeBytes)
    {
        if(_sizeBytes > 0)
        {
            const auto status = hipMalloc(&_devicePtr, _sizeBytes);
            if(status != hipSuccess)
            {
                throw std::runtime_error("Failed to allocate device memory: "
                                         + std::string(hipGetErrorString(status)));
            }
        }
    }

    ~DeviceBuffer()
    {
        if(_devicePtr != nullptr)
        {
            (void)hipFree(_devicePtr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _devicePtr(other._devicePtr)
        , _sizeBytes(other._sizeBytes)
    {
        other._devicePtr = nullptr;
        other._sizeBytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(_devicePtr != nullptr)
            {
                (void)hipFree(_devicePtr);
            }
            _devicePtr = other._devicePtr;
            _sizeBytes = other._sizeBytes;
            other._devicePtr = nullptr;
            other._sizeBytes = 0;
        }
        return *this;
    }

    void copyFromHost(const void* hostPtr)
    {
        if(_devicePtr == nullptr || hostPtr == nullptr)
        {
            throw std::runtime_error("Invalid pointers for copy operation");
        }
        const auto status = hipMemcpy(_devicePtr, hostPtr, _sizeBytes, hipMemcpyHostToDevice);
        if(status != hipSuccess)
        {
            throw std::runtime_error("Failed to copy from host: "
                                     + std::string(hipGetErrorString(status)));
        }
    }

    void copyToHost(void* hostPtr)
    {
        if(_devicePtr == nullptr || hostPtr == nullptr)
        {
            throw std::runtime_error("Invalid pointers for copy operation");
        }
        const auto status = hipMemcpy(hostPtr, _devicePtr, _sizeBytes, hipMemcpyDeviceToHost);
        if(status != hipSuccess)
        {
            throw std::runtime_error("Failed to copy to host: "
                                     + std::string(hipGetErrorString(status)));
        }
    }

    void* ptr()
    {
        return _devicePtr;
    }
    size_t size() const
    {
        return _sizeBytes;
    }

    // Fill with zeros
    void zeros()
    {
        if(_devicePtr != nullptr)
        {
            const auto status = hipMemset(_devicePtr, 0, _sizeBytes);
            if(status != hipSuccess)
            {
                throw std::runtime_error("Failed to zero memory: "
                                         + std::string(hipGetErrorString(status)));
            }
        }
    }
};

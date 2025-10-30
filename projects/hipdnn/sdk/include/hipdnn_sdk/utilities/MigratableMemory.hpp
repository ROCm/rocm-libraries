// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/utilities/Allocators.hpp>
#include <memory>
#include <stdexcept>

namespace hipdnn_sdk
{
namespace utilities
{

enum class MemoryLocation
{
    HOST,
    DEVICE,
    BOTH,
    NONE
};

// NOLINTBEGIN(portability-template-virtual-member-function)

class IMigratableMemory
{
public:
    virtual ~IMigratableMemory() = default;

    virtual void* deviceData() = 0;
    virtual void* deviceDataAsync() = 0;

    virtual void markHostModified() = 0;
    virtual void markDeviceModified() = 0;

    virtual size_t count() const = 0;
    virtual bool empty() const = 0;
    virtual MemoryLocation location() const = 0;

    virtual void resize(size_t newCount) = 0;
    virtual void clear() = 0;
};

template <typename T>
class MigratableMemoryBase : public IMigratableMemory
{
public:
    ~MigratableMemoryBase() override = default;

    virtual T* hostData() = 0;
    virtual T* hostDataAsync() = 0;
    virtual const T* hostData() const = 0;
    virtual const T* hostDataAsync() const = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class MigratableMemory : public MigratableMemoryBase<T>
{
    static_assert(std::is_base_of_v<IHostAllocator<T>, HostAlloc>,
                  "HostAlloc must derive from IHostAllocator<T>");
    static_assert(std::is_base_of_v<IDeviceAllocator<T>, DeviceAlloc>,
                  "DeviceAlloc must derive from IDeviceAllocator<T>");

public:
    explicit MigratableMemory(size_t count = 0, hipStream_t stream = nullptr)
        : _count(count)
        , _itemSize(sizeof(T))
        , _totalSize(count * _itemSize)
        , _stream(stream)
    {
        if(count > 0)
        {
            allocateHost();
        }
    }

    ~MigratableMemory() override
    {
        cleanup();
    }

    MigratableMemory(MigratableMemory&& other) noexcept
        : _hostPtr(other._hostPtr)
        , _devicePtr(other._devicePtr)
        , _count(other._count)
        , _itemSize(other._itemSize)
        , _totalSize(other._totalSize)
        , _currentLocation(other._currentLocation)
        , _hostValid(other._hostValid)
        , _deviceValid(other._deviceValid)
        , _stream(other._stream)
        , _hostAllocator(std::move(other._hostAllocator))
        , _deviceAllocator(std::move(other._deviceAllocator))
    {
        other._hostPtr = nullptr;
        other._devicePtr = nullptr;
        other._count = 0;
        other._itemSize = 0;
        other._totalSize = 0;
        other._currentLocation = MemoryLocation::NONE;
        other._stream = nullptr;
        other._hostValid = false;
        other._deviceValid = false;
    }

    MigratableMemory& operator=(MigratableMemory&& other) noexcept
    {
        if(this != &other)
        {
            cleanup();
            _hostPtr = other._hostPtr;
            _devicePtr = other._devicePtr;
            _count = other._count;
            _itemSize = other._itemSize;
            _totalSize = other._totalSize;
            _currentLocation = other._currentLocation;
            _hostValid = other._hostValid;
            _deviceValid = other._deviceValid;
            _stream = other._stream;
            _deviceAllocator = std::move(other._deviceAllocator);
            _hostAllocator = std::move(other._hostAllocator);

            other._hostPtr = nullptr;
            other._devicePtr = nullptr;
            other._count = 0;
            other._itemSize = 0;
            other._totalSize = 0;
            other._currentLocation = MemoryLocation::NONE;
            other._stream = nullptr;
            other._hostValid = false;
            other._deviceValid = false;
        }
        return *this;
    }

    MigratableMemory(const MigratableMemory&) = delete;
    MigratableMemory& operator=(const MigratableMemory&) = delete;

    void resize(size_t newCount) override
    {
        cleanup();
        _count = newCount;
        _totalSize = newCount * _itemSize;
        _currentLocation = MemoryLocation::NONE;
        _hostValid = false;
        _deviceValid = false;
        if(newCount > 0)
        {
            allocateHost();
        }
    }

    // Get host pointer (migrates if needed)
    T* hostData() override
    {
        ensureHostValid();
        return static_cast<T*>(_hostPtr);
    }

    T* hostDataAsync() override
    {
        ensureHostValid(true);
        return static_cast<T*>(_hostPtr);
    }

    // Get host pointer (migrates if needed)
    const T* hostData() const override
    {
        ensureConstHostValid();
        return static_cast<T*>(_hostPtr);
    }

    const T* hostDataAsync() const override
    {
        ensureConstHostValid();
        return static_cast<T*>(_hostPtr);
    }

    void* deviceData() override
    {
        ensureDeviceValid();
        return static_cast<T*>(_devicePtr);
    }

    void* deviceDataAsync() override
    {
        ensureDeviceValid(true);
        return static_cast<T*>(_devicePtr);
    }

    // Mark memory as modified on host
    void markHostModified() override
    {
        _hostValid = true;
        _deviceValid = false;
        _currentLocation = MemoryLocation::HOST;
    }

    // Mark memory as modified on device
    void markDeviceModified() override
    {
        _deviceValid = true;
        _hostValid = false;
        _currentLocation = MemoryLocation::DEVICE;
    }

    size_t count() const override
    {
        return _count;
    }

    bool empty() const override
    {
        return _count == 0;
    }

    MemoryLocation location() const override
    {
        return _currentLocation;
    }

    void clear() override
    {
        cleanup();
        _count = 0;
        _itemSize = 0;
        _totalSize = 0;
        _currentLocation = MemoryLocation::NONE;
        _hostValid = false;
        _deviceValid = false;
    }

private:
    static void throwOnError(hipError_t err, const char* msg)
    {
        if(err != hipSuccess)
        {
            throw std::runtime_error(msg);
        }
    }

    static void logOnError(hipError_t err, [[maybe_unused]] const char* msg)
    {
        if(err != hipSuccess)
        {
            HIPDNN_LOG_ERROR("{}: HIP error: {}", msg, hipGetErrorString(err));
        }
    }

    void allocateHost()
    {
        if((_hostPtr == nullptr) && _count > 0)
        {
            _hostPtr = _hostAllocator.allocate(_count);
            _hostValid = true;
            _currentLocation = MemoryLocation::HOST;
        }
    }

    void allocateDevice()
    {
        if((_devicePtr == nullptr) && _count > 0)
        {
            _devicePtr = _deviceAllocator.allocate(_count);
        }
    }

    void ensureConstHostValid() const
    {
        if((_hostPtr == nullptr) && _count > 0)
        {
            throw std::runtime_error("Host memory not allocated.");
        }

        if(!_hostValid && _deviceValid && (_devicePtr != nullptr))
        {
            throw std::runtime_error(
                "Host memory is out of date and requires non-const access to update.");
        }
    }

    void ensureHostValid(bool async = false)
    {
        if(_count == 0)
        {
            return;
        }

        allocateHost();

        if(!_hostValid && _deviceValid && (_devicePtr != nullptr))
        {
            if(async)
            {
                throwOnError(hipMemcpyAsync(
                                 _hostPtr, _devicePtr, _totalSize, hipMemcpyDeviceToHost, _stream),
                             "Failed to copy from device to host");
            }
            else
            {
                throwOnError(hipMemcpyWithStream(
                                 _hostPtr, _devicePtr, _totalSize, hipMemcpyDeviceToHost, _stream),
                             "Failed to copy from device to host");
            }
            _hostValid = true;
            _currentLocation = MemoryLocation::BOTH;
        }
    }

    void ensureDeviceValid(bool async = false)
    {
        if(_count == 0)
        {
            return;
        }

        allocateDevice();

        if(!_deviceValid && _hostValid && (_hostPtr != nullptr))
        {
            if(async)
            {
                throwOnError(hipMemcpyAsync(
                                 _devicePtr, _hostPtr, _totalSize, hipMemcpyHostToDevice, _stream),
                             "Failed to copy from host to device");
            }
            else
            {
                throwOnError(hipMemcpyWithStream(
                                 _devicePtr, _hostPtr, _totalSize, hipMemcpyHostToDevice, _stream),
                             "Failed to copy from host to device");
            }
            _deviceValid = true;
            _currentLocation = MemoryLocation::BOTH;
        }
    }

    void cleanup()
    {
        if(_hostPtr != nullptr)
        {
            _hostAllocator.deallocate(static_cast<T*>(_hostPtr), _count);
            _hostPtr = nullptr;
        }
        if(_devicePtr != nullptr)
        {
            _deviceAllocator.deallocate(static_cast<T*>(_devicePtr), _count);
            _devicePtr = nullptr;
        }
        _hostValid = false;
        _deviceValid = false;
        _currentLocation = MemoryLocation::NONE;
    }

    void* _hostPtr{nullptr};
    void* _devicePtr{nullptr};
    size_t _count;
    size_t _itemSize;
    size_t _totalSize;
    MemoryLocation _currentLocation{MemoryLocation::NONE};
    bool _hostValid{false};
    bool _deviceValid{false};
    hipStream_t _stream{nullptr};
    HostAlloc _hostAllocator;
    DeviceAlloc _deviceAllocator;
};

} // namespace utilities
} // namespace hipdnn_sdk

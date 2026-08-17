// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>

#include <cstddef>
#include <stdexcept>

namespace hipdnn_gpu_ref::detail
{

template <class T>
class ShallowDeviceOnlyMigratableMemory : public hipdnn_data_sdk::utilities::MigratableMemoryBase<T>
{
public:
    explicit ShallowDeviceOnlyMigratableMemory(void* deviceMemory = nullptr, size_t count = 0)
        : _deviceMemory(deviceMemory)
        , _count(count)
    {
    }

    ShallowDeviceOnlyMigratableMemory(ShallowDeviceOnlyMigratableMemory&& other) noexcept
        : _deviceMemory(other._deviceMemory)
        , _count(other._count)
    {
        other._deviceMemory = nullptr;
        other._count = 0;
    }

    ShallowDeviceOnlyMigratableMemory& operator=(ShallowDeviceOnlyMigratableMemory&& other) noexcept
    {
        if(this != &other)
        {
            _deviceMemory = other._deviceMemory;
            _count = other._count;
            other._deviceMemory = nullptr;
            other._count = 0;
        }
        return *this;
    }

    ShallowDeviceOnlyMigratableMemory(const ShallowDeviceOnlyMigratableMemory&) = delete;
    ShallowDeviceOnlyMigratableMemory& operator=(const ShallowDeviceOnlyMigratableMemory&) = delete;

    T* hostData() override
    {
        throwNotSupported();
        return nullptr;
    }
    T* hostDataAsync() override
    {
        throwNotSupported();
        return nullptr;
    }
    const T* hostData() const override
    {
        throwNotSupported();
        return nullptr;
    }
    const T* hostDataAsync() const override
    {
        throwNotSupported();
        return nullptr;
    }
    void* deviceData() override
    {
        return _deviceMemory;
    }
    void* deviceDataAsync() override
    {
        return _deviceMemory;
    }

    void markHostModified() override
    {
        throwNotSupported();
    }
    void markDeviceModified() override
    {
        // GPU ops call this to indicate device memory was modified — no-op for non-owning memory
    }

    size_t count() const override
    {
        return _count;
    }
    bool empty() const override
    {
        return _count == 0;
    }
    hipdnn_data_sdk::utilities::MemoryLocation location() const override
    {
        return hipdnn_data_sdk::utilities::MemoryLocation::DEVICE;
    }

    void resize([[maybe_unused]] size_t size) override
    {
        throwNotSupported();
    }
    void clear() override
    {
        throwNotSupported();
    }

private:
    static void throwNotSupported()
    {
        throw std::runtime_error(
            "ShallowDeviceOnlyMigratableMemory only supports device data memory access.");
    }

    void* _deviceMemory;
    size_t _count;
};

} // namespace hipdnn_gpu_ref::detail

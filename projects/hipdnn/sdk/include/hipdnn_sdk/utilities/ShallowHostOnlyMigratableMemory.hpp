// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>

namespace hipdnn_sdk
{
namespace utilities
{

template <class T>
class ShallowHostOnlyMigratableMemory : public MigratableMemoryBase<T>
{
public:
    explicit ShallowHostOnlyMigratableMemory(void* shallowMemory = nullptr, size_t count = 0)
        : _shallowMemory(static_cast<T*>(shallowMemory))
        , _count(count)
    {
    }

    ShallowHostOnlyMigratableMemory(ShallowHostOnlyMigratableMemory&& other) noexcept
        : _shallowMemory(other._shallowMemory)
        , _count(other._count)
    {
        other._shallowMemory = nullptr;
        other._count = 0;
    }

    ShallowHostOnlyMigratableMemory& operator=(ShallowHostOnlyMigratableMemory&& other) noexcept
    {
        if(this != &other)
        {
            _shallowMemory = other._shallowMemory;
            _count = other._count;
            other._shallowMemory = nullptr;
            other._count = 0;
        }
        return *this;
    }

    ShallowHostOnlyMigratableMemory(const ShallowHostOnlyMigratableMemory&) = delete;
    ShallowHostOnlyMigratableMemory& operator=(const ShallowHostOnlyMigratableMemory&) = delete;

    T* hostData() override
    {
        return _shallowMemory;
    }
    T* hostDataAsync() override
    {
        return _shallowMemory;
    }
    const T* hostData() const override
    {
        return _shallowMemory;
    }
    const T* hostDataAsync() const override
    {
        return _shallowMemory;
    }
    void* deviceData() override
    {
        throwNotSupported();
        return nullptr;
    }
    void* deviceDataAsync() override
    {
        throwNotSupported();
        return nullptr;
    }

    void markHostModified() override
    {
        // reference ops call this function to indicate that the host memory has been modified
        // we dont want to throw in this case
    }
    void markDeviceModified() override
    {
        throwNotSupported();
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
        return MemoryLocation::HOST;
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
            "ShallowHostOnlyMigratableMemory only supports host data memory access. Resizes and "
            "allocations need to be done using MigratableMemeory.");
    }

    T* _shallowMemory;
    size_t _count;
};

}
}

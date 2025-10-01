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
class ShallowHostOnlyMigratableMemory : public IMigratableMemory<T>
{
public:
    ShallowHostOnlyMigratableMemory(void* memory, size_t count)
        : _memory(static_cast<T*>(memory))
        , _count(count)
    {
    }

    T* hostData() override
    {
        return _memory;
    }
    T* hostDataAsync() override
    {
        return _memory;
    }
    const T* hostData() const override
    {
        return _memory;
    }
    const T* hostDataAsync() const override
    {
        return _memory;
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

    T* _memory;
    size_t _count;
};

}
}

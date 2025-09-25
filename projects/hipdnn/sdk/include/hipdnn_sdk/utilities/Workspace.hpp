// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <type_traits>

#include <hipdnn_sdk/utilities/Allocators.hpp>

namespace hipdnn_sdk
{
namespace utilities
{

template <class DeviceAlloc = DeviceAllocator<char>>
class Workspace
{
    static_assert(std::is_base_of_v<IDeviceAllocator<char>, DeviceAlloc>,
                  "DeviceAlloc must derive from IDeviceAllocator<char>");

public:
    Workspace(size_t size)
        : _size(size)
    {
        if(size != 0)
        {
            _ptr = _allocator.allocate(size);
        }
    }

    ~Workspace()
    {
        if(_ptr != nullptr)
        {
            _allocator.deallocate(static_cast<char*>(_ptr), _size);
        }
    }

    Workspace(Workspace&& other) noexcept
        : _ptr(other._ptr)
        , _size(other._size)
    {
        other._ptr = nullptr;
        other._size = 0;
    }

    Workspace& operator=(Workspace&& other) noexcept
    {
        if(this != &other)
        {
            if(_ptr != nullptr)
            {
                _allocator.deallocate(static_cast<char*>(_ptr), _size);
            }

            _ptr = other._ptr;
            _size = other._size;

            other._ptr = nullptr;
            other._size = 0;
        }
        return *this;
    }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    void* get() const
    {
        return _ptr;
    }

private:
    DeviceAlloc _allocator;
    void* _ptr = nullptr;
    size_t _size;
};

} // namespace utilities
} // namespace hipdnn_sdk

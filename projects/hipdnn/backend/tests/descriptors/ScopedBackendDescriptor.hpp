// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>

#include <utility>

namespace hipdnn_backend::test_utilities
{

/// RAII wrapper for hipdnnBackendDescriptor_t in backend tests.
/// Calls hipdnnBackendDestroyDescriptor on destruction if non-null.
class ScopedBackendDescriptor
{
public:
    ScopedBackendDescriptor()
        : _desc(nullptr)
    {
    }

    explicit ScopedBackendDescriptor(hipdnnBackendDescriptor_t desc)
        : _desc(desc)
    {
    }

    ~ScopedBackendDescriptor()
    {
        if(_desc != nullptr)
        {
            hipdnnBackendDestroyDescriptor(_desc);
        }
    }

    ScopedBackendDescriptor(const ScopedBackendDescriptor&) = delete;
    ScopedBackendDescriptor& operator=(const ScopedBackendDescriptor&) = delete;

    ScopedBackendDescriptor(ScopedBackendDescriptor&& other) noexcept
        : _desc(other._desc)
    {
        other._desc = nullptr;
    }

    ScopedBackendDescriptor& operator=(ScopedBackendDescriptor&& other) noexcept
    {
        if(this != &other)
        {
            if(_desc != nullptr)
            {
                hipdnnBackendDestroyDescriptor(_desc);
            }
            _desc = other._desc;
            other._desc = nullptr;
        }
        return *this;
    }

    hipdnnBackendDescriptor_t get() const
    {
        return _desc;
    }

    hipdnnBackendDescriptor_t release()
    {
        auto d = _desc;
        _desc = nullptr;
        return d;
    }

private:
    hipdnnBackendDescriptor_t _desc;
};

} // namespace hipdnn_backend::test_utilities

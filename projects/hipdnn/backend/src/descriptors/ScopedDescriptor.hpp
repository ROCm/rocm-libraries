// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_backend.h"

namespace hipdnn_backend
{
struct ScopedDescriptor
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ScopedDescriptor() = default;
    explicit ScopedDescriptor(hipdnnBackendDescriptor_t desc)
        : descriptor(desc)
    {
    }
    ~ScopedDescriptor()
    {
        delete descriptor;
    }
    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;
    ScopedDescriptor(ScopedDescriptor&& other) noexcept
    {
        descriptor = other.descriptor;
        other.descriptor = nullptr;
    }
    ScopedDescriptor& operator=(ScopedDescriptor&& other) noexcept
    {
        if(this != &other)
        {
            delete descriptor;
            descriptor = other.descriptor;
            other.descriptor = nullptr;
        }
        return *this;
    }
    hipdnnBackendDescriptor_t get() const
    {
        return descriptor;
    }
    hipdnnBackendDescriptor_t* getPtr()
    {
        return &descriptor;
    }
};
}

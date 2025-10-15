// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// ScopedResource is a utility class that manages a resource with a custom destructor.

#include <functional>
#include <utility>

namespace hipdnn_sdk::utilities
{

template <typename T>
class ScopedResource
{
public:
    ScopedResource() = default;

    // Accept any callable type for destructor, convert to std::function
    template <typename Callable>
    ScopedResource(T resource, Callable&& destructor)
        : _resource(resource)
        , _destructor(std::forward<Callable>(destructor))
        , _empty(false)
    {
    }

    ~ScopedResource()
    {
        if(!_empty)
        {
            _destructor(_resource);
        }
    }

    // Prevent copying
    ScopedResource(const ScopedResource&) = delete;
    ScopedResource& operator=(const ScopedResource&) = delete;

    ScopedResource(ScopedResource&& other) noexcept
        : _resource(std::move(other._resource))
        , _destructor(std::move(other._destructor))
        , _empty(other._empty)
    {
        other._empty = true;
    }

    ScopedResource& operator=(ScopedResource&& other) noexcept
    {
        if(this != &other)
        {
            if(!_empty)
            {
                _destructor(_resource);
            }

            _empty = other._empty;
            if(!other._empty)
            {
                _resource = std::move(other._resource);
                _destructor = std::move(other._destructor);
                other._empty = true;
            }
        }
        return *this;
    }

    bool isEmpty() const
    {
        return _empty;
    }

    T get() const
    {
        return _resource;
    }

private:
    T _resource;
    std::function<void(T)> _destructor;
    bool _empty = true;
};

} // namespace hipdnn_sdk::utilities

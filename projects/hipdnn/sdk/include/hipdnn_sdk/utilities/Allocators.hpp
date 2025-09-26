// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace hipdnn_sdk
{
namespace utilities
{

// NOLINTBEGIN(portability-template-virtual-member-function)

/// @brief Interface for host memory allocators
template <typename T>
class IHostAllocator
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = IHostAllocator<U>;
    };

    virtual ~IHostAllocator() = default;

    /// @brief Allocate memory
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated memory
    /// @throws std::bad_alloc if allocation fails
    [[nodiscard]] virtual T* allocate(std::size_t n) = 0;

    /// @brief Deallocate memory
    /// @param p Pointer to memory to deallocate
    /// @param n Number of elements (may be used by some allocators)
    virtual void deallocate(T* p, std::size_t n) noexcept = 0;

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args)
    {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p)
    {
        p->~U();
    }
};

/// @brief Interface for device memory allocators
template <typename T>
class IDeviceAllocator
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = IDeviceAllocator<U>;
    };

    virtual ~IDeviceAllocator() = default;

    /// @brief Allocate device memory
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated device memory
    /// @throws std::bad_alloc if allocation fails
    [[nodiscard]] virtual T* allocate(std::size_t n) = 0;

    /// @brief Deallocate device memory
    /// @param p Pointer to device memory to deallocate
    /// @param n Number of elements (may be used by some allocators)
    virtual void deallocate(T* p, std::size_t n) noexcept = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

/// @brief Standard host allocator using malloc/free
template <typename T>
class HostAllocator : public IHostAllocator<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = HostAllocator<U>;
    };

    HostAllocator() noexcept = default;
    HostAllocator(const HostAllocator&) noexcept = default;

    template <typename U>
    HostAllocator([[maybe_unused]] const HostAllocator<U>& other) noexcept
    {
    }

    ~HostAllocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        auto ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if(!ptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept override
    {
        std::free(p);
    }
};

template <typename T, typename U>
bool operator==([[maybe_unused]] const HostAllocator<T>& lhs,
                [[maybe_unused]] const HostAllocator<U>& rhs) noexcept
{
    return true;
}

template <typename T, typename U>
bool operator!=([[maybe_unused]] const HostAllocator<T>& lhs,
                [[maybe_unused]] const HostAllocator<U>& rhs) noexcept
{
    return false;
}

/// @brief Pinned host allocator using hipHostMalloc/hipHostFree
template <typename T>
class PinnedHostAllocator : public IHostAllocator<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = PinnedHostAllocator<U>;
    };

    PinnedHostAllocator() noexcept = default;
    PinnedHostAllocator(const PinnedHostAllocator&) noexcept = default;

    template <typename U>
    PinnedHostAllocator([[maybe_unused]] const PinnedHostAllocator<U>& other) noexcept
    {
    }

    ~PinnedHostAllocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, n * sizeof(T));
        if(err != hipSuccess)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept override
    {
        std::ignore = hipHostFree(p);
    }
};

template <typename T, typename U>
bool operator==([[maybe_unused]] const PinnedHostAllocator<T>& lhs,
                [[maybe_unused]] const PinnedHostAllocator<U>& rhs) noexcept
{
    return true;
}

template <typename T, typename U>
bool operator!=([[maybe_unused]] const PinnedHostAllocator<T>& lhs,
                [[maybe_unused]] const PinnedHostAllocator<U>& rhs) noexcept
{
    return false;
}

/// @brief Device allocator using hipMalloc/hipFree
template <typename T>
class DeviceAllocator : public IDeviceAllocator<T>
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <typename U>
    struct rebind // NOLINT(readability-identifier-naming)
    {
        using other = DeviceAllocator<U>;
    };

    DeviceAllocator() noexcept = default;
    DeviceAllocator(const DeviceAllocator&) noexcept = default;

    template <typename U>
    DeviceAllocator([[maybe_unused]] const DeviceAllocator<U>& other) noexcept
    {
    }

    ~DeviceAllocator() override = default;

    [[nodiscard]] T* allocate(std::size_t n) override
    {
        if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, n * sizeof(T));
        if(err != hipSuccess)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept override
    {
        std::ignore = hipFree(p);
    }

    // Note: construct and destroy are not provided for device allocator
    // as they would require host code to run on device memory addresses
    // which is not valid. Device memory should be initialized using kernels.
};

template <typename T, typename U>
bool operator==([[maybe_unused]] const DeviceAllocator<T>& lhs,
                [[maybe_unused]] const DeviceAllocator<U>& rhs) noexcept
{
    return true;
}

template <typename T, typename U>
bool operator!=([[maybe_unused]] const DeviceAllocator<T>& lhs,
                [[maybe_unused]] const DeviceAllocator<U>& rhs) noexcept
{
    return false;
}

} // namespace utilities
} // namespace hipdnn_sdk

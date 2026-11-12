// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_UTILS_DEVICE_PTR_HPP
#define ROCPRIM_UTILS_DEVICE_PTR_HPP

#include "utils.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace common
{

/// \brief An RAII friendly class to manage the memory allocated on device.
///
/// \tparam A Template type used by the class.
template<typename ValueType = void>
class device_ptr
{
public:
    using decay_type = std::decay_t<ValueType>;
    using size_type  = std::size_t;
    using value_type = ValueType;

private:
    // If value_type is void we want to emulate allocating bytes (uchar).
    using value_type_proxy
        = std::conditional_t<std::is_same<decay_type, void>::value, unsigned char, ValueType>;

public:
    static constexpr size_t value_size = sizeof(value_type_proxy);

    device_ptr() : device_raw_ptr_(nullptr), number_of_ele_(0){};

    /// \brief Construct with a pre-allocated memory space.
    device_ptr(size_type pre_alloc_number_of_ele)
        : device_raw_ptr_(nullptr), number_of_ele_(pre_alloc_number_of_ele)
    {
        size_type storage_size = number_of_ele_ * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
    };

    device_ptr(device_ptr const&) = delete;

    device_ptr(device_ptr&& other) noexcept
        : device_raw_ptr_(other.device_raw_ptr_), number_of_ele_(other.number_of_ele_)
    {
        other.leak();
    };

    /// \brief Construct by host vectors with the same sized value_type
    template<typename InValueType, typename Allocator>
    explicit device_ptr(std::vector<InValueType, Allocator> const& data)
        : device_raw_ptr_(nullptr), number_of_ele_(data.size())
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = number_of_ele_ * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpy(device_raw_ptr_, data.data(), storage_size, hipMemcpyHostToDevice));
    }

    template<typename InValueType>
    explicit device_ptr(std::vector<InValueType> const& data, hipStream_t stream)
        : device_raw_ptr_(nullptr), number_of_ele_(data.size())
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = number_of_ele_ * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 data.data(),
                                 storage_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    template<typename InValueType, size_type Size>
    explicit device_ptr(std::array<InValueType, Size> const& data)
        : device_raw_ptr_(nullptr), number_of_ele_(Size)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = Size * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpy(device_raw_ptr_, data.data(), storage_size, hipMemcpyHostToDevice));
    }

    template<typename InValueType, size_type Size>
    explicit device_ptr(std::array<InValueType, Size> const& data, hipStream_t stream)
        : device_raw_ptr_(nullptr), number_of_ele_(Size)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = Size * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 data.data(),
                                 storage_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    template<typename InValueType, typename DeleterType>
    explicit device_ptr(std::unique_ptr<InValueType[], DeleterType> const& uptr, size_type size)
        : device_raw_ptr_(nullptr), number_of_ele_(size)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = size * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpy(device_raw_ptr_, uptr.get(), storage_size, hipMemcpyHostToDevice));
    }

    template<typename InValueType, typename DeleterType>
    explicit device_ptr(std::unique_ptr<InValueType[], DeleterType> const& uptr,
                        size_type                                          size,
                        hipStream_t                                        stream)
        : device_raw_ptr_(nullptr), number_of_ele_(size)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        size_type storage_size = size * value_size;
        HIP_CHECK(common::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 uptr.get(),
                                 storage_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    ~device_ptr()
    {
        free_manually();
    };

    device_ptr& operator=(device_ptr const&) = delete;

    device_ptr& operator=(device_ptr&& other) noexcept
    {
        free_manually();

        device_raw_ptr_ = other.device_raw_ptr_;
        number_of_ele_  = other.number_of_ele_;
        other.leak();

        return *this;
    };

    /// \brief Do copy on the device.
    ///
    /// \return A new `device_ptr` rvalue.
    device_ptr duplicate() const
    {
        device_ptr ret;
        ret.number_of_ele_     = number_of_ele_;
        size_type storage_size = number_of_ele_ * value_size;
        HIP_CHECK(common::hipMallocHelper(&ret.device_raw_ptr_, storage_size));
        HIP_CHECK(
            hipMemcpy(ret.device_raw_ptr_, device_raw_ptr_, storage_size, hipMemcpyDeviceToDevice));
        return ret;
    }

    device_ptr duplicate_async(hipStream_t stream) const
    {
        device_ptr ret;
        ret.number_of_ele_     = number_of_ele_;
        size_type storage_size = number_of_ele_ * value_size;
        HIP_CHECK(common::hipMallocHelper(&ret.device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpyAsync(ret.device_raw_ptr_,
                                 device_raw_ptr_,
                                 storage_size,
                                 hipMemcpyDeviceToDevice,
                                 stream));
        return ret;
    }

    /// \brief Do type cast and move the ownership to the new `device_ptr<TargetPtrType>`.
    ///
    /// \return A new `device_ptr<TargetPtrType>` rvalue.
    template<typename TargetPtrType>
    device_ptr<TargetPtrType> move_cast() noexcept
    {
        using target_value_t = typename device_ptr<TargetPtrType>::value_type;

        auto ret_deivce_raw_ptr_
            = static_cast<target_value_t*>(static_cast<void*>(device_raw_ptr_));
        auto ret_number_of_ele_ = value_size * number_of_ele_ / sizeof(target_value_t);
        leak();
        return {ret_deivce_raw_ptr_, ret_number_of_ele_};
    }

    /// \brief Get the device raw pointer
    value_type* get() const noexcept
    {
        return device_raw_ptr_;
    }

    /// \brief Clean every thing on this instance, which could lead to memory leak. Should call `get()` and free the raw pointer manually
    void leak() noexcept
    {
        device_raw_ptr_ = nullptr;
        number_of_ele_  = 0;
    }

    /// \brief Call this function to garbage the memory in advance
    void free_manually()
    {
        if(device_raw_ptr_ != nullptr)
        {
            HIP_CHECK(hipFree(device_raw_ptr_));
        }
        leak();
    }

    void resize(size_type new_number_of_ele)
    {
        if(new_number_of_ele == 0)
        {
            free_manually();
        }
        else
        {
            value_type* device_temp_ptr = nullptr;
            HIP_CHECK(common::hipMallocHelper(&device_temp_ptr, new_number_of_ele * value_size));
            HIP_CHECK(hipMemcpy(device_temp_ptr,
                                device_raw_ptr_,
                                std::min(new_number_of_ele, number_of_ele_) * value_size,
                                hipMemcpyDeviceToDevice));
            free_manually();
            device_raw_ptr_ = device_temp_ptr;
            number_of_ele_  = new_number_of_ele;
        }
    }

    void resize_async(size_type new_number_of_ele, hipStream_t stream)
    {
        if(new_number_of_ele == 0)
        {
            free_manually();
        }
        else
        {
            value_type* device_temp_ptr = nullptr;
            HIP_CHECK(common::hipMallocHelper(&device_temp_ptr, new_number_of_ele * value_size));
            HIP_CHECK(hipMemcpyAsync(device_temp_ptr,
                                     device_raw_ptr_,
                                     std::min(new_number_of_ele, number_of_ele_) * value_size,
                                     hipMemcpyDeviceToDevice,
                                     stream));
            free_manually();
            device_raw_ptr_ = device_temp_ptr;
            number_of_ele_  = new_number_of_ele;
        }
    }

    // if got error hipErrorOutOfMemory` return false, else return `true`
    bool resize_with_memory_check(size_type new_number_of_ele)
    {
        if(new_number_of_ele == 0)
        {
            free_manually();
        }
        else
        {
            value_type* device_temp_ptr = nullptr;
            const auto  err
                = common::hipMallocHelper(&device_temp_ptr, new_number_of_ele * value_size);
            if(err == hipErrorOutOfMemory)
            {
                (void) hipGetLastError(); // reset internally recorded HIP error
                return false;
            }
            HIP_CHECK(err);
            HIP_CHECK(hipMemcpy(device_temp_ptr,
                                device_raw_ptr_,
                                std::min(new_number_of_ele, number_of_ele_) * value_size,
                                hipMemcpyDeviceToDevice));
            free_manually();
            device_raw_ptr_ = device_temp_ptr;
            number_of_ele_  = new_number_of_ele;
        }
        return true;
    }

    bool resize_with_memory_check_async(size_type new_number_of_ele, hipStream_t stream)
    {
        if(new_number_of_ele == 0)
        {
            free_manually();
        }
        else
        {
            value_type* device_temp_ptr = nullptr;
            const auto  err
                = common::hipMallocHelper(&device_temp_ptr, new_number_of_ele * value_size);
            if(err == hipErrorOutOfMemory)
            {
                return false;
            }
            HIP_CHECK(err);
            HIP_CHECK(hipMemcpyAsync(device_temp_ptr,
                                     device_raw_ptr_,
                                     std::min(new_number_of_ele, number_of_ele_) * value_size,
                                     hipMemcpyDeviceToDevice,
                                     stream));
            free_manually();
            device_raw_ptr_ = device_temp_ptr;
            number_of_ele_  = new_number_of_ele;
        }
        return true;
    }

    /// \brief Get the size of this memory space
    size_type msize() const noexcept
    {
        return number_of_ele_ * value_size;
    }

    /// \brief Get the number of elements
    size_type size() const noexcept
    {
        return number_of_ele_;
    }

    /// \brief Copy from host to device
    template<typename InValueType, typename Allocator>
    void store(std::vector<InValueType, Allocator> const& host_vec, size_type offset = 0)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        if(host_vec.size() + offset > number_of_ele_)
        {
            resize(host_vec.size() + offset);
        }

        HIP_CHECK(hipMemcpy(device_raw_ptr_ + offset,
                            host_vec.data(),
                            host_vec.size() * value_size,
                            hipMemcpyHostToDevice));
    }

    template<typename InValueType, size_type Size>
    void store(std::array<InValueType, Size> const& host_arr)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        if(Size > number_of_ele_)
        {
            resize(Size);
        }

        HIP_CHECK(
            hipMemcpy(device_raw_ptr_, host_arr.data(), Size * value_size, hipMemcpyHostToDevice));
    }

    template<typename InValueType>
    void
        store(std::unique_ptr<InValueType[]> const& uptr, size_type offset, size_type number_of_ele)
    {
        static_assert(
            sizeof(InValueType) == value_size,
            "value_type of input unique_ptr must have the same size with device_ptr::value_type");

        if(offset + number_of_ele > number_of_ele_)
        {
            resize(offset + number_of_ele);
        }
        HIP_CHECK(hipMemcpy(device_raw_ptr_ + offset,
                            uptr.get(),
                            number_of_ele * value_size,
                            hipMemcpyHostToDevice));
    }

    template<typename InValueType>
    void store_async(std::vector<InValueType> const& host_vec, hipStream_t stream)
    {
        static_assert(
            sizeof(InValueType) == value_size,
            "value_type of input vector must have the same size with device_ptr::value_type");

        if(host_vec.size() > number_of_ele_)
        {
            resize(host_vec.size());
        }

        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 host_vec.data(),
                                 host_vec.size() * value_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    template<typename InValueType, size_type Size>
    void store_async(std::array<InValueType, Size> const& host_arr, hipStream_t stream)
    {
        static_assert(sizeof(InValueType) == value_size,
                      "value_type of input must have the same size with device_ptr::value_type");

        if(Size > number_of_ele_)
        {
            resize(Size);
        }

        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 host_arr.data(),
                                 Size * value_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    template<typename InValueType>
    void store_async(std::unique_ptr<InValueType[]> const& uptr,
                     size_type                             offset,
                     size_type                             number_of_ele,
                     hipStream_t                           stream)
    {
        static_assert(
            sizeof(InValueType) == value_size,
            "value_type of input unique_ptr must have the same size with device_ptr::value_type");

        if(offset + number_of_ele > number_of_ele_)
        {
            resize(offset + number_of_ele);
        }
        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_ + offset,
                                 uptr.get(),
                                 number_of_ele * value_size,
                                 hipMemcpyHostToDevice,
                                 stream));
    }

    // will not check the boundary
    void store_value_at(size_type pos, value_type_proxy const& value)
    {
        HIP_CHECK(hipMemcpy(device_raw_ptr_ + pos, &value, value_size, hipMemcpyHostToDevice));
    }

    // will not check the boundary
    template<typename T = value_type>
    void store_value_at_async(size_type pos, value_type_proxy const& value, hipStream_t stream)
    {
        HIP_CHECK(
            hipMemcpy(device_raw_ptr_ + pos, &value, value_size, hipMemcpyHostToDevice, stream));
    }

    /// \brief Copy from device to device
    template<typename InPtrValueType>
    void replace(device_ptr<InPtrValueType> const& device_ptr)
    {
        static_assert(sizeof(InPtrValueType) == value_size,
                      "sizeof(InPtrValueType) must equal to value_size");

        if(device_ptr.number_of_ele_ > number_of_ele_)
        {
            resize(device_ptr.number_of_ele_);
        }

        HIP_CHECK(hipMemcpy(device_raw_ptr_,
                            device_ptr.device_raw_ptr_,
                            device_ptr.number_of_ele_ * value_size,
                            hipMemcpyDeviceToDevice));
    }

    template<typename InPtrValueType>
    void replace_async(device_ptr<InPtrValueType> const& device_ptr, hipStream_t stream)
    {
        static_assert(sizeof(InPtrValueType) == value_size,
                      "sizeof(InPtrValueType) must equal to value_size");

        if(device_ptr.number_of_ele_ > number_of_ele_)
        {
            resize(device_ptr.number_of_ele_);
        }

        HIP_CHECK(hipMemcpyAsync(device_raw_ptr_,
                                 device_ptr.device_raw_ptr_,
                                 device_ptr.number_of_ele_ * value_size,
                                 hipMemcpyDeviceToDevice,
                                 stream));
    }

    void memset(size_type offset, int value, size_type size_bytes)
    {
        HIP_CHECK(hipMemset(reinterpret_cast<char*>(device_raw_ptr_) + offset,
                            value,
                            static_cast<size_t>(size_bytes)));
    }

    void memset_async(size_type offset, int value, size_type size_bytes, hipStream_t stream)
    {
        HIP_CHECK(hipMemsetAsync(reinterpret_cast<char*>(device_raw_ptr_) + offset,
                                 value,
                                 static_cast<size_t>(size_bytes),
                                 stream));
    }

    /// \brief Copy from device to host
    /// This function will store loaded values into std::vector
    auto load() const
    {
        std::vector<value_type> ret(number_of_ele_);
        HIP_CHECK(hipMemcpy(ret.data(),
                            device_raw_ptr_,
                            number_of_ele_ * value_size,
                            hipMemcpyDeviceToHost));
        return ret;
    }

    auto load_async(hipStream_t stream) const
    {
        std::vector<value_type> ret(number_of_ele_);
        HIP_CHECK(hipMemcpyAsync(ret.data(),
                                 device_raw_ptr_,
                                 number_of_ele_ * value_size,
                                 hipMemcpyDeviceToHost,
                                 stream));
        return ret;
    }

    template<size_type Size>
    auto load_to_array() const
    {
        std::array<value_type, Size> ret;
        HIP_CHECK(hipMemcpy(ret.data(),
                            device_raw_ptr_,
                            std::min<size_type>(number_of_ele_, Size) * value_size,
                            hipMemcpyDeviceToHost));
        return ret;
    }

    template<size_type Size>
    auto load_to_array_async(hipStream_t stream) const
    {
        std::array<value_type, Size> ret;
        HIP_CHECK(hipMemcpyAsync(ret.data(),
                                 device_raw_ptr_,
                                 std::min<size_type>(number_of_ele_, Size) * value_size,
                                 hipMemcpyDeviceToHost,
                                 stream));
        return ret;
    }

    auto load_to_unique_ptr() const
    {
        std::unique_ptr<value_type[]> ret(new value_type[number_of_ele_]);
        HIP_CHECK(hipMemcpy(ret.get(),
                            device_raw_ptr_,
                            number_of_ele_ * value_size,
                            hipMemcpyDeviceToHost));
        return ret;
    }

    auto load_to_unique_ptr_async(hipStream_t stream) const
    {
        std::unique_ptr<value_type[]> ret(new value_type[number_of_ele_]);
        HIP_CHECK(hipMemcpyAsync(ret.get(),
                                 device_raw_ptr_,
                                 number_of_ele_ * value_size,
                                 hipMemcpyDeviceToHost,
                                 stream));
        return ret;
    }

    auto load_value_at(size_type pos) const
    {
        value_type ret;
        HIP_CHECK(hipMemcpy(&ret, device_raw_ptr_ + pos, value_size, hipMemcpyDeviceToHost));
        return ret;
    }

    auto load_value_at_async(size_type pos, hipStream_t stream) const
    {
        value_type ret;
        HIP_CHECK(
            hipMemcpyAsync(&ret, device_raw_ptr_ + pos, value_size, hipMemcpyDeviceToHost, stream));
        return ret;
    }

private:
    value_type* device_raw_ptr_;
    size_type   number_of_ele_;
};

} // namespace common

#endif

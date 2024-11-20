// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TEST_UTILS_DEVICE_PTR_HPP
#define ROCPRIM_TEST_UTILS_DEVICE_PTR_HPP

#include <cstddef>
#include <vector>

#include "../common_test_header.hpp"

namespace test_utils
{

/// \brief An RAII friendly class to manage the memory allocated on device.
///
/// \tparam A Template type used by the class.
template<typename PointerType = void>
class device_ptr
{
public:
    using decay_type = std::decay_t<PointerType>;
    using size_type  = std::size_t;
    using value_type = typename std::
        conditional_t<std::is_same<decay_type, void>::value, unsigned char, PointerType>;

    device_ptr() : device_raw_ptr_(nullptr), number_of_ele_(0) {};

    /// \brief Construct with a pre-allocated memory space.
    device_ptr(size_type pre_alloc_number_of_ele)
        : device_raw_ptr_(nullptr), number_of_ele_(pre_alloc_number_of_ele)
    {
        size_type storage_size = number_of_ele_ * sizeof(value_type);
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_raw_ptr_, storage_size));
    };

    device_ptr(device_ptr const&) = delete;

    device_ptr(device_ptr&& other) noexcept
        : device_raw_ptr_(other.device_raw_ptr_), number_of_ele_(other.number_of_ele_)
    {
        other.leak();
    };

    /// \brief Construct by host vectors with the same sized value_type
    template<typename InVecValueType>
    explicit device_ptr(std::vector<InVecValueType> const& data)
        : device_raw_ptr_(nullptr), number_of_ele_(data.size())
    {
        static_assert(
            sizeof(InVecValueType) == sizeof(value_type),
            "value_type of input vector must have the same size with device_ptr::value_type");

        size_type storage_size = number_of_ele_ * sizeof(value_type);
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpy(device_raw_ptr_, data.data(), storage_size, hipMemcpyHostToDevice));
    }

    /// \brief Construct with a copy of this `host_buffer`
    ///
    /// \param _number_of_ele be aware, this is NOT the sizeof `host_buffer`, this is the `number of elements` in the `host_buffer`
    device_ptr(const void* host_buffer, size_type _number_of_ele)
        : device_raw_ptr_(nullptr), number_of_ele_(_number_of_ele)
    {
        size_type storage_size = number_of_ele_ * sizeof(value_type);
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_raw_ptr_, storage_size));
        HIP_CHECK(hipMemcpy(device_raw_ptr_, host_buffer, storage_size, hipMemcpyHostToDevice));
    };

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
    };

    /// \brief Do copy on the device.
    ///
    /// \return A new `device_ptr` rvalue.
    device_ptr duplicate() const
    {
        device_ptr ret;
        ret.number_of_ele_     = number_of_ele_;
        size_type storage_size = number_of_ele_ * sizeof(value_type);
        HIP_CHECK(test_common_utils::hipMallocHelper(&ret.device_raw_ptr_, storage_size));
        HIP_CHECK(
            hipMemcpy(ret.device_raw_ptr_, device_raw_ptr_, storage_size, hipMemcpyDeviceToDevice));
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
        auto ret_number_of_ele_ = sizeof(value_type) * number_of_ele_ / sizeof(target_value_t);
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
        if(device_raw_ptr_)
        {
            HIP_CHECK(hipFree(device_raw_ptr_));
        }
        leak();
    }

    void resize(size_type _new_number_of_ele)
    {
        if(_new_number_of_ele == 0)
        {
            free_manually();
        }
        else
        {
            value_type* device_temp_ptr = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&device_temp_ptr,
                                                         _new_number_of_ele * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(device_temp_ptr,
                                device_raw_ptr_,
                                std::min(_new_number_of_ele, number_of_ele_) * sizeof(value_type),
                                hipMemcpyDeviceToDevice));
            free_manually();
            device_raw_ptr_ = device_temp_ptr;
            number_of_ele_  = _new_number_of_ele;
        }
    }

    /// \brief Get the size of this memory space
    size_type msize() const noexcept
    {
        return number_of_ele_ * sizeof(value_type);
    }

    /// \brief Get the number of elements
    size_type size() const noexcept
    {
        return number_of_ele_;
    }

    /// \brief Copy from host to device
    template<typename InVecValueType>
    void store(std::vector<InVecValueType> const& host_vec, size_type offset = 0)
    {
        static_assert(
            sizeof(InVecValueType) == sizeof(value_type),
            "value_type of input vector must have the same size with device_ptr::value_type");

        if(host_vec.size() + offset > number_of_ele_)
        {
            resize(host_vec.size() + offset);
        }

        HIP_CHECK(hipMemcpy(device_raw_ptr_ + offset,
                            host_vec.data(),
                            host_vec.size() * sizeof(value_type),
                            hipMemcpyHostToDevice));
    }

    /// \brief Copy from host to device
    template<typename InPtrValueType>
    void store(device_ptr<InPtrValueType> const& device_ptr, size_type offset = 0)
    {
        static_assert(sizeof(InPtrValueType) == sizeof(value_type),
                      "sizeof(InPtrValueType) must equal to sizeof(value_type)");

        if(device_ptr.number_of_ele_ + offset > number_of_ele_)
        {
            resize(device_ptr.number_of_ele_ + offset);
        }

        HIP_CHECK(hipMemcpy(device_raw_ptr_ + offset,
                            device_ptr.device_raw_ptr_,
                            device_ptr.number_of_ele_ * sizeof(value_type),
                            hipMemcpyDeviceToDevice));
    }

    /// \brief Copy from device to host
    std::vector<value_type> load() const
    {
        std::vector<value_type> ret(number_of_ele_);
        HIP_CHECK(hipMemcpy(ret.data(),
                            device_raw_ptr_,
                            number_of_ele_ * sizeof(value_type),
                            hipMemcpyDeviceToHost));
        return ret;
    }

private:
    value_type* device_raw_ptr_;
    size_type   number_of_ele_;
};

} // namespace test_utils

#endif

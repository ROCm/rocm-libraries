// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — RAII device memory wrapper. Requires HIP runtime.
//
// RAII device buffer with automatic float-to-typed conversion for upload
// and typed-to-float conversion for download. Simplifies host code that
// needs to work with multiple data types (fp32, fp16, bf16).

#pragma once

#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <vector>

namespace rocm_ck {

/// RAII device buffer that stores elements in a specified DataType.
/// Upload converts from float; download converts back to float.
/// Non-copyable.
class TypedBuffer
{
    public:
    TypedBuffer(DataType dtype, int count)
        : dtype_(dtype), count_(count), elem_bytes_(data_type_bits(dtype) / 8)
    {
        HIP_CHECK(hipMalloc(&device_ptr_, static_cast<size_t>(count_) * elem_bytes_));
    }

    ~TypedBuffer()
    {
        // Destructor can't propagate errors — best-effort cleanup
        if(device_ptr_ != nullptr)
            (void)hipFree(device_ptr_);
    }

    TypedBuffer(const TypedBuffer&)            = delete;
    TypedBuffer& operator=(const TypedBuffer&) = delete;

    TypedBuffer(TypedBuffer&& other) noexcept
        : dtype_(other.dtype_),
          count_(other.count_),
          elem_bytes_(other.elem_bytes_),
          device_ptr_(other.device_ptr_)
    {
        other.device_ptr_ = nullptr;
    }

    TypedBuffer& operator=(TypedBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(device_ptr_ != nullptr)
                (void)hipFree(device_ptr_);
            dtype_            = other.dtype_;
            count_            = other.count_;
            elem_bytes_       = other.elem_bytes_;
            device_ptr_       = other.device_ptr_;
            other.device_ptr_ = nullptr;
        }
        return *this;
    }

    /// Convert float array to typed format and upload to device.
    void upload(const float* src)
    {
        size_t buf_size = static_cast<size_t>(count_) * elem_bytes_;
        std::vector<char> host_buf(buf_size);
        for(int i = 0; i < count_; ++i)
            float_to_typed(dtype_, src[i], host_buf.data() + i * elem_bytes_);
        HIP_CHECK(hipMemcpy(device_ptr_, host_buf.data(), buf_size, hipMemcpyHostToDevice));
    }

    /// Download from device and convert typed format back to float array.
    void download(float* dst) const
    {
        size_t buf_size = static_cast<size_t>(count_) * elem_bytes_;
        std::vector<char> host_buf(buf_size);
        HIP_CHECK(hipMemcpy(host_buf.data(), device_ptr_, buf_size, hipMemcpyDeviceToHost));
        for(int i = 0; i < count_; ++i)
            dst[i] = typed_to_float(dtype_, host_buf.data() + i * elem_bytes_);
    }

    /// Zero the device buffer.
    void zero() { HIP_CHECK(hipMemset(device_ptr_, 0, static_cast<size_t>(count_) * elem_bytes_)); }

    void* ptr() { return device_ptr_; }
    const void* ptr() const { return device_ptr_; }
    DataType dtype() const { return dtype_; }
    int count() const { return count_; }

    private:
    DataType dtype_;
    int count_;
    int elem_bytes_;
    void* device_ptr_ = nullptr;
};

} // namespace rocm_ck

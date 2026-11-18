// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <vector>

namespace hipblaslt_py
{

    // Bytes per element of a type DeviceArray holds, or 0 for one it doesn't.
    size_t host_itemsize(hipDataType dtype);

    // A block of device memory holding one hipDataType. Every copy checks the host buffer's
    // byte count against it; the dtype checks happen where the host array arrives.
    class DeviceArray
    {
    public:
        DeviceArray(size_t nbytes, hipDataType dtype, std::vector<int64_t> shape);
        ~DeviceArray();
        DeviceArray(const DeviceArray&)            = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;

        void free();
        void copy_from_host(const void* src, size_t nbytes);
        void copy_to_host(void* dst, size_t nbytes) const;

        void*  raw() const;
        size_t nbytes() const
        {
            return nbytes_;
        }
        hipDataType dtype() const
        {
            return dtype_;
        }
        const std::vector<int64_t>& shape() const
        {
            return shape_;
        }
        bool freed() const
        {
            return d_ == nullptr;
        }

    private:
        void free_nothrow() noexcept;

        void*                d_      = nullptr;
        size_t               nbytes_ = 0;
        hipDataType          dtype_;
        std::vector<int64_t> shape_;
    };

} // namespace hipblaslt_py

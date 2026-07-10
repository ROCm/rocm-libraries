// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <string>
#include <vector>

namespace hipblaslt_py {

class DeviceArray
{
public:
    DeviceArray(size_t nbytes, hipDataType dtype,
                std::vector<int64_t> shape, std::string host_dtype);
    ~DeviceArray();
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void  free();
    void  copy_from_host(const void* src, size_t nbytes);
    void  copy_to_host(void* dst, size_t nbytes) const;

    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }
    void*          raw() const { return d_; }
    size_t         nbytes() const { return nbytes_; }
    hipDataType    dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::string& host_dtype() const { return host_dtype_; }

private:
    void*                d_ = nullptr;
    size_t               nbytes_ = 0;
    hipDataType          dtype_;
    std::vector<int64_t> shape_;
    std::string          host_dtype_;   // numpy dtype string, e.g. "float32"
};

} // namespace hipblaslt_py

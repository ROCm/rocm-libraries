// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "global_test_memory.hpp"

#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "utils.hpp"
#include "device_prng.hpp"

#include <cstddef>
#include <cstring>

#include <hip/hip_runtime.h>

namespace test::gtest::global_buffer {

struct GlobalDeviceMemoryImpl
{

    GlobalDeviceMemoryImpl() { hipInit(0); }

    ~GlobalDeviceMemoryImpl()
    {
        if(buffer_ != nullptr)
        {
            CHECK_HIP_ERROR(hipFree(buffer_), "Failed to free device buffer");
        }

        if(host_mirror_ != nullptr)
        {
            CHECK_HIP_ERROR(hipHostFree(host_mirror_), "Failed to free host mirror buffer");
        }
    }

    void EnsureBufferSize(const size_t requested_total_size)
    {
        if(requested_total_size > total_size_)
        {
            AllocateNewBuffer(requested_total_size);
        }
    }

    void FreeBuffer()
    {
        if(buffer_ != nullptr)
        {
            CHECK_HIP_ERROR(hipFree(buffer_), "Failed to free device buffer");
        }
        if(host_mirror_ != nullptr)
        {
            CHECK_HIP_ERROR(hipHostFree(host_mirror_), "Failed to free host mirror buffer");
        }
    }

    void AllocateNewBuffer(const size_t new_size)
    {
        FreeBuffer();
        total_size_ = new_size;

        CHECK_HIP_ERROR(hipMalloc(&buffer_, total_size_), "Failed to allocate device buffer");
        CHECK_HIP_ERROR(hipHostMalloc(&host_mirror_, total_size_, hipHostMallocDefault),
                        "Failed to allocate host mirror buffer");
    }

    void ReleaseBuffer() { cur_offset_ = 0; }

    DeviceBufferObject GetBufferChunk(const size_t chunk_size)
    {
        const auto start_point = cur_offset_;
        cur_offset_ += chunk_size;

        if(cur_offset_ > total_size_)
        {
            MIOPEN_THROW_HIP_STATUS(hipErrorOutOfMemory,
                                    "Global device buffer chunk allocation overflow");
        }

        return DeviceBufferObject{static_cast<char*>(buffer_) + start_point,
                                  chunk_size,
                                  static_cast<char*>(host_mirror_) + start_point,
                                  get_handle().GetStream()};
    }

    void* buffer_      = nullptr;
    void* host_mirror_ = nullptr;
    size_t cur_offset_ = 0;
    size_t total_size_ = 0;

    std::unordered_map<void*, hipEvent_t> pending_events_;
};

inline GlobalDeviceMemoryImpl& GetInstance()
{
    static GlobalDeviceMemoryImpl impl;
    return impl;
}

void EnsureBufferSize(size_t size_in_bytes) { GetInstance().EnsureBufferSize(size_in_bytes); }
void ReleaseBuffer() { GetInstance().ReleaseBuffer(); }
DeviceBufferObject GetDeviceBuffer(size_t size) { return GetInstance().GetBufferChunk(size); }
void FreeBuffer() { GetInstance().FreeBuffer(); }

template <typename T>
DeviceBufferObject DeviceBufferObject::fill(T val) const
{
    ::test::gtest::gpu_fill<T>(static_cast<T*>(ptr_), val, size_bytes_, stream_);

    return *this;
}

template <typename T>
DeviceBufferObject DeviceBufferObject::randomize(unsigned long long seed) const
{
    size_t num_elements = size_bytes_ / sizeof(T);

    RandomizeBuffer(static_cast<T*>(ptr_), num_elements, seed, stream_);

    return *this;
}

void DeviceBufferObject::HostMirror() const
{
    if(size_bytes_ == 0)
        return;

    CHECK_HIP_ERROR(hipMemcpy(host_mirror_, ptr_, size_bytes_, hipMemcpyDeviceToHost),
                    "Failed to copy device buffer to host mirror");
}

void DeviceBufferObject::HostMirrorAsync() const
{
    if(size_bytes_ == 0)
        return;

    if(ptr_ == nullptr || host_mirror_ == nullptr)
    {
        MIOPEN_THROW_HIP_STATUS(hipErrorInvalidValue,
                                "Invalid buffer or host mirror pointer in HostMirror");
    }

    hipEvent_t e;
    hipEventCreate(&e);

    CHECK_HIP_ERROR(hipMemcpyAsync(host_mirror_, ptr_, size_bytes_, hipMemcpyDeviceToHost, stream_),
                    "Failed to async copy device buffer to host mirror");

    hipEventRecord(e, stream_);
    GetInstance().pending_events_[host_mirror_] = e;
}

bool DeviceBufferObject::HostMirrorReady() const
{
    auto it = GetInstance().pending_events_.find(host_mirror_);
    if(it == GetInstance().pending_events_.end())
    {
        return true;
    }

    hipError_t status = hipEventQuery(it->second);
    if(status == hipSuccess)
    {
        hipEventDestroy(it->second);
        GetInstance().pending_events_.erase(it);
        return true;
    }

    return false;
}

void DeviceBufferObject::HostMirrorWait() const
{
    auto it = GetInstance().pending_events_.find(host_mirror_);
    if(it == GetInstance().pending_events_.end())
    {
        return;
    }

    CHECK_HIP_ERROR(hipEventSynchronize(it->second), "Failed to synchronize host mirror event");

    hipEventDestroy(it->second);
    GetInstance().pending_events_.erase(it);
}

// -----------------------------------------------------------------------------
// EXPLICIT INSTANTIATION
// -----------------------------------------------------------------------------

template DeviceBufferObject DeviceBufferObject::fill<double>(double) const;
template DeviceBufferObject DeviceBufferObject::randomize<double>(unsigned long long) const;

template DeviceBufferObject DeviceBufferObject::fill<float>(float) const;
template DeviceBufferObject DeviceBufferObject::randomize<float>(unsigned long long) const;

template DeviceBufferObject DeviceBufferObject::fill<int8_t>(int8_t) const;
template DeviceBufferObject DeviceBufferObject::randomize<int8_t>(unsigned long long) const;

template DeviceBufferObject DeviceBufferObject::fill<int32_t>(int32_t) const;
template DeviceBufferObject DeviceBufferObject::randomize<int32_t>(unsigned long long) const;

template DeviceBufferObject DeviceBufferObject::fill<bfloat16>(bfloat16) const;
template DeviceBufferObject DeviceBufferObject::randomize<bfloat16>(unsigned long long) const;

template DeviceBufferObject DeviceBufferObject::fill<half_float::half>(half_float::half) const;
template DeviceBufferObject
DeviceBufferObject::randomize<half_float::half>(unsigned long long) const;

} // namespace test::gtest::global_buffer

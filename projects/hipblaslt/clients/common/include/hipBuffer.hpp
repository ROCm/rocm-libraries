/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "d_vector.hpp"
#include "datatype_interface.hpp"
#include "hipblaslt_ostream.hpp"
#include <algorithm>
#include <memory>
#include <roc/host_validation/tensor.hpp>
#include <span>

class HipDeviceBuffer : public d_vector_type
{
public:
    HipDeviceBuffer(hipDataType dtype, std::size_t numElements, bool HMM = false)
        : d_vector_type(dtype, numElements, HMM)
        , numBytes(realDataTypeSize(dtype) * numElements)
        , buffer(this->device_vector_setup())
    {
    }

    ~HipDeviceBuffer()
    {
        this->device_vector_teardown(static_cast<char*>(buffer));
        buffer = nullptr;
    }

    HipDeviceBuffer(const HipDeviceBuffer&)            = delete;
    HipDeviceBuffer(HipDeviceBuffer&&)                 = default;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;
    HipDeviceBuffer& operator=(HipDeviceBuffer&&)      = default;

    void* buf()
    {
        return buffer;
    }

    const void* buf() const
    {
        return buffer;
    }

    std::size_t getNumBytes() const
    {
        return numBytes;
    }

    template <typename T>
    T* as()
    {
        return reinterpret_cast<T*>(buf());
    }

    template <typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(buf());
    }

    hipError_t memcheck() const
    {
        return !this->nmemb() || buf() ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    std::size_t numBytes;
    void*       buffer;
};

class HipHostBuffer
{
    struct PooledHostMemory
    {
        explicit PooledHostMemory(h_memory memory_)
            : memory(std::move(memory_))
        {
        }

        ~PooledHostMemory()
        {
            memory_pool<h_memory>::Restore(memory);
        }

        h_memory memory;
    };

public:
    HipHostBuffer(hipDataType dtype, std::size_t numElements)
        : buffer(std::make_shared<PooledHostMemory>(memory_pool<h_memory>::Get(
              realDataTypeSize(dtype) * numElements ? realDataTypeSize(dtype) * numElements
                                                    : realDataTypeSize(dtype))))
    {
    }

    ~HipHostBuffer()                               = default;
    HipHostBuffer(const HipHostBuffer&)            = delete;
    HipHostBuffer(HipHostBuffer&&)                 = default;
    HipHostBuffer& operator=(const HipHostBuffer&) = delete;
    HipHostBuffer& operator=(HipHostBuffer&&)      = default;

    void* end()
    {
        return (void*)((char*)buffer->memory.get() + getNumBytes());
    }

    const void* end() const
    {
        return (void*)((const char*)buffer->memory.get() + getNumBytes());
    }

    void* buf()
    {
        return buffer->memory.get();
    }

    const void* buf() const
    {
        return buffer->memory.get();
    }

    std::size_t getNumBytes() const
    {
        return buffer->memory.bytes();
    }

    template <typename T>
    T* as()
    {
        return reinterpret_cast<T*>(buf());
    }

    template <typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(buf());
    }

    roc::host_validation::TensorStorage tensorStorage() const
    {
        return roc::host_validation::TensorStorage::wrap(
            buffer,
            std::span<std::byte>(reinterpret_cast<std::byte*>(buffer->memory.get()),
                                 getNumBytes()));
    }

    roc::host_validation::Tensor tensor(roc::host_validation::ScalarType type,
                                        roc::host_validation::Layout     layout) const
    {
        return roc::host_validation::Tensor::wrapStorage(type, std::move(layout), tensorStorage());
    }

    static roc::host_validation::TensorStorage allocateTensorStorage(size_t bytes)
    {
        const size_t allocationBytes = std::max<size_t>(bytes, 1);
        auto         owner
            = std::make_shared<PooledHostMemory>(memory_pool<h_memory>::Get(allocationBytes));
        return roc::host_validation::TensorStorage::wrap(
            owner, std::span<std::byte>(reinterpret_cast<std::byte*>(owner->memory.get()), bytes));
    }

    static roc::host_validation::TensorStorageAllocator tensorAllocator()
    {
        return [](size_t bytes) { return allocateTensorStorage(bytes); };
    }

private:
    std::shared_ptr<PooledHostMemory> buffer;
};

inline hipError_t synchronize(HipDeviceBuffer&     dBuf,
                              const HipHostBuffer& hBuf,
                              std::size_t          block_count = 1,
                              hipStream_t          stream      = nullptr)
{
    hipError_t hip_err;

    // Perform async copy for all blocks
    for(size_t i_block = 0; i_block < block_count; i_block++)
    {
        hip_err = hipMemcpyAsync(dBuf.as<char>() + i_block * dBuf.getNumBytes() / block_count,
                                 hBuf.as<char>(),
                                 dBuf.getNumBytes() / block_count,
                                 dBuf.use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice,
                                 stream);

        if(hip_err != hipSuccess)
        {
            return hip_err;
        }
    }

    return hipStreamSynchronize(stream);
}

inline hipError_t broadcast(HipDeviceBuffer& dBuf, std::size_t repeats)
{
    hipError_t hip_err = hipSuccess;
    for(size_t i = 1; i < repeats; ++i)
    {
        hip_err = hipMemcpy(dBuf.as<char>() + i * dBuf.getNumBytes() / repeats,
                            dBuf.as<char>(),
                            dBuf.getNumBytes() / repeats,
                            dBuf.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToDevice);

        if(hip_err != hipSuccess)
        {
            return hip_err;
        }
    }
    return hip_err;
}

inline hipError_t synchronize(HipHostBuffer&         hBuf,
                              const HipDeviceBuffer& dBuf,
                              size_t                 batch       = 0,
                              size_t                 row         = 0,
                              size_t                 col         = 0,
                              size_t                 lda         = 0,
                              size_t                 elementSize = 1,
                              bool                   needSwizzle = false,
                              hipStream_t            stream      = nullptr)
{
    // lda is only used by the swizzled row-by-row copy below; the plain copy path
    // ignores it, so only the swizzle path can have an out-of-range leading dimension.
    if(needSwizzle && row > lda)
        hipblaslt_cerr << "invalid values of lda in synchronize()" << std::endl;
    hipError_t hip_err;

    // Synchronize to ensure prior work is complete
    hip_err = hipStreamSynchronize(stream);
    if(hip_err != hipSuccess)
        return hip_err;

    if(!needSwizzle)
    {
        hip_err = hipMemcpyAsync(
            hBuf.as<char>(), dBuf.as<char>(), hBuf.getNumBytes(), hipMemcpyDeviceToHost, stream);
        if(hip_err != hipSuccess)
            return hip_err;
        return hipStreamSynchronize(stream);
    }

    for(size_t j = 0; j < batch * col; j++)
    {
        hip_err = hipMemcpyAsync(hBuf.as<char>() + (j * lda * elementSize),
                                 dBuf.as<char>() + (j * row * elementSize),
                                 row * elementSize,
                                 hipMemcpyDeviceToHost,
                                 stream);

        if(hip_err != hipSuccess)
            return hip_err;
    }

    return hipStreamSynchronize(stream);
}

/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GUARD_RPP_CONTEXT_HPP_
#define GUARD_RPP_CONTEXT_HPP_

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <unordered_map>

#include "rpp.h"
#include "rppdefs.h"
#include "common.hpp"
#include "object.hpp"
#include "allocator.hpp"

#if RPP_USE_ROCBLAS
#include "manage_ptr.hpp"
#include <rocblas.h>
#endif

namespace rpp {

struct HandleImpl;
#if RPP_USE_RPPGEMM
struct GemmGeometry;
using GemmKey = std::pair<std::string, std::string>;
#endif

#if RPP_USE_ROCBLAS
using rocblas_handle_ptr = RPP_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);
#endif

#if !GPU_SUPPORT

struct Handle : rppHandle
{
    Handle(size_t nBatchSize, Rpp32u numThreads = 0);
    ~Handle();

    InitHandle* GetInitHandle() const;
    size_t GetBatchSize() const;
    Rpp32u GetNumThreads() const;
    RppBackend GetBackend() const;
    void SetBatchSize(size_t bSize) const;
    void rpp_destroy_object_host();
    std::unique_ptr<HandleImpl> impl;
};

#else

struct Handle : rppHandle
{
    // Host handle related
    Handle(size_t nBatchSize, Rpp32u numThreads = 0);
    ~Handle();
    InitHandle*  GetInitHandle() const;
    size_t GetBatchSize() const;
    Rpp32u GetNumThreads() const;
    RppBackend GetBackend() const;
    void SetBatchSize(size_t bSize) const;
    void rpp_destroy_object_host();

    // Allocator related
    void SetAllocator(rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext) const;

    // Device handle related
    Handle(size_t nBatchSize, rppAcceleratorQueue_t stream);
    void rpp_destroy_object_gpu();
    rppAcceleratorQueue_t GetStream() const;
    void SetStream(rppAcceleratorQueue_t streamID) const;

    // Memory related
    std::size_t GetLocalMemorySize();
    std::size_t GetGlobalMemorySize();
    std::size_t GetMaxComputeUnits();
    std::size_t m_MaxMemoryAllocSizeCached = 0;
    std::size_t GetMaxMemoryAllocSize();

    // Other
    std::string GetDeviceName();
    std::unique_ptr<HandleImpl> impl;
};

#endif // GPU_SUPPORT

} // namespace rpp

RPP_DEFINE_OBJECT(rppHandle, rpp::Handle);

#endif // GUARD_RPP_CONTEXT_HPP_

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
#include "kernel.hpp"
#include "object.hpp"
#include "simple_hash.hpp"
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
    void SetBatchSize(size_t bSize) const;
    void rpp_destroy_object_host();

    // Allocator related
    void SetAllocator(rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext) const;

    // Device handle related
    Handle(size_t nBatchSize, rppAcceleratorQueue_t stream);
    void rpp_destroy_object_gpu();
    rppAcceleratorQueue_t GetStream() const;
    void SetStream(rppAcceleratorQueue_t streamID) const;

    // Profiling and timing related
    void EnableProfiling(bool enable = true);
    void ResetKernelTime();
    void AccumKernelTime(float curr_time);
    float GetKernelTime() const;
    bool IsProfilingEnabled() const;

    // Kernel related
    KernelInvoke AddKernel(const std::string& algorithm,
                           const std::string& network_config,
                           const std::string& program_name,
                           const std::string& kernel_name,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           const std::string& params,
                           std::size_t cache_index       = 0,
                           bool is_kernel_str            = false,
                           const std::string& kernel_src = "");

    bool HasKernel(const std::string& algorithm, const std::string& network_config) const;
    void ClearKernels(const std::string& algorithm, const std::string& network_config);
    auto GetKernels(const std::string& algorithm, const std::string& network_config);
    KernelInvoke GetKernel(const std::string& algorithm, const std::string& network_config);
    KernelInvoke Run(Kernel k);
    const std::vector<Kernel>& GetKernelsImpl(const std::string& algorithm, const std::string& network_config);
    Program LoadProgram(const std::string& program_name, std::string params, bool is_kernel_str, const std::string& kernel_src);
    void Finish() const;
    void Flush() const;

    // Memory related
    std::size_t GetLocalMemorySize();
    std::size_t GetGlobalMemorySize();
    std::size_t GetMaxComputeUnits();
    std::size_t m_MaxMemoryAllocSizeCached = 0;
    std::size_t GetMaxMemoryAllocSize();

    // Other
    std::string GetDeviceName();
    std::ostream& Print(std::ostream& os) const;
    void Copy(ConstData_t src, Data_t dest, std::size_t size);
    Allocator::ManageDataPtr Create(std::size_t sz);
    Allocator::ManageDataPtr& WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz);
    void ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz);
#if HIP_COMPILE
    shared<ConstData_t> CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t size);
#elif OCL_COMPILE
    shared<Data_t> CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size);
#endif

    template <class T>
    Allocator::ManageDataPtr Create(std::size_t sz)
    {
        return this->Create(sz * sizeof(T));
    }

    template <class Container>
    Allocator::ManageDataPtr Write(const Container& c)
    {
        using type = typename Container::value_type;
        auto buf   = this->Create<type>(c.size());
        return std::move(
            this->WriteTo(reinterpret_cast<const void*>(c.data()), buf, c.size() * sizeof(type)));
    }

    template <class T>
    std::vector<T> Read(const Allocator::ManageDataPtr& ddata, std::size_t sz)
    {
        std::vector<T> result(sz);
        this->ReadTo(result.data(), ddata, sz * sizeof(T));
        return result;
    }

    std::string GetDbBasename()
    {
        return GetDeviceName() + "_" + std::to_string(GetMaxComputeUnits());
    }

    std::unique_ptr<HandleImpl> impl;
};

inline std::ostream& operator<<(std::ostream& os, const Handle& handle) { return handle.Print(os); }

struct AutoEnableProfiling
{
    AutoEnableProfiling(Handle& x) : h(x)
    {
        prev_state = h.IsProfilingEnabled();
        h.EnableProfiling();
    }

    ~AutoEnableProfiling()
    {
        h.EnableProfiling(prev_state);
        h.ResetKernelTime();
    }

    private:
    Handle& h;
    bool prev_state;
};

#endif // GPU_SUPPORT

} // namespace rpp

RPP_DEFINE_OBJECT(rppHandle, rpp::Handle);

#endif // GUARD_RPP_CONTEXT_HPP_

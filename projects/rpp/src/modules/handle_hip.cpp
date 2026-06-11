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

#ifndef _WIN32
#include <unistd.h>
#endif

#include <thread>
#include "device_name.hpp"
#include "errors.hpp"
#include "handle.hpp"

namespace rpp {

// Get current context
// We leak resources for now as there is no hipCtxRetain API
hipCtx_t get_ctx()
{
    auto status = hipInit(0);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "hipInit failed");
    hipCtx_t ctx{};
    return ctx;
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz)
{
    if(sz > GetAvailableMemory())
        RPP_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        status = hipHostMalloc(&result, sz);
        if(status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "Hip error creating buffer " + std::to_string(sz) + ": ");
    }
    return result;
}

void default_deallocator(void*, void* mem)
{
    auto status = hipFree(mem);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "hipFree failed");
}

int get_device_id() // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        RPP_THROW("No device");
    return device;
}

void set_ctx(hipCtx_t ctx)
{
    auto status = 0;
    if(status != hipSuccess)
        RPP_THROW("Error setting context");
}

struct HandleImpl
{
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    hipCtx_t ctx;
    StreamPtr stream = nullptr;
    int device = -1;
    Allocator allocator{};
    bool enable_profiling = false;
    float profiling_result = 0.0;
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    InitHandle* initHandle = nullptr;

    HandleImpl() : ctx(get_ctx()) {}

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        if(enable_profiling)
        {
            auto status = hipEventElapsedTime(&this->profiling_result, start, stop);
            if(status != hipSuccess)
                RPP_THROW_HIP_STATUS(status, "hipEventElapsedTime failed");
        }
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(
            &HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    void set_ctx()
    {
        rpp::set_ctx(this->ctx);
        // rpp::set_device(this->device);
        // Check device matches
        if(this->device != get_device_id())
            RPP_THROW("Running handle on wrong device");
    }

    void PreInitializeBufferCPU()
    {
        this->initHandle = new InitHandle();
        this->initHandle->mem.mcpu.scratchBufferHost = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }

    void PreInitializeBuffer()
    {
        this->initHandle = new InitHandle();
        this->PreInitializeBufferCPU();

#ifdef AUDIO_SUPPORT
        // If AUDIO_SUPPORT is enabled, 'scratchBufferHip' needed to run RNNT training successfully are larger.
        // Current max allocation size = sizeof(Rpp32f) * 372877312, which is based on Spectrogram requirements
        // 1. Spectrogram requirements:
        //      - 372877312 = (512 * 3754 * 192) + (512 * 3754 * 2)
        //      - Above is the maximum scratch memory required for Spectrogram HIP kernel used in RNNT training (uses a batchsize 192)
        //      - (512 * 3754 * 192) is the maximum size that will be required for window output based on Librispeech dataset in RNNT training
        //      - (512 * 3754 * 2) is the size required for storing sin and cos coefficients required for FFT computation in Spectrogram HIP kernel in RNNT training
        // 2. Non Silent Region Detection requirements:
        //      - 115293120 = (600000 + 293 + 192) * 192
        //      - Above is the maximum scratch memory required for Non Silent Region Detection HIP kernel used in RNNT training (uses a batchsize 192)
        //      - 600000 is the maximum size that will be required for MMS buffer based on Librispeech dataset
        //      - 293 is the size required for storing reduction outputs for 600000 size sample
        //      - 192 is the size required for storing cutOffDB values for batch size 192
        auto status = hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 372877312);
#else
        auto status = hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 8294400);   // 3840 x 2160
#endif
        if(status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipMalloc failed for scratchBufferHip");
        status = hipHostMalloc(&(this->initHandle->mem.mgpu.scratchBufferPinned.floatmem), sizeof(Rpp32f) * 8294400);    // 3840 x 2160
        if(status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipHostMalloc failed for scratchBufferPinned");
    }
};

Handle::Handle(size_t batchSize, rppAcceleratorQueue_t stream) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    impl->backend = RppBackend::RPP_HIP_BACKEND;
    this->impl->device = get_device_id();
    this->impl->ctx = get_ctx();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBuffer();
}

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    impl->backend = RppBackend::RPP_HOST_BACKEND;
    numThreads = std::min(numThreads, std::thread::hardware_concurrency());
    if(numThreads == 0)
        numThreads = batchSize;
    impl->numThreads = numThreads;
    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBufferCPU();
}

Handle::~Handle() {}

void Handle::SetStream(rppAcceleratorQueue_t streamID) const
{
    this->impl->stream = HandleImpl::reference_stream(streamID);
}

void Handle::rpp_destroy_object_gpu()
{
    this->rpp_destroy_object_host();

    auto status = hipFree(this->GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "hipFree failed for scratchBufferHip");
    status = hipHostFree(this->GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "hipHostFree failed for scratchBufferPinned");

    delete this->GetInitHandle();
    this->impl = nullptr;
}

void Handle::rpp_destroy_object_host()
{
    free(this->GetInitHandle()->mem.mcpu.scratchBufferHost);
}

size_t Handle::GetBatchSize() const
{
    return this->impl->nBatchSize;
}

Rpp32u Handle::GetNumThreads() const
{
    return this->impl->numThreads;
}

RppBackend Handle::GetBackend() const
{
    return this->impl->backend;
}

void Handle::SetBatchSize(size_t bSize) const
{
    this->impl->nBatchSize = bSize;
}

rppAcceleratorQueue_t Handle::GetStream() const
{
    return impl->stream.get();
}

InitHandle* Handle::GetInitHandle() const
{
    return impl->initHandle;
}

void Handle::SetAllocator(rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext) const
{
    this->impl->allocator.allocator = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;
    this->impl->allocator.context = allocatorContext;
}

std::size_t Handle::GetLocalMemorySize()
{
    int result;
    auto status = hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetGlobalMemorySize()
{
    size_t result;
    auto status = hipDeviceTotalMem(&result, this->impl->device);

    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status);

    return result;
}

std::string Handle::GetDeviceName()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, this->impl->device);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "hipGetDeviceProperties failed");
    std::string name(props.gcnArchName);
    return name;
}

// No HIP API that could return maximum memory allocation size for a single object.
std::size_t Handle::GetMaxMemoryAllocSize()
{
    if(m_MaxMemoryAllocSizeCached == 0)
    {
        size_t free, total;
        auto status = hipMemGetInfo(&free, &total);
        if(status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "Failed getting available memory");
        m_MaxMemoryAllocSizeCached = floor(total * 0.85);
    }

    return m_MaxMemoryAllocSizeCached;
}

std::size_t Handle::GetMaxComputeUnits()
{
    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status);

    return result;
}

} // namespace rpp

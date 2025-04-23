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
#include "logger.hpp"
#include "handle.hpp"
#include "kernel_cache.hpp"
#include "binary_cache.hpp"

namespace rpp {

// Get current context
// We leak resources for now as there is no hipCtxRetain API
hipCtx_t get_ctx()
{
    hipInit(0);
    hipCtx_t ctx;
    auto status = 0;
    if(status != hipSuccess)
        RPP_THROW("No device");
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
    CHECK_RETURN_STATUS(hipFree(mem));
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
    auto status =  0;
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
    KernelCache cache;
    bool enable_profiling = false;
    float profiling_result = 0.0;
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    InitHandle* initHandle = nullptr;

    HandleImpl() : ctx(get_ctx()) {}

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        if(enable_profiling)
            hipEventElapsedTime(&this->profiling_result, start, stop);
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

        this->initHandle->nbatchSize = this->nBatchSize;
        this->initHandle->mem.mcpu.srcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxSrcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxDstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.roiPoints = (RppiROI *)malloc(sizeof(RppiROI) * this->nBatchSize);
        this->initHandle->mem.mcpu.srcBatchIndex = (Rpp64u *)malloc(sizeof(Rpp64u) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstBatchIndex = (Rpp64u *)malloc(sizeof(Rpp64u) * this->nBatchSize);
        this->initHandle->mem.mcpu.inc = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstInc = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);

        for(int i = 0; i < 10; i++)
        {
            this->initHandle->mem.mcpu.floatArr[i].floatmem = (Rpp32f *)malloc(sizeof(Rpp32f) * this->nBatchSize);
            this->initHandle->mem.mcpu.uintArr[i].uintmem = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
            this->initHandle->mem.mcpu.intArr[i].intmem = (Rpp32s *)malloc(sizeof(Rpp32s) * this->nBatchSize);
            this->initHandle->mem.mcpu.ucharArr[i].ucharmem = (Rpp8u *)malloc(sizeof(Rpp8u) * this->nBatchSize);
            this->initHandle->mem.mcpu.charArr[i].charmem = (Rpp8s *)malloc(sizeof(Rpp8s) * this->nBatchSize);
        }

        this->initHandle->mem.mcpu.rgbArr.rgbmem = (RpptRGB *)malloc(sizeof(RpptRGB) * this->nBatchSize);
        this->initHandle->mem.mcpu.scratchBufferHost = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }

    void PreInitializeBuffer()
    {
        this->initHandle = new InitHandle();
        this->PreInitializeBufferCPU();

        this->initHandle->mem.mgpu.csrcSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.csrcSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cdstSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cdstSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxSrcSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxSrcSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxDstSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxDstSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.x = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.y = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.roiHeight = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.roiWidth = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.srcSize.height), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.srcSize.width), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.dstSize.height), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.dstSize.width), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.maxSrcSize.height), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.maxSrcSize.width), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.maxDstSize.height), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.maxDstSize.width), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.roiPoints.x), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.roiPoints.y), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.roiPoints.roiHeight), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.roiPoints.roiWidth), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.inc), sizeof(Rpp32u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.dstInc), sizeof(Rpp32u) * this->nBatchSize));

        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.srcBatchIndex), sizeof(Rpp64u) * this->nBatchSize));
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.dstBatchIndex), sizeof(Rpp64u) * this->nBatchSize));

        for(int i = 0; i < 10; i++)
        {
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.floatArr[i].floatmem), sizeof(Rpp32f) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.uintArr[i].uintmem), sizeof(Rpp32u) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.intArr[i].intmem), sizeof(Rpp32s) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.ucharArr[i].ucharmem), sizeof(Rpp8u) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.charArr[i].charmem), sizeof(Rpp8s) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.float3Arr[i].floatmem), sizeof(Rpp32f) * this->nBatchSize * 3));
        }

        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.rgbArr.rgbmem), sizeof(RpptRGB) * this->nBatchSize));
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
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 372877312));
#else
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 8294400));   // 3840 x 2160
#endif
        CHECK_RETURN_STATUS(hipHostMalloc(&(this->initHandle->mem.mgpu.scratchBufferPinned.floatmem), sizeof(Rpp32f) * 8294400));    // 3840 x 2160
    }
};

Handle::Handle(size_t batchSize, rppAcceleratorQueue_t stream) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    this->impl->device = get_device_id();
    this->impl->ctx = get_ctx();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBuffer();
    RPP_LOG_I(*this);
}

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
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

    free(this->GetInitHandle()->mem.mgpu.csrcSize.height);
    free(this->GetInitHandle()->mem.mgpu.csrcSize.width);
    free(this->GetInitHandle()->mem.mgpu.cdstSize.height);
    free(this->GetInitHandle()->mem.mgpu.cdstSize.width);
    free(this->GetInitHandle()->mem.mgpu.cmaxSrcSize.height);
    free(this->GetInitHandle()->mem.mgpu.cmaxSrcSize.width);
    free(this->GetInitHandle()->mem.mgpu.cmaxDstSize.height);
    free(this->GetInitHandle()->mem.mgpu.cmaxDstSize.width);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.x);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.y);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.roiHeight);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.roiWidth);
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.srcSize.height));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.srcSize.width));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.dstSize.height));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.dstSize.width));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.maxSrcSize.height));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.maxSrcSize.width));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.maxDstSize.height));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.maxDstSize.width));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.roiPoints.x));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.roiPoints.y));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.roiPoints.roiHeight));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.roiPoints.roiWidth));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.inc));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.dstInc));

    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.srcBatchIndex));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.dstBatchIndex));

    for(int i = 0; i < 10; i++)
    {
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.floatArr[i].floatmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.uintArr[i].uintmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.intArr[i].intmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.ucharArr[i].ucharmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.charArr[i].charmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.float3Arr[i].floatmem));
    }

    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.rgbArr.rgbmem));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem));
    CHECK_RETURN_STATUS(hipHostFree(this->GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem));
}

void Handle::rpp_destroy_object_host()
{
    free(this->GetInitHandle()->mem.mcpu.srcSize);
    free(this->GetInitHandle()->mem.mcpu.dstSize);
    free(this->GetInitHandle()->mem.mcpu.maxSrcSize);
    free(this->GetInitHandle()->mem.mcpu.maxDstSize);
    free(this->GetInitHandle()->mem.mcpu.roiPoints);
    free(this->GetInitHandle()->mem.mcpu.srcBatchIndex);
    free(this->GetInitHandle()->mem.mcpu.dstBatchIndex);
    free(this->GetInitHandle()->mem.mcpu.inc);
    free(this->GetInitHandle()->mem.mcpu.dstInc);

    for(int i = 0; i < 10; i++)
    {
        free(this->GetInitHandle()->mem.mcpu.floatArr[i].floatmem);
        free(this->GetInitHandle()->mem.mcpu.uintArr[i].uintmem);
        free(this->GetInitHandle()->mem.mcpu.intArr[i].intmem);
        free(this->GetInitHandle()->mem.mcpu.ucharArr[i].ucharmem);
        free(this->GetInitHandle()->mem.mcpu.charArr[i].charmem);
    }

    free(this->GetInitHandle()->mem.mcpu.rgbArr.rgbmem);
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

void Handle::EnableProfiling(bool enable)
{
    this->impl->enable_profiling = enable;
}

void Handle::ResetKernelTime()
{
    this->impl->profiling_result = 0.0;
}

void Handle::AccumKernelTime(float curr_time)
{
    this->impl->profiling_result += curr_time;
}

float Handle::GetKernelTime() const
{
    return this->impl->profiling_result;
}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               bool is_kernel_str,
                               const std::string& kernel_src)
{
    auto obj = this->impl->cache.AddKernel(*this,
                                           algorithm,
                                           network_config,
                                           program_name,
                                           kernel_name,
                                           vld,
                                           vgd,
                                           params,
                                           cache_index,
                                           is_kernel_str,
                                           kernel_src);
    return this->Run(obj);
}

bool Handle::HasKernel(const std::string& algorithm, const std::string& network_config) const
{
    return this->impl->cache.HasKernels(algorithm, network_config);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config)
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}

const std::vector<Kernel>& Handle::GetKernelsImpl(const std::string& algorithm, const std::string& network_config)
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k)
{
    this->impl->set_ctx();
    if(this->impl->enable_profiling)
        return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else
        return k.Invoke(this->GetStream());
}

Program Handle::LoadProgram(const std::string& program_name,
                            std::string params,
                            bool is_kernel_str,
                            const std::string& kernel_src)
{
    this->impl->set_ctx();

    params += " -mcpu=" + this->GetDeviceName();
    auto cache_file =
        rpp::LoadBinary(this->GetDeviceName(), program_name, params, is_kernel_str);
    if(cache_file.empty())
    {
        auto p =
            HIPOCProgram{program_name, params, is_kernel_str, this->GetDeviceName(), kernel_src};

        return p;
    }
    else
    {
        return HIPOCProgram{program_name, cache_file};
    }
}

void Handle::Finish() const
{
    this->impl->set_ctx();
    auto ev = make_hip_event();
    hipEventRecord(ev.get(), this->GetStream());
    auto status = hipEventSynchronize(ev.get()); // hipStreamSynchronize is broken, so we use hipEventSynchronize instead
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Failed hip sychronization");
}

void Handle::Flush() const {}

bool Handle::IsProfilingEnabled() const
{
    return this->impl->enable_profiling;
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
    hipGetDeviceProperties(&props, this->impl->device);
    std::string name(props.gcnArchName);
    return name;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    return os;
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

Allocator::ManageDataPtr Handle::Create(std::size_t sz)
{
    this->Finish();
    return this->impl->allocator(sz);
}

Allocator::ManageDataPtr& Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
    return ddata;
}

void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size)
{
    this->impl->set_ctx();
    auto status = hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Hip error copying buffer: ");
}

shared<ConstData_t> Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t)
{
    auto cdata = reinterpret_cast<const char*>(data);
    return {cdata + offset, null_deleter{}};
}

} // namespace rpp

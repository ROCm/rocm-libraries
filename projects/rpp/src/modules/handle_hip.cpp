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

#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

#include "device_name.hpp"
#include "errors.hpp"
#include "handle.hpp"
#include "rocfft_runtime.hpp"

namespace rpp {

// Get current context
// We leak resources for now as there is no hipCtxRetain API
hipCtx_t get_ctx() {
    auto status = hipInit(0);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "hipInit failed");
    hipCtx_t ctx{};
    return ctx;
}

std::size_t GetAvailableMemory() {
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz) {
    if (sz > GetAvailableMemory())
        RPP_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = hipMalloc(&result, sz);
    if (status != hipSuccess) {
        status = hipHostMalloc(&result, sz);
        if (status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "Hip error creating buffer " + std::to_string(sz) + ": ");
    }
    return result;
}

void default_deallocator(void*, void* mem) {
    auto status = hipFree(mem);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "hipFree failed");
}

int get_device_id()  // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) RPP_THROW("No device");
    return device;
}

void set_ctx(hipCtx_t ctx) {
    auto status = 0;
    if (status != hipSuccess) RPP_THROW("Error setting context");
}

struct HandleImpl {
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    hipCtx_t ctx{};
    StreamPtr stream = nullptr;
    int device = -1;
    Allocator allocator{};
    bool enable_profiling = false;
    float profiling_result = 0.0;
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    InitHandle* initHandle = nullptr;

    // Whether rocfft_setup() succeeded for this handle. False when librocfft could not be
    // resolved at runtime, in which case no rocFFT entry point may be called. Set lazily on the
    // first spectrogram call rather than at construction, so that handles which never touch audio
    // never pay for the dlopen - see ensure_rocfft_initialized().
    bool rocfft_initialized = false;

    // Set when librocfft was loaded successfully but rocfft_setup() then failed. Distinguishes a
    // missing library (for which dlerror has a message) from a library that is present but could
    // not initialize (for which it does not), so the diagnostic can say which happened.
    bool rocfft_setup_failed = false;

    // Whether the lazy rocFFT initialization above has already been attempted, successfully or
    // not. Keeps a failed attempt from being retried (and re-reported) on every call.
    bool rocfft_init_attempted = false;

    // A cached rocFFT plan, together with everything whose lifetime is tied to it.
    struct RocfftPlanEntry {
        rocfft_plan plan = nullptr;
        rocfft_plan_description desc = nullptr;
        size_t planBatch = 0;      // batch count baked into the plan at creation
        void* workBuffer = nullptr;  // sized for this plan; owned here, not by the caller
        size_t workBufferSize = 0;
    };

    // rocFFT plan cache, keyed by transform size. One entry per distinct nfft, so a pipeline
    // feeding variable-length audio through a constant nfft hits the cache every call.
    //
    // Mutated without a lock, consistent with the one-handle-per-thread assumption the rest of
    // this handle already makes.
    std::map<int, RocfftPlanEntry> rocfft_plan_cache;

    // Audio scratch buffer size (the buffer itself is mem.mgpu.scratchBufferHip, shared with the
    // image kernels; this tracks how many floats it currently holds).
    size_t audioScratchBufferSize = 0;

    // A host handle must not initialize HIP. The GPU-facing Handle constructor
    // initializes ctx explicitly after constructing this backend-neutral state.
    HandleImpl() = default;

    static StreamPtr reference_stream(hipStream_t s) {
        return StreamPtr{s, null_deleter{}};
    }

    void elapsed_time(hipEvent_t start, hipEvent_t stop) {
        if (enable_profiling) {
            auto status = hipEventElapsedTime(&this->profiling_result, start, stop);
            if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "hipEventElapsedTime failed");
        }
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler() {
        return std::bind(&HandleImpl::elapsed_time, this, std::placeholders::_1,
                         std::placeholders::_2);
    }

    void set_ctx() {
        rpp::set_ctx(this->ctx);
        // rpp::set_device(this->device);
        // Check device matches
        if (this->device != get_device_id()) RPP_THROW("Running handle on wrong device");
    }

    void PreInitializeBufferCPU() {
        this->initHandle = new InitHandle();
        this->initHandle->mem.mcpu.scratchBufferHost =
            (Rpp32f*)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize);  // 7680 * 4320 * 3
    }

    void PreInitializeBuffer() {
        this->PreInitializeBufferCPU();

        // Default scratch buffer sized for image processing (4K resolution: 3840 x 2160)
        // Audio functions will reallocate to larger size when needed via EnsureAudioScratchBuffer()
        auto status = hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem),
                                sizeof(Rpp32f) * 8294400);  // 3840 x 2160
        if (status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipMalloc failed for scratchBufferHip");
        this->audioScratchBufferSize = 8294400;
        status = hipHostMalloc(&(this->initHandle->mem.mgpu.scratchBufferPinned.floatmem),
                               sizeof(Rpp32f) * 8294400);  // 3840 x 2160
        if (status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipHostMalloc failed for scratchBufferPinned");
    }

    // Ensure scratch buffer is large enough for audio operations.
    // Called lazily by audio functions when they need more scratch memory.
    // Returns RPP_SUCCESS if buffer is sufficient or was successfully reallocated.
    RppStatus EnsureAudioScratchBuffer(size_t requiredFloats) {
        if (requiredFloats <= this->audioScratchBufferSize) return RPP_SUCCESS;

        // Allocate the replacement before releasing the existing buffer. The scratch buffer is
        // shared with the image kernels, which use it without a null check, so a failed regrow
        // must leave the handle exactly as it was rather than strand a null pointer in it.
        Rpp32f* newBuffer = nullptr;
        auto status = hipMalloc(&newBuffer, sizeof(Rpp32f) * requiredFloats);
        if (status != hipSuccess) return RPP_ERROR_NOT_ENOUGH_MEMORY;

        status = hipFree(this->initHandle->mem.mgpu.scratchBufferHip.floatmem);
        if (status != hipSuccess) {
            (void)hipFree(newBuffer);
            return RPP_ERROR_HIP_RUNTIME;
        }

        this->initHandle->mem.mgpu.scratchBufferHip.floatmem = newBuffer;
        this->audioScratchBufferSize = requiredFloats;
        return RPP_SUCCESS;
    }
};

Handle::Handle(size_t batchSize, rppAcceleratorQueue_t stream) : impl(new HandleImpl()) {
    impl->nBatchSize = batchSize;
    impl->backend = RppBackend::RPP_HIP_BACKEND;
    this->impl->device = get_device_id();
    this->impl->ctx = get_ctx();

    if (stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);

    // rocFFT is deliberately not touched here. It backs the spectrogram kernel only, is resolved
    // at runtime rather than linked (see rocfft_dlopen/rocfft_runtime.hpp), and is loaded on the
    // first call that needs it - so an image-only pipeline never pays the dlopen, and a missing
    // or unusable librocfft cannot stop a handle being created for the rest of RPP. See
    // ensure_rocfft_initialized() / get_rocfft_plan() below.
    impl->PreInitializeBuffer();
}

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl()) {
    impl->nBatchSize = batchSize;
    impl->backend = RppBackend::RPP_HOST_BACKEND;
    numThreads = std::min(numThreads, std::thread::hardware_concurrency());
    if (numThreads == 0) numThreads = batchSize;
    impl->numThreads = numThreads;
    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBufferCPU();
}

Handle::~Handle() {}

void Handle::SetStream(rppAcceleratorQueue_t streamID) const {
    this->impl->stream = HandleImpl::reference_stream(streamID);
}

void Handle::rpp_destroy_object_gpu() {
    // Before any of the memory teardown below, which ends in rpp_destroy_object_host() clearing
    // impl. Only touch rocFFT if it was resolved and set up for this handle; otherwise its entry
    // points are unbound and the plan cache is necessarily empty. Clearing the flag keeps this
    // safe to re-enter if a HIP release below fails and the caller destroys the handle again.
    if (this->impl->rocfft_initialized) {
        // Destroy all cached rocFFT plans, along with the work buffer each one owns
        for (auto& cache_entry : this->impl->rocfft_plan_cache) {
            auto& entry = cache_entry.second;
            rocfft_plan_destroy(entry.plan);
            rocfft_plan_description_destroy(entry.desc);
            if (entry.workBuffer != nullptr) (void)hipFree(entry.workBuffer);
        }
        this->impl->rocfft_plan_cache.clear();

        // Cleanup rocFFT library once per handle (best-effort, don't throw to allow destruction
        // to proceed)
        rocfft_cleanup();
        this->impl->rocfft_initialized = false;
    }

    auto& memory = this->GetInitHandle()->mem;
    // A failed HIP release leaves the handle available for another destroy attempt.
    // Never release an allocation twice or discard the remaining allocation state.
    if (memory.mgpu.scratchBufferHip.floatmem != nullptr) {
        auto status = hipFree(memory.mgpu.scratchBufferHip.floatmem);
        if (status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipFree failed for scratchBufferHip");
        memory.mgpu.scratchBufferHip.floatmem = nullptr;
    }
    if (memory.mgpu.scratchBufferPinned.floatmem != nullptr) {
        auto status = hipHostFree(memory.mgpu.scratchBufferPinned.floatmem);
        if (status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "hipHostFree failed for scratchBufferPinned");
        memory.mgpu.scratchBufferPinned.floatmem = nullptr;
    }
    this->rpp_destroy_object_host();
}

void Handle::rpp_destroy_object_host() {
    free(this->GetInitHandle()->mem.mcpu.scratchBufferHost);
    delete this->GetInitHandle();
    this->impl = nullptr;
}

size_t Handle::GetBatchSize() const {
    return this->impl->nBatchSize;
}

Rpp32u Handle::GetNumThreads() const {
    return this->impl->numThreads;
}

RppBackend Handle::GetBackend() const {
    return this->impl->backend;
}

void Handle::SetBatchSize(size_t bSize) const {
    this->impl->nBatchSize = bSize;
}

rppAcceleratorQueue_t Handle::GetStream() const {
    return impl->stream.get();
}

InitHandle* Handle::GetInitHandle() const {
    return impl->initHandle;
}

void Handle::SetAllocator(rppAllocatorFunction allocator, rppDeallocatorFunction deallocator,
                          void* allocatorContext) const {
    this->impl->allocator.allocator = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;
    this->impl->allocator.context = allocatorContext;
}

std::size_t Handle::GetLocalMemorySize() {
    int result;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                        this->impl->device);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetGlobalMemorySize() {
    size_t result;
    auto status = hipDeviceTotalMem(&result, this->impl->device);

    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status);

    return result;
}

std::string Handle::GetDeviceName() {
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, this->impl->device);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "hipGetDeviceProperties failed");
    std::string name(props.gcnArchName);
    return name;
}

// No HIP API that could return maximum memory allocation size for a single object.
std::size_t Handle::GetMaxMemoryAllocSize() {
    if (m_MaxMemoryAllocSizeCached == 0) {
        size_t free, total;
        auto status = hipMemGetInfo(&free, &total);
        if (status != hipSuccess) RPP_THROW_HIP_STATUS(status, "Failed getting available memory");
        m_MaxMemoryAllocSizeCached = floor(total * 0.85);
    }

    return m_MaxMemoryAllocSizeCached;
}

std::size_t Handle::GetMaxComputeUnits() {
    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device);
    if (status != hipSuccess) RPP_THROW_HIP_STATUS(status);

    return result;
}

RppStatus Handle::EnsureAudioScratchBuffer(size_t requiredFloats) {
    return this->impl->EnsureAudioScratchBuffer(requiredFloats);
}

namespace {

// Load librocfft and run rocfft_setup() once per handle, on the first call that needs it.
// Returns true when rocFFT is usable. A failed attempt is remembered, not retried.
bool ensure_rocfft_initialized(HandleImpl& impl) {
    if (impl.rocfft_init_attempted) return impl.rocfft_initialized;
    impl.rocfft_init_attempted = true;

    // rocFFT backs the spectrogram kernel only, and is resolved at runtime rather than linked
    // (see rocfft_dlopen/rocfft_runtime.hpp). It is therefore an optional component: a missing
    // or unusable librocfft must not stop the rest of RPP working.
    if (rocfft_dlopen::initialize()) {
        // rocFFT increments its global usage count before some setup failures, so balance it
        // with rocfft_cleanup() to keep a later handle from skipping init.
        if (rocfft_setup() != rocfft_status_success) {
            rocfft_cleanup();
            impl.rocfft_setup_failed = true;
        } else {
            impl.rocfft_initialized = true;
        }
    }
    return impl.rocfft_initialized;
}

}  // namespace

// Get or create the handle's cached rocFFT plan for this transform size. See the declaration in
// hip_tensor_executors.hpp for the contract around *planBatch.
RppStatus get_rocfft_plan(Handle& handle, int nfft, size_t batchCount, rocfft_plan* plan,
                          size_t* planBatch, void** workBuffer, size_t* workBufferSize) {
    // rocFFT is the only FFT implementation on the HIP path - there is no fallback to fall back
    // to. Fail the call rather than return a plan that cannot be executed, and say once why.
    if (!ensure_rocfft_initialized(*handle.impl)) {
        static std::once_flag reported;
        bool setupFailed = handle.impl->rocfft_setup_failed;
        std::call_once(reported, [setupFailed]() {
            std::cerr << "RPP: GPU audio FFT is unavailable because ";
            if (setupFailed)
                std::cerr << "rocfft_setup() failed. librocfft was loaded, so this is a rocFFT "
                          << "runtime or device initialization problem rather than a missing "
                          << "package.";
            else
                std::cerr << "librocfft could not be loaded (" << rocfft_dlopen::error_message()
                          << "). Install the rocFFT runtime package to enable the HIP "
                          << "spectrogram kernel.";
            std::cerr << std::endl;
        });
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    auto& cache = handle.impl->rocfft_plan_cache;

    // A rocFFT plan bakes in its batch count, so an entry can only serve a call whose batch fits
    // inside it. Keeping the largest batch seen for this nfft means variable-length audio at a
    // constant nfft stops missing the cache after the first few calls; the caller pads its input
    // with zeros up to planBatch. The trade is that one outlier-large batch makes every later
    // call transform the surplus windows, so the entry is grown rather than kept per batch size.
    auto it = cache.find(nfft);
    if (it != cache.end() && it->second.planBatch >= batchCount) {
        *plan = it->second.plan;
        *planBatch = it->second.planBatch;
        *workBuffer = it->second.workBuffer;
        *workBufferSize = it->second.workBufferSize;
        return RPP_SUCCESS;
    }

    // Create new plan
    rocfft_plan_description new_desc = nullptr;
    if (rocfft_plan_description_create(&new_desc) != rocfft_status_success)
        return RPP_ERROR_NOT_ENOUGH_MEMORY;

    size_t lengths[1] = {static_cast<size_t>(nfft)};
    size_t inStride[1] = {1};
    size_t outStride[1] = {1};
    int numBins = (nfft / 2 + 1);

    if (rocfft_plan_description_set_data_layout(
            new_desc, rocfft_array_type_real, rocfft_array_type_hermitian_interleaved, nullptr,
            nullptr, 1, inStride, static_cast<size_t>(nfft), 1, outStride,
            static_cast<size_t>(numBins)) != rocfft_status_success) {
        rocfft_plan_description_destroy(new_desc);
        return RPP_ERROR_INVALID_ARGUMENTS;
    }

    rocfft_plan new_plan = nullptr;
    if (rocfft_plan_create(&new_plan, rocfft_placement_notinplace,
                           rocfft_transform_type_real_forward, rocfft_precision_single, 1, lengths,
                           batchCount, new_desc) != rocfft_status_success) {
        rocfft_plan_description_destroy(new_desc);
        return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }

    // The work buffer depends only on the plan, so it is allocated with the plan and reused for
    // its lifetime rather than being malloc'd and freed on every call.
    size_t newWorkBufferSize = 0;
    if (rocfft_plan_get_work_buffer_size(new_plan, &newWorkBufferSize) != rocfft_status_success) {
        rocfft_plan_destroy(new_plan);
        rocfft_plan_description_destroy(new_desc);
        return RPP_ERROR;
    }

    void* newWorkBuffer = nullptr;
    if (newWorkBufferSize > 0 &&
        hipMalloc(&newWorkBuffer, newWorkBufferSize) != hipSuccess) {
        rocfft_plan_destroy(new_plan);
        rocfft_plan_description_destroy(new_desc);
        return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }

    // Only now that the replacement is fully built, retire the entry being grown. Doing it in
    // this order leaves the cache holding a usable plan if any step above failed.
    if (it != cache.end()) {
        rocfft_plan_destroy(it->second.plan);
        rocfft_plan_description_destroy(it->second.desc);
        if (it->second.workBuffer != nullptr) (void)hipFree(it->second.workBuffer);
        cache.erase(it);
    }

    HandleImpl::RocfftPlanEntry entry;
    entry.plan = new_plan;
    entry.desc = new_desc;
    entry.planBatch = batchCount;
    entry.workBuffer = newWorkBuffer;
    entry.workBufferSize = newWorkBufferSize;
    cache[nfft] = entry;

    *plan = new_plan;
    *planBatch = batchCount;
    *workBuffer = newWorkBuffer;
    *workBufferSize = newWorkBufferSize;

    return RPP_SUCCESS;
}

}  // namespace rpp

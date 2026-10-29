// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/config.h>
#include <miopen/handle.hpp>

#include <miopen/binary_cache.hpp>
#include <miopen/config.hpp>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/timer.hpp>
#include <miopen/unique_path.hpp>

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
#include <miopen/write_file.hpp>
#endif

#include <miopen/filesystem.hpp>
#include <miopen/load_file.hpp>

#include <miopen/solver/ck_impl_lib_loader.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iterator>
#include <mutex>
#include <shared_mutex>
#include <thread>

#if MIOPEN_USE_HIPBLASLT
#include <miopen/hipblaslt_errors.hpp>
#endif

/// hipMemGetInfo constantly fails on gfx906/900 and Navi21.
/// Brute-force W/A: return fixed values.
#define WORKAROUND_FAULTY_HIPMEMGETINFO_VEGA_NAVI2X (HIP_PACKAGE_VERSION_FLAT >= 5007000000ULL)

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEVICE_CU)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS)

namespace miopen {

namespace {

// Eagerly load the per-architecture Composable Kernel dynamic library so the
// one-time dlopen cost is paid at handle creation instead of on first CK call.
// The loader caches per process and never throws, so this is safe to call
// unconditionally during construction. Takes the device name by value (rather
// than the handle) so it can be called from the constructor without invoking the
// virtual GetTargetProperties on a not-yet-fully-constructed Handle.
void PrefetchComposableKernel(const std::string& device_name)
{
    (void)solver::CkImplLibLoader::Get(device_name);
}

hipError_t hip_mem_get_info_wrapper(std::size_t* const free, std::size_t* const total)
{
#if WORKAROUND_FAULTY_HIPMEMGETINFO_VEGA_NAVI2X
    const auto status = hipMemGetInfo(free, total);
    if(status == hipSuccess)
        return status;
    MIOPEN_LOG_W("hipMemGetInfo error, status: " << status);
    assert(free != nullptr && total != nullptr);
    *free  = 16ULL * 1024 * 1024 * 1024; // 16 GiB
    *total = *free;
    return hipSuccess;
#else
    return hipMemGetInfo(free, total);
#endif
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hip_mem_get_info_wrapper(&free, &total);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz)
{
    const auto available = GetAvailableMemory();
    MIOPEN_LOG_I2("GetAvailableMemory " << available);
    if(sz > available)
        MIOPEN_LOG_I("GetAvailableMemory reports unsufficient memory to allocate " << sz);
    void* ptr;
    const auto status = hipMalloc(&ptr, sz);
    if(status == hipSuccess)
    {
        MIOPEN_LOG_I2("hipMalloc " << sz << " at " << ptr << " Ok");
        return ptr;
    }
    const auto status_host = hipHostMalloc(&ptr, sz);
    if(status_host == hipSuccess)
    {
        MIOPEN_LOG_I2("hipHostMalloc " << sz << " at " << ptr << " Ok");
        return ptr;
    }
    MIOPEN_LOG_W("hipMalloc " << sz << " status: " << status);
    MIOPEN_THROW_HIP_STATUS(status_host, "hipHostMalloc " + std::to_string(sz));
}

[[maybe_unused]] inline std::string to_string(void* const ptr)
{
    std::ostringstream oss;
    oss << ptr;
    return oss.str();
}

void default_deallocator(void*, void* mem)
{
    size_t size = 0;
    auto status = hipMemPtrGetInfo(mem, &size);
    if(status != hipSuccess)
        MIOPEN_LOG_W("hipMemPtrGetInfo at " << mem << " status: " << status);
    status = hipFree(mem);
    if(status != hipSuccess)
    {
        MIOPEN_THROW_HIP_STATUS(status,
                                "hipFree " + std::to_string(size) + " at " + to_string(mem));
    }
    else
        MIOPEN_LOG_I2("hipFree " << size << " at " << mem << " Ok");
}

} // namespace

// MIOPEN_INTERNALS_EXPORT set here because the function isn't present in headers
// but called from test/gtest/handle_hip_device.cpp
MIOPEN_INTERNALS_EXPORT int get_device_id() // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "No device");
    return device;
}

// MIOPEN_INTERNALS_EXPORT set here because the function isn't present in headers
// but called from test/gtest/handle_hip_device.cpp
MIOPEN_INTERNALS_EXPORT void set_device(int id)
{
    auto status = hipSetDevice(id);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Error setting device " + std::to_string(id));
}

#if MIOPEN_BUILD_DEV
int set_default_device()
{
    int n;
    auto status = hipGetDeviceCount(&n);
    if(status != hipSuccess)
        MIOPEN_THROW("Error getting device count");
    // Pick device based on process id
    auto pid = ::getpid();
    assert(pid > 0);
    set_device(pid % n);
    return (pid % n);
}
#endif

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static thread_local unsigned int miopenHandle_current_stream_id = 0;
// Overrides the stream-pool index when set. See Handle::SetExclusiveStream.
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static thread_local hipStream_t miopenHandle_exclusive_stream = nullptr;
struct HandleImpl
{
    // typedef MIOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl() { (void)hipInit(0); }

    StreamPtr create_stream()
    {
        hipStream_t result;
        auto status = hipStreamCreate(&result);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    StreamPtr create_stream_non_blocking()
    {
        hipStream_t result;
        auto status = hipStreamCreateWithFlags(&result, hipStreamNonBlocking);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s) { return StreamPtr{s, null_deleter{}}; }

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        if(enable_profiling)
            (void)hipEventElapsedTime(&this->profiling_result, start, stop);
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(
            &HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    void set_ctx() const { miopen::set_device(this->device); }

    std::string get_device_name() const
    {
        hipDeviceProp_t props{};
        (void)hipGetDeviceProperties(&props, device);
        const std::string name(props.gcnArchName);
        MIOPEN_LOG_NQI("Raw device name: " << name);
        return name; // NOLINT (performance-no-automatic-move)
    }

    std::shared_timed_mutex stream_pool_mutex;
    // the main stream and main rocblas_handle rhandle_

#if MIOPEN_USE_ROCBLAS
    rocblas_handle_ptr rhandle_;
    using RocblasHandlePtrPool = std::vector<rocblas_handle_ptr>;
#endif
#if MIOPEN_USE_HIPBLASLT
    hipblasLt_handle_ptr hip_blasLt_handle;
    using HipblasLtHandlePtrPool = std::vector<hipblasLt_handle_ptr>;
#endif

    StreamPtr root_stream = nullptr;

    using StreamPtrPool = std::vector<StreamPtr>;
    struct MultiStreamResourses
    {

#if MIOPEN_USE_ROCBLAS && MIOPEN_USE_HIPBLASLT
        //  (rocBLAS doc):rocBLAS handle contains allocated device memory that must not be shared by
        //  multiple
        //  asynchronous streams at the same time.
        //  Each parallel thread must use its own rocblas_handle.
        RocblasHandlePtrPool rhandle_pool;
        HipblasLtHandlePtrPool hhandle_pool;
        void add_resours(StreamPtr s_ptr, rocblas_handle_ptr r_ptr, hipblasLt_handle_ptr h_ptr)
        {
            stream_pool.push_back(std::move(s_ptr));
            rhandle_pool.push_back(std::move(r_ptr));
            hhandle_pool.push_back(std::move(h_ptr));
        }
#elif MIOPEN_USE_ROCBLAS
        RocblasHandlePtrPool rhandle_pool;
        void add_resours(StreamPtr s_ptr, rocblas_handle_ptr r_ptr)
        {
            stream_pool.push_back(s_ptr);
            rhandle_pool.push_back(std::move(r_ptr));
        }
#elif MIOPEN_USE_HIPBLASLT
        HipblasLtHandlePtrPool hhandle_pool;
        void add_resours(StreamPtr s_ptr, hipblasLt_handle_ptr h_ptr)
        {
            stream_pool.push_back(s_ptr);
            hhandle_pool.push_back(std::move(h_ptr));
        }
#else
        void add_stream(StreamPtr s_ptr) { stream_pool.push_back(s_ptr); }
#endif
        //  stream_pool used as cache for parallel streams created by MIOpen.
        StreamPtrPool stream_pool;
    };

    MultiStreamResourses* ms_resourse_ptr;
    std::map<miopenAcceleratorQueue_t, MultiStreamResourses> extra_stream_map;

    bool enable_profiling  = false;
    float profiling_result = 0.0;
    int device             = -1;
    Allocator allocator{};
    KernelCache cache;
    TargetProperties target_properties;
    mutable StreamTracker stream_tracker_;
    /// Guards the two below: two threads sharing a handle can run find, and
    /// so reach GetScratchBuffer, at once.
    std::mutex scratch_mutex_;
    std::weak_ptr<ScratchAllocation> active_scratch_;
    std::size_t scratch_cap_ = 0;
};

Handle::Handle(miopenAcceleratorQueue_t stream) : impl(std::make_unique<HandleImpl>())
{
    miopenHandle_current_stream_id = 0;
    this->impl->device             = get_device_id();

    if(stream == nullptr)
        this->impl->root_stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->root_stream = HandleImpl::reference_stream(stream);

    this->impl->extra_stream_map.emplace(stream, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map.begin()->second);

    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    this->impl->rhandle_ = CreateRocblasHandle(stream);
#endif
#if MIOPEN_USE_HIPBLASLT
    this->impl->hip_blasLt_handle = CreateHipblasLtHandle();
#endif
    this->impl->target_properties.Init(this);
    PrefetchComposableKernel(this->impl->target_properties.Name());
    MIOPEN_LOG_NQI(*this);
}

Handle::Handle() : impl(std::make_unique<HandleImpl>())
{
    miopenHandle_current_stream_id = 0;
#if MIOPEN_BUILD_DEV
    this->impl->device      = set_default_device();
    this->impl->root_stream = impl->create_stream();
#else
    this->impl->device      = get_device_id();
    this->impl->root_stream = HandleImpl::reference_stream(nullptr);
#endif
    auto root_stream = this->impl->root_stream.get();
    this->impl->extra_stream_map.emplace(root_stream, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map.begin()->second);

    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    this->impl->rhandle_ = CreateRocblasHandle(root_stream);
#endif
#if MIOPEN_USE_HIPBLASLT
    this->impl->hip_blasLt_handle = CreateHipblasLtHandle();
#endif
    this->impl->target_properties.Init(this);
    PrefetchComposableKernel(this->impl->target_properties.Name());
    MIOPEN_LOG_NQI(*this);
}

Handle::~Handle() {}

// not MT safe
void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    miopenHandle_current_stream_id = 0;

    this->impl->root_stream = HandleImpl::reference_stream(streamID);

    this->impl->extra_stream_map.emplace(streamID, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map[streamID]);

#if MIOPEN_USE_ROCBLAS
    rocblas_set_stream(this->rhandle().get(), this->GetStream());
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

void Handle::SetStreamFromPool(int streamID) const { miopenHandle_current_stream_id = streamID; }

void Handle::ReserveExtraStreamsInPool(int cnt) const
{
    std::unique_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);

    int last_stream_id = this->impl->ms_resourse_ptr->stream_pool.size();

    if(last_stream_id < cnt)
    {
        for(; last_stream_id < cnt; last_stream_id++)
        {
            auto new_stream = this->impl->create_stream_non_blocking();
#if MIOPEN_USE_ROCBLAS && MIOPEN_USE_HIPBLASLT
            auto new_rhandle = CreateRocblasHandle(new_stream.get());
            auto new_hhandle = CreateHipblasLtHandle();
            this->impl->ms_resourse_ptr->add_resours(
                std::move(new_stream), std::move(new_rhandle), std::move(new_hhandle));
#elif MIOPEN_USE_ROCBLAS
            auto new_rhandle = CreateRocblasHandle(new_stream.get());
            this->impl->ms_resourse_ptr->add_resours(std::move(new_stream), std::move(new_rhandle));
#elif MIOPEN_USE_HIPBLASLT
            auto new_hhandle = CreateHipblasLtHandle();
            this->impl->ms_resourse_ptr->add_resours(std::move(new_stream), std::move(new_hhandle));
#else
            this->impl->ms_resourse_ptr->add_stream(std::move(new_stream));
#endif
        }
    }
}

StreamTracker::StreamPtr Handle::CreateExclusiveStream() const
{
    impl->set_ctx();
    return impl->create_stream_non_blocking();
}

void Handle::SetExclusiveStream(miopenAcceleratorQueue_t stream) const
{
    miopenHandle_exclusive_stream = stream;
}

miopenAcceleratorQueue_t Handle::GetStream() const
{
    if(miopenHandle_exclusive_stream != nullptr)
        return miopenHandle_exclusive_stream;
    if(miopenHandle_current_stream_id == 0)
        return impl->root_stream.get();
    // locking only if handle in multistream mode
    std::shared_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);
    return this->impl->ms_resourse_ptr->stream_pool.at(miopenHandle_current_stream_id - 1).get();
}

StreamTracker& Handle::GetStreamTracker() const { return impl->stream_tracker_; }

std::shared_ptr<ScratchAllocation> Handle::GetScratchBuffer(std::size_t sz) const
{
    if(sz == 0)
        return nullptr;

    const std::lock_guard<std::mutex> lock(impl->scratch_mutex_);
    if(impl->scratch_cap_ == 0)
        impl->scratch_cap_ = GetGlobalMemorySize() / 8;

    if(sz > impl->scratch_cap_)
        return nullptr;

    if(auto existing = impl->active_scratch_.lock(); existing && sz <= existing->size)
        return existing;

    auto alloc            = std::make_shared<ScratchAllocation>();
    alloc->buffer         = impl->allocator(sz);
    alloc->size           = sz;
    impl->active_scratch_ = alloc;
    return alloc;
}

namespace {

/// How long an acquire() blocked on a full backlog waits for the worker before
/// re-testing the backlog itself. The worker normally wakes it within
/// microseconds of a retirement; this only bounds the fallback for a callback
/// that will never arrive.
constexpr auto kDrainCheckInterval = std::chrono::milliseconds(10);

/// Exempts the calling thread from HIP's stream-capture restrictions for its
/// lifetime, restoring the previous mode on exit.
///
/// Keeps the tracker from invalidating a capture it has nothing to do with. Its
/// queries and deallocator calls run on the worker thread or during find, either
/// of which can coincide with a capture on another thread; under the default
/// global mode HIP refuses them and invalidates that capture (issue #12121).
///
/// The streams queried are the tracker's own and never part of a caller's
/// graph, and the scratch released is referenced by none, so the exemption is
/// honest rather than a way to force something through. It cannot cover a call
/// HIP refuses during any capture whatever the mode, such as a synchronous
/// hipMemcpy made by a caller's deallocator.
class ScopedRelaxedCaptureMode
{
public:
    ScopedRelaxedCaptureMode()
    {
        // Swaps in mode_ and writes the mode it replaced back into it.
        active_ = hipThreadExchangeStreamCaptureMode(&mode_) == hipSuccess;
    }

    ~ScopedRelaxedCaptureMode()
    {
        if(active_)
            (void)hipThreadExchangeStreamCaptureMode(&mode_);
    }

    ScopedRelaxedCaptureMode(const ScopedRelaxedCaptureMode&)            = delete;
    ScopedRelaxedCaptureMode& operator=(const ScopedRelaxedCaptureMode&) = delete;

private:
    hipStreamCaptureMode mode_ = hipStreamCaptureModeRelaxed;
    bool active_               = false;
};

} // namespace

std::size_t StreamTracker::MaxDrainingFor(const char* gpu_max_hw_queues)
{
    // ROCclr's own default for GPU_MAX_HW_QUEUES (rocclr/utils/flags.hpp).
    // HIP has no API reporting the effective value, so mirror its parsing.
    long long hw_queues = 4;
    if(gpu_max_hw_queues != nullptr)
    {
        try
        {
            // Signed, so "-1" is rejected rather than wrapping to a huge cap.
            const auto parsed = std::stoll(gpu_max_hw_queues);
            if(parsed > 0)
                hw_queues = parsed;
            else
                MIOPEN_LOG_W("Ignoring non-positive GPU_MAX_HW_QUEUES=" << gpu_max_hw_queues);
        }
        catch(const std::exception&)
        {
            MIOPEN_LOG_W("Ignoring unparseable GPU_MAX_HW_QUEUES=" << gpu_max_hw_queues);
        }
    }

    // Two rings are already spoken for whenever this matters: the handle's
    // own stream, and the slot the blocked acquire() is about to hand out.
    return hw_queues > 3 ? static_cast<std::size_t>(hw_queues - 2) : std::size_t{1};
}

std::size_t StreamTracker::DefaultMaxDraining()
{
    static const std::size_t cap = [] {
        const auto raw = env::getEnvironmentVariable("GPU_MAX_HW_QUEUES");
        return MaxDrainingFor(raw ? raw->c_str() : nullptr);
    }();
    return cap;
}

struct StreamTracker::RetireContext
{
    StreamTracker* tracker = nullptr;
    std::uint64_t ticket   = 0;
};

hipError_t StreamTracker::SweepLocked()
{
    if(draining_.empty())
        return hipSuccess;

    const ScopedRelaxedCaptureMode capture_guard;

    std::vector<std::shared_ptr<ScratchAllocation>> released;
    auto status = hipSuccess;
    for(auto it = draining_.begin(); it != draining_.end();)
    {
        const auto query = it->done ? hipEventQuery(it->done.get()) : hipStreamQuery(it->stream);
        if(query == hipSuccess)
        {
            // Its callback may still arrive; the cleared ticket makes the worker
            // ignore it.
            if(it->scratch)
                released.push_back(std::move(it->scratch));
            it->done.reset();
            it->ticket = 0;
            available_.push_back(std::move(*it));
            it = draining_.erase(it);
            continue;
        }

        // Only hipSuccess proves the kernel retired; every other answer leaves the
        // slot - and the scratch the kernel may still be writing - where it is.
        // The completion check itself answers only hipSuccess or hipErrorNotReady.
        // No capture rejection can reach here: the guard above puts this thread in
        // relaxed mode, which exempts it from every thread's captures, and these
        // streams and events are never captured themselves. The guard fails only
        // on the runtime's process-wide device error, which every call returns
        // before looking at the stream or event, and which never clears - so any
        // other answer means this backlog can no longer drain, and no later query
        // in this loop could answer differently.
        if(query != hipErrorNotReady)
        {
            status = query;
            break;
        }
        ++it;
    }

    // Freed by the worker, never here: this runs inside find with mutex_ held,
    // and the default deallocator's hipFree waits for every stream on the
    // device, so freeing here would stall find - and the worker - until every
    // other abandoned kernel had finished. Lock order is mutex_ then
    // retire_mutex_; nothing takes them the other way round.
    if(!released.empty())
    {
        {
            const std::lock_guard<std::mutex> lock(retire_mutex_);
            std::move(released.begin(), released.end(), std::back_inserter(deferred_scratch_));
        }
        retire_cv_.notify_one();
    }
    return status;
}

void StreamTracker::OnSlotRetired(void* context)
{
    const std::unique_ptr<RetireContext> retire{static_cast<RetireContext*>(context)};
    StreamTracker& tracker = *retire->tracker;
    {
        const std::lock_guard<std::mutex> lock(tracker.retire_mutex_);
        tracker.retired_tickets_.push_back(retire->ticket);
    }
    tracker.retire_cv_.notify_one();
}

void StreamTracker::RetireLoop()
{
    std::unique_lock<std::mutex> retire_lock(retire_mutex_);
    while(true)
    {
        retire_cv_.wait(retire_lock, [this] {
            return !retired_tickets_.empty() || !deferred_scratch_.empty() || stop_worker_;
        });
        if(retired_tickets_.empty() && deferred_scratch_.empty())
            return; // Stopping, with every queued retirement handled.

        auto tickets = std::move(retired_tickets_);
        retired_tickets_.clear();
        auto released = std::move(deferred_scratch_);
        deferred_scratch_.clear();
        retire_lock.unlock();

        if(!tickets.empty())
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            for(const auto ticket : tickets)
            {
                const auto it =
                    std::find_if(draining_.begin(), draining_.end(), [ticket](const Slot& s) {
                        return s.ticket == ticket;
                    });
                // acquire() reclaimed it on its done event first.
                if(it == draining_.end())
                    continue;
                if(it->scratch)
                    released.push_back(std::move(it->scratch));
                it->done.reset();
                it->ticket = 0;
                available_.push_back(std::move(*it));
                draining_.erase(it);
            }
            retired_cv_.notify_all();
        }

        {
            // Outside mutex_, so a deallocator that blocks - hipFree waits for
            // every stream on the device - holds up neither find nor the
            // runtime's callback thread.
            const ScopedRelaxedCaptureMode capture_guard;
            released.clear();
        }
        retire_lock.lock();
    }
}

void StreamTracker::release(Slot slot)
{
    // Back to the pool clean, whatever the caller left on it: an idle slot must
    // not pin a buffer, and a done event left behind would read as complete if
    // this stream were later abandoned without a fresh one. The scratch goes
    // before the lock is taken, so a deallocator that blocks holds up nobody
    // else.
    slot.scratch.reset();
    slot.done.reset();
    slot.ticket = 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    available_.push_back(std::move(slot));
}

void StreamTracker::abandon(Slot slot)
{
    auto retire       = std::make_unique<RetireContext>();
    retire->tracker   = this;
    const auto stream = slot.stream;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if(!worker_.joinable())
            worker_ = std::thread([this] { RetireLoop(); });
        slot.ticket = retire->ticket = ++next_ticket_;
        draining_.push_back(std::move(slot));
    }

    // Queued behind everything already on the stream, so it runs once the
    // evaluation's last kernel has retired; the runtime holds the stream until
    // it returns. Enqueued outside mutex_, which it has no need of. The callback
    // owns the context once queued; if queuing fails, ownership comes back here.
    RetireContext* const context = retire.release();
    const auto status = hipLaunchHostFunc(stream, &StreamTracker::OnSlotRetired, context);
    if(status != hipSuccess)
    {
        const std::unique_ptr<RetireContext> unqueued{context};
        MIOPEN_LOG_W("Could not queue retirement of an abandoned evaluation stream: "
                     << hipGetErrorString(status) << "; acquire() reclaims it once idle");
    }
}

StreamTracker::~StreamTracker()
{
    const ScopedRelaxedCaptureMode capture_guard;

    if(!owned_streams_.empty())
    {
        // The worker may still be retiring slots, so the count is read under the
        // lock it changes them under.
        const auto abandoned = [this] {
            const std::lock_guard<std::mutex> lock(mutex_);
            return draining_.size();
        }();
        MIOPEN_LOG_I("Waiting for " << abandoned << " abandoned stream(s) to drain");

        // Every owned stream, not just the draining ones. Kernels left running by
        // a timed-out evaluation execute from code objects the Handle unloads
        // during teardown; synchronizing all of them does not depend on every
        // caller having released only finished work, and an idle stream costs
        // nothing.
        const auto start = std::chrono::steady_clock::now();
        for(auto& stream : owned_streams_)
            (void)hipStreamSynchronize(stream.get());
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start;

        MIOPEN_LOG_I("Drained " << owned_streams_.size() << " evaluation stream(s) in "
                                << elapsed.count() << " ms");
    }

    // Each callback holds its stream until it returns, so every one has run by
    // now and the worker has the complete set to finish before it stops.
    if(worker_.joinable())
    {
        {
            const std::lock_guard<std::mutex> lock(retire_mutex_);
            stop_worker_ = true;
        }
        retire_cv_.notify_one();
        worker_.join();
    }

    // Slots whose callback could not be queued, released under the guard rather
    // than by member destruction after it.
    draining_.clear();
    available_.clear();
}

StreamTracker::Slot StreamTracker::acquire(const Handle& handle)
{
    std::unique_lock<std::mutex> lock(mutex_);

    // Takes back slots whose kernel has finished but whose callback the worker
    // has not handled yet, so a retired stream is reused rather than duplicated.
    auto status = SweepLocked();

    // Reclaiming costs nothing new, so the cap applies only to growth below.
    const auto cap = MaxDraining();
    if(available_.empty() && draining_.size() >= cap)
    {
        MIOPEN_LOG_I("Waiting for one of " << draining_.size() << " abandoned stream(s) to retire");
        while(available_.empty() && draining_.size() >= cap)
        {
            if(status != hipSuccess)
                MIOPEN_THROW_HIP_STATUS(status, "Waiting for an abandoned evaluation stream");
            // Woken by the worker as it retires a slot. The timeout re-tests the
            // backlog directly, for a callback that will never arrive: one that
            // could not be queued, or a device error.
            retired_cv_.wait_for(lock, kDrainCheckInterval);
            status = SweepLocked();
        }
    }

    if(!available_.empty())
    {
        auto slot = std::move(available_.back());
        available_.pop_back();
        return slot;
    }

    owned_streams_.push_back(handle.CreateExclusiveStream());
    return {owned_streams_.back().get(), {}, {}, 0};
}

void Handle::SetAllocator(miopenAllocatorFunction allocator,
                          miopenDeallocatorFunction deallocator,
                          void* allocatorContext) const
{
    this->impl->allocator.allocator   = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;

    this->impl->allocator.context = allocatorContext;
}

void Handle::EnableProfiling(bool enable) const { this->impl->enable_profiling = enable; }

float Handle::GetKernelTime() const { return this->impl->profiling_result; }

Allocator::ManageDataPtr Handle::Create(std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    return this->impl->allocator(sz);
}

Allocator::ManageDataPtr&
Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz) const
{
    WriteTo(data, ddata.get(), sz);
    return ddata;
}

void Handle::WriteTo(const void* data, Data_t ddata, std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = hipMemcpyWithStream(ddata, data, sz, hipMemcpyHostToDevice, this->GetStream());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
}

void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz) const
{
    ReadTo(data, ddata.get(), sz);
}

void Handle::ReadTo(void* data, ConstData_t ddata, std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = hipMemcpyWithStream(data, ddata, sz, hipMemcpyDeviceToHost, this->GetStream());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size) const
{
    MIOPEN_HANDLE_LOCK
    this->impl->set_ctx();
    auto status = hipMemcpyWithStream(dest, src, size, hipMemcpyDeviceToDevice, this->GetStream());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error copying buffer: ");
}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const fs::path& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               const std::string& kernel_src) const
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
                                           kernel_src);
    return this->Run(obj);
}

Invoker Handle::PrepareInvoker(const InvokerFactory& factory,
                               const std::vector<solver::KernelInfo>& kernels,
                               std::vector<Program>* programs_out) const
{
    std::vector<Kernel> built;
    built.reserve(kernels.size());
    if(programs_out != nullptr)
        programs_out->resize(kernels.size());

    for(auto i = 0; i < kernels.size(); ++i)
    {
        const auto& k        = kernels[i];
        Program* program_out = programs_out != nullptr ? &(*programs_out)[i] : nullptr;

        MIOPEN_LOG_I2("Preparing kernel: " << k.kernel_name);

        const auto kernel = this->impl->cache.AddKernel(*this,
                                                        "",
                                                        "",
                                                        k.kernel_file,
                                                        k.kernel_name,
                                                        k.l_wk,
                                                        k.g_wk,
                                                        k.comp_options,
                                                        kernels.size(),
                                                        "",
                                                        program_out);
        built.push_back(kernel);
    }
    return factory(built);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config) const
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}

std::vector<Kernel> Handle::GetKernelsImpl(const std::string& algorithm,
                                           const std::string& network_config) const
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k, bool coop_launch) const
{
    this->impl->set_ctx();
    auto callback = (this->impl->enable_profiling || MIOPEN_GPU_SYNC)
                        ? this->impl->elapsed_time_handler()
                        : nullptr;
    return k.Invoke(this->GetStream(), callback, coop_launch);
}

Program Handle::LoadProgram(const fs::path& program_name,
                            std::string params,
                            const std::string& kernel_src,
                            bool force_attach_binary) const
{
    this->impl->set_ctx();
    std::string arch_name = this->GetTargetProperties().Name();

    std::string orig_params = params; // make a copy for target ID fallback

#if WORKAROUND_ISSUE_3001
    if(program_name.extension() != ".mlir")
        params = params + " -mcpu=" + this->GetTargetProperties().Name();
#else
    if(program_name.extension() == ".mlir")
    { // no -mcpu
    }
    else if(program_name.extension() == ".s")
    {
        params += " -mcpu=" + LcOptionTargetStrings{this->GetTargetProperties()}.targetId;
    }
    else
    {
        params += " -mcpu=" + this->GetTargetProperties().Name();
    }
#endif

    auto hsaco = miopen::LoadBinary(
        this->GetTargetProperties(), this->GetMaxComputeUnits(), program_name, params);
    if(hsaco.empty())
    {
        const auto arch_target_id = miopen::SplitDelim(arch_name, ':');
        if(arch_target_id.size() > 1)
        {
            // The target name has target ID in there, fall back on the generic code object
            const auto base_arch = arch_target_id.at(0);
            hsaco                = miopen::LoadBinary(this->GetTargetProperties(),
                                       this->GetMaxComputeUnits(),
                                       program_name,
                                       orig_params + " -mcpu=" + base_arch);
        }
    }

    // Still unable to find the object, build it with the available compiler possibly a target ID
    // specific code object
    if(hsaco.empty())
    {
        CompileTimer ct;
        auto p =
            HIPOCProgram{program_name.string(), params, this->GetTargetProperties(), kernel_src};
        ct.Log("Kernel", program_name.string());

        // Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        std::vector<char> binary;
        if(!p.IsCodeObjectInMemory())
            binary = miopen::LoadFile(p.GetCodeObjectPathname());

        miopen::SaveBinary(p.IsCodeObjectInMemory() ? p.GetCodeObjectBlob() : binary,
                           this->GetTargetProperties(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params);

        if(force_attach_binary && p.IsCodeObjectInTempFile())
        {
            MIOPEN_LOG_I2("Attaching a binary to the program for future serialization");
            p.AttachBinary(std::vector<char>{binary.data(), binary.data() + binary.size()});
        }
        else
        {
            MIOPEN_LOG_I2("Skipped attaching a binary to the program for future serialization as "
                          "it is in permanent file storage");
        }

        p.FreeCodeObjectFileStorage();
#else
        fs::path cache_path;

        // If cache is disabled we don't need to dump binary and move it there
        if(!miopen::IsCacheDisabled())
        {
            const auto path = miopen::GetCachePath(false) / miopen::unique_path();
            if(p.IsCodeObjectInMemory())
                miopen::WriteFile(p.GetCodeObjectBlob(), path);
            else
                fs::copy_file(p.GetCodeObjectPathname(), path);
            cache_path =
                miopen::SaveBinary(path, this->GetTargetProperties(), program_name, params);
        }

        if(force_attach_binary && p.IsCodeObjectInTempFile())
        {
            MIOPEN_LOG_I2("Attaching a binary to the program for future serialization");
            if(cache_path.empty())
                p.AttachBinary(LoadFile(p.GetCodeObjectPathname()));
            else
                p.AttachBinary(cache_path);
        }

        p.FreeCodeObjectFileStorage();
#endif
        return p;
    }
    else
    {
        auto p = HIPOCProgram{program_name, hsaco};
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        if(force_attach_binary)
        {
            MIOPEN_LOG_I2("Attaching a binary to the program for future serialization");
            p.AttachBinary(std::vector<char>{hsaco.data(), hsaco.data() + hsaco.size()});
        }
#endif
        return p;
    }
}

bool Handle::HasProgram(const fs::path& program_name, const std::string& params) const
{
    return this->impl->cache.HasProgram(program_name, params);
}

void Handle::AddProgram(Program prog, const fs::path& program_name, const std::string& params) const
{
    this->impl->cache.AddProgram(prog, program_name, params);
}

void Handle::ClearProgram(const fs::path& program_name, const std::string& params) const
{
    this->impl->cache.ClearProgram(program_name, params);
}

void Handle::Finish() const
{
    this->impl->set_ctx();
#if 0
    auto start = std::chrono::system_clock::now();
    auto ev    = make_hip_event();
    hipEventRecord(ev.get(), this->GetStream());
    while(hipEventQuery(ev.get()) == hipErrorNotReady)
    {
        std::this_thread::yield();
        if((std::chrono::system_clock::now() - start) > std::chrono::seconds(60))
        {
            std::cerr << "Timeout: Handle::Finish" << std::endl;
            std::abort();
        }
    }
#else
    // hipStreamSynchronize is broken, so we use hipEventSynchronize instead
    auto ev     = make_hip_event();
    auto status = hipEventRecord(ev.get(), this->GetStream());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed hip event recording");
    status = hipEventSynchronize(ev.get());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed hip sychronization");
#endif
}
void Handle::Flush() const {}

bool Handle::IsProfilingEnabled() const { return this->impl->enable_profiling; }

void Handle::ResetKernelTime() const { this->impl->profiling_result = 0.0; }
void Handle::AccumKernelTime(float curr_time) const { this->impl->profiling_result += curr_time; }

std::size_t Handle::GetLocalMemorySize() const
{
    int result;
    auto status = hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetGlobalMemorySize() const
{
    size_t result;
    auto status = hipDeviceTotalMem(&result, this->impl->device);

    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetL2CacheSize() const
{
    int result  = 0;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeL2CacheSize, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetClockRateKhz() const
{
    int result  = 0;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeClockRate, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetMaxComputeUnits() const
{
    const std::size_t num_cu = env::value(MIOPEN_DEVICE_CU);
    if(num_cu > 0)
        return num_cu;

    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetImage3dMaxWidth() const
{
    int result;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeMaxGridDimX, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetWavefrontWidth() const
{
    hipDeviceProp_t props{};
    (void)hipGetDeviceProperties(&props, this->impl->device);
    auto result = static_cast<size_t>(props.warpSize);
    return result;
}

// No HIP API that could return maximum memory allocation size
// for a single object.
std::size_t Handle::GetMaxMemoryAllocSize() const
{
    if(m_MaxMemoryAllocSizeCached == 0)
    {
        size_t free, total;
        auto status = hip_mem_get_info_wrapper(&free, &total);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
        m_MaxMemoryAllocSizeCached = floor(total * 0.85);
    }

    return m_MaxMemoryAllocSizeCached;
}

bool Handle::CooperativeLaunchSupported() const
{
    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeCooperativeLaunch, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result == 1;
}

std::string Handle::GetDeviceNameImpl() const { return this->impl->get_device_name(); }

std::string Handle::GetDeviceName() const { return this->GetTargetProperties().Name(); }

const TargetProperties& Handle::GetTargetProperties() const
{
    return this->impl->target_properties;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    os << "stream: " << GetStream() << ", device_id: " << this->impl->device;
    return os;
}

namespace {

enum class SubBufferCheck
{
    None,
    Front,
    Back,
};

SubBufferCheck GetSubBufferCheck()
{
    const auto check = env::value(MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS);

    switch(check)
    {
    case 1: return SubBufferCheck::Front;
    case 2: return SubBufferCheck::Back;
    case 0:
    default: return SubBufferCheck::None;
    }
}

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<void*, void*> SubBuffersToMemMap;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
std::mutex SubBuffersToMemMapMutex;

void* subBufferPageAlignMalloc(size_t size, bool alignLeft)
{
    constexpr size_t maxPadding = 2ULL * 1024 * 1024 - 1;

    auto roundUpToPageAlignment = [&](size_t bytes) { return (bytes + maxPadding) & ~maxPadding; };

    const auto totalSize = roundUpToPageAlignment(size);
    void* mem            = nullptr;
    auto status          = hipMalloc(&mem, totalSize);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "hipMalloc failed " + std::to_string(size));

    if(alignLeft)
    {
        MIOPEN_LOG_T("hipMalloc left-align " << size << " at " << std::hex << mem << " Ok");
        return mem;
    }

    void* subBuffer = static_cast<char*>(mem) + totalSize - size;
    MIOPEN_LOG_T("hipMalloc right-align " << size << " at " << std::hex << subBuffer << " Ok");

    std::lock_guard<std::mutex> lck{SubBuffersToMemMapMutex};
    SubBuffersToMemMap[subBuffer] = mem;

    return subBuffer;
}

struct right_aligned_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        std::lock_guard<std::mutex> lck{SubBuffersToMemMapMutex};
        void* mem   = SubBuffersToMemMap[x];
        auto status = hipFree(mem);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status,
                                    "hipFree on right-aligned memory failed at " + to_string(mem));
        MIOPEN_LOG_T("hipFree (right-aligned) at " << std::hex << mem << " Ok");
    }
};

} // namespace

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size) const
{
    const auto check = GetSubBufferCheck();
    switch(check)
    {
    case SubBufferCheck::None: {
        auto cdata = reinterpret_cast<char*>(data);
        return {cdata + offset, null_deleter{}};
    }
    case SubBufferCheck::Front: {
        void* mem = subBufferPageAlignMalloc(size, true /* left-align */);
        return {mem, hipFree};
    }
    case SubBufferCheck::Back: {
        void* mem = subBufferPageAlignMalloc(size, false /* right-align */);
        return {mem, right_aligned_deleter{}};
    }
    }
}

shared<ConstData_t>
Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t size) const
{
    const auto check = GetSubBufferCheck();
    switch(check)
    {
    case SubBufferCheck::None: {
        auto cdata = reinterpret_cast<const char*>(data);
        return {cdata + offset, null_deleter{}};
    }
    case SubBufferCheck::Front: {
        void* mem = subBufferPageAlignMalloc(size, true /* left-align */);
        return {mem, hipFree};
    }
    case SubBufferCheck::Back: {
        void* mem = subBufferPageAlignMalloc(size, false /* right-align */);
        return {mem, right_aligned_deleter{}};
    }
    }
}

#if MIOPEN_USE_ROCBLAS

const rocblas_handle_ptr& Handle::rhandle() const
{
    if(miopenHandle_exclusive_stream != nullptr)
        MIOPEN_THROW("rocBLAS handles are bound to the stream pool and cannot be used while an "
                     "exclusive stream is set");
    if(miopenHandle_current_stream_id == 0)
        return this->impl->rhandle_;
    // locking only if handle in multistream mode
    std::shared_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);
    return this->impl->ms_resourse_ptr->rhandle_pool.at(miopenHandle_current_stream_id - 1);
}

rocblas_handle_ptr Handle::CreateRocblasHandle(miopenAcceleratorQueue_t stream) const
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    rocblas_set_stream(result.get(), stream);
    return result;
}
#endif

#if MIOPEN_USE_HIPBLASLT
const hipblasLt_handle_ptr& Handle::HipblasLtHandle() const
{
    if(miopenHandle_exclusive_stream != nullptr)
        MIOPEN_THROW("hipBLASLt handles are bound to the stream pool and cannot be used while an "
                     "exclusive stream is set");
    if(miopenHandle_current_stream_id == 0)
        return this->impl->hip_blasLt_handle;
    // locking only if handle in multistream mode
    std::shared_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);
    return this->impl->ms_resourse_ptr->hhandle_pool.at(miopenHandle_current_stream_id - 1);
}

hipblasLt_handle_ptr Handle::CreateHipblasLtHandle() const
{
    hipblasLtHandle_t handle = nullptr;
    const auto status        = hipblasLtCreate(&handle);
    if(status != hipblasStatus_t::HIPBLAS_STATUS_SUCCESS)
    {
        MIOPEN_THROW(miopenStatusUnknownError, HipblasLtErrorMessage(status, "hipblasLtCreate"));
    }

    return hipblasLt_handle_ptr{handle};
}
#endif
} // namespace miopen

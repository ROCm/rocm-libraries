// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/handle.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>

#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#define WORKAROUND_SWDEV_448157 1

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

namespace miopen {

HipEventProfiler::HipEventProfiler(const Handle& handle_)
    : handle(handle_), start(nullptr), stop(nullptr)
{
    if(handle.IsProfilingEnabled())
    {
        start = make_hip_event();
        stop  = make_hip_event();
        (void)hipEventRecord(start.get(), handle.GetStream());
    }
}

HipEventProfiler::~HipEventProfiler()
{
    if(start)
    {
        (void)hipEventRecord(stop.get(), handle.GetStream());
        (void)hipEventSynchronize(stop.get());
        float event_time = 0.0f;
        (void)hipEventElapsedTime(&event_time, start.get(), stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(event_time);
    }
}

static std::string DimToFormattedString(const size_t* dims, size_t count)
{
    std::stringstream ss;
    ss << '{';
    for(size_t i = 0; i < count; ++i)
    {
        if(i > 0)
            ss << ", ";
        else
            ss << ' ';
        ss << dims[i];
    }
    ss << " }";
    return ss.str();
}

// Rationale and the exact failure modes are documented on the declaration in
// miopen/hipoc_kernel.hpp.
void ValidateGlobalWorkSize(const std::array<size_t, 3>& gdims, const std::string& name)
{
    constexpr size_t max_global_work_size = 1ULL << 32;

    for(size_t i = 0; i < gdims.size(); ++i)
    {
        if(gdims[i] >= max_global_work_size)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Kernel " + name + ": global work size " +
                             DimToFormattedString(gdims.data(), gdims.size()) + " exceeds the " +
                             "32-bit limit in dimension " + std::to_string(i) +
                             "; the launch would silently truncate to a wrong grid");
    }
}

size_t GetMaxWorkgroupsYZ(const std::string& gcn_arch_name)
{
    // gcnArchName may carry a feature suffix, for example "gfx1250-strict" or "gfx1201:xnack-".
    if(StartsWith(gcn_arch_name, "gfx1200") || StartsWith(gcn_arch_name, "gfx1201"))
        return size_t{1} << 16;
    if(StartsWith(gcn_arch_name, "gfx1250"))
        return (size_t{1} << 16) - 1;
    return std::numeric_limits<size_t>::max();
}

void CheckWorkgroupCount(const std::array<size_t, 3>& gdims,
                         const std::array<size_t, 3>& ldims,
                         const std::string& name,
                         size_t max_workgroups_yz)
{
    for(size_t i = 1; i < 3; ++i)
    {
        const auto workgroups = (gdims[i] + ldims[i] - 1) / ldims[i];
        if(workgroups > max_workgroups_yz)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Kernel " + name + ": " + std::to_string(workgroups) +
                             " workgroups in dimension " + std::to_string(i) +
                             " exceed the limit of " + std::to_string(max_workgroups_yz) +
                             " of this architecture; the launch would silently return wrong "
                             "results");
    }
}

namespace {

/// Maximum Y/Z workgroup count of the device that runs \p stream. Looked up once per device.
/// The stream, not the creating handle's target, decides: an invoker may be run on a stream of
/// another device than the one its kernel was built for.
size_t GetStreamMaxWorkgroupsYZ(hipStream_t stream)
{
    static std::mutex mutex;
    static std::unordered_map<int, size_t> cache;

    int device = 0;
    if(hipStreamGetDevice(stream, &device) != hipSuccess && hipGetDevice(&device) != hipSuccess)
        return std::numeric_limits<size_t>::max();

    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(device);
    if(it != cache.end())
        return it->second;

    // A failed query says nothing about the device, so it is not cached.
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::numeric_limits<size_t>::max();

    const size_t limit = GetMaxWorkgroupsYZ(std::string{props.gcnArchName});
    cache.emplace(device, limit);
    return limit;
}

} // namespace

void ValidateLaunch(const std::array<size_t, 3>& gdims,
                    const std::array<size_t, 3>& ldims,
                    const std::string& name,
                    const std::function<size_t()>& get_max_workgroups_yz)
{
    ValidateGlobalWorkSize(gdims, name);

    // The smallest limit of any architecture: grids below it need no device lookup.
    constexpr size_t smallest_limit = (size_t{1} << 16) - 1;
    const auto workgroups = [&](size_t i) { return (gdims[i] + ldims[i] - 1) / ldims[i]; };
    if(workgroups(1) <= smallest_limit && workgroups(2) <= smallest_limit)
        return;

    CheckWorkgroupCount(gdims, ldims, name, get_max_workgroups_yz());
}

void ValidateLaunch(const std::array<size_t, 3>& gdims,
                    const std::array<size_t, 3>& ldims,
                    const std::string& name,
                    hipStream_t stream)
{
    ValidateLaunch(gdims, ldims, name, [stream] { return GetStreamMaxWorkgroupsYZ(stream); });
}

void HIPOCKernelInvoke::run(void* args, std::size_t size) const
{
    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;
    void* config[]    = {// HIP_LAUNCH_PARAM_* are macros that do horrible things
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      // NOLINTNEXTLINE cppcoreguidelines-pro-type-cstyle-cast
                      HIP_LAUNCH_PARAM_END};
       if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    ValidateLaunch(gdims, ldims, GetName(), stream);

    MIOPEN_HANDLE_LOCK

    auto status = hipExtModuleLaunchKernel(fun,
                                           gdims[0],
                                           gdims[1],
                                           gdims[2],
                                           ldims[0],
                                           ldims[1],
                                           ldims[2],
                                           0,
                                           stream,
                                           nullptr,
                                           reinterpret_cast<void**>(&config),
                                           start.get(),
                                           stop.get());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(callback)
    {
#if 0
        auto start_time = std::chrono::system_clock::now();
        while(hipEventQuery(stop.get()) == hipErrorNotReady)
        {
            std::this_thread::yield();
            if((std::chrono::system_clock::now() - start_time) > std::chrono::seconds(60))
            {
                std::cerr << "Timeout: HIPOCKernelInvoke::run" << std::endl;
                std::abort();
            }
        }
#else
        (void)hipEventSynchronize(stop.get());
#endif
        if(IsLoggingKernel())
        {
            float elapsed_time = 0.0f;
            (void)hipEventElapsedTime(&elapsed_time, start.get(), stop.get());

            const bool is_transpose = IsTransposeOrTransformKernel(GetName());
            AddKernelToJsonAccumulator(GetName(), elapsed_time, is_transpose);
        }
        callback(start.get(), stop.get());
    }
}

void HIPOCKernelInvoke::run_cooperative(void** kern_args) const
{
    hipError_t status;

    MIOPEN_LOG_I2("kernel_name = "
                  << GetName() << ", global_work_dim = " << DimToFormattedString(gdims.data(), 3)
                  << ", local_work_dim = " << DimToFormattedString(ldims.data(), 3));

    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
    {
        MIOPEN_THROW("MIOPEN_DEVICE_ARCH used, escaping launching kernel");
    }

    HipEventPtr start = nullptr;
    HipEventPtr stop  = nullptr;

    if(callback)
    {
        start = make_hip_event();
        stop  = make_hip_event();
    }

#if WORKAROUND_SWDEV_448157
    ValidateLaunch(gdims, ldims, GetName(), stream);

    // Unlike run(), this path passes the workgroup count rather than the work-item
    // count, so the division below must be exact or the remainder is silently dropped.
    if(gdims[0] % ldims[0] != 0 || gdims[1] % ldims[1] != 0 || gdims[2] % ldims[2] != 0)
        MIOPEN_THROW(miopenStatusInternalError);

    unsigned grid_dim_x = gdims[0] / ldims[0];
    unsigned grid_dim_y = gdims[1] / ldims[1];
    unsigned grid_dim_z = gdims[2] / ldims[2];

    MIOPEN_HANDLE_LOCK

    if(callback)
    {
        status = hipEventRecord(start.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }

    status = hipModuleLaunchCooperativeKernel(fun,
                                              grid_dim_x,
                                              grid_dim_y,
                                              grid_dim_z,
                                              ldims[0],
                                              ldims[1],
                                              ldims[2],
                                              0,
                                              stream,
                                              kern_args);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed to launch kernel");

    if(callback)
    {
        status = hipEventRecord(stop.get(), stream);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventRecord() failed");
    }
#else
#error "Doesn't work without workaround"
#endif // WORKAROUND_SWDEV_448157

    if(callback)
    {
        status = hipEventSynchronize(stop.get());
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "hipEventSynchronize() failed");
        if(IsLoggingKernel())
        {
            float elapsed_time = 0.0f;
            (void)hipEventElapsedTime(&elapsed_time, start.get(), stop.get());

            const bool is_transpose = IsTransposeOrTransformKernel(GetName());
            AddKernelToJsonAccumulator(GetName(), elapsed_time, is_transpose);
        }
        callback(start.get(), stop.get());
    }
}

HIPOCKernelInvoke HIPOCKernel::Invoke(hipStream_t stream,
                                      std::function<void(hipEvent_t, hipEvent_t)> callback,
                                      bool coop_launch) const
{
    return HIPOCKernelInvoke{stream, fun, ldims, gdims, name, callback, coop_launch};
}
} // namespace miopen

#pragma once

// Device queries and per-device handles.
//
// No other library file may call the HIP device functions test/device_query_check.cmake lists.
// See docs/device-contract.md.

#include "hip_util.h"

#include <hip/hip_runtime.h>

#include <mutex>
#include <string>
#include <unordered_map>

#ifdef HIPCONV_ENABLE_EXPLICIT_GEMM
#include <hipblaslt/hipblaslt.h>
#endif

namespace hipconv
{

// The device `stream` runs on; the null stream runs on the current device.
inline int stream_device(hipStream_t stream)
{
    hipDevice_t device = 0;
    HIP_CHECK(hipStreamGetDevice(stream, &device));
    return static_cast<int>(device);
}

#ifdef HIPCONV_ENABLE_EXPLICIT_GEMM

// The hipBLASLt handle for `device`, created on first use with `device` current.
//
// Never destroyed: destroying a handle during static destruction can outlive the HIP
// runtime, and the driver reclaims it at exit.
inline hipblasLtHandle_t hipblaslt_handle(int device)
{
    static std::mutex mutex;
    static auto* handles = new std::unordered_map<int, hipblasLtHandle_t>;

    const std::lock_guard<std::mutex> lock(mutex);
    if(const auto found = handles->find(device); found != handles->end())
        return found->second;

    int previous = 0;
    HIP_CHECK(hipGetDevice(&previous));
    HIP_CHECK(hipSetDevice(device));
    hipblasLtHandle_t created{};
    const hipblasStatus_t status = hipblasLtCreate(&created);
    if(status == HIPBLAS_STATUS_SUCCESS)
        handles->emplace(device, created);
    // Restore before throwing, so a failure leaves the caller's device current.
    HIP_CHECK(hipSetDevice(previous));
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw HipError(hipErrorUnknown,
                       "hipblasLtCreate failed: " + std::to_string(static_cast<int>(status)));
    return created;
}

#endif

} // namespace hipconv

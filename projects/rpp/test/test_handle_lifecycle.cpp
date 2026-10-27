// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rpp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if RPP_BACKEND_HIP
#include <hip/hip_runtime_api.h>

namespace {
// These interceptors replace only the HIP boundary. All RPP calls exercise the
// shared library. No kernels run, so one byte represents each HIP allocation.
struct HipCalls {
    bool forbidden = true;
    bool failDeviceFree = false;
    bool failPinnedFree = false;
    int total = 0;
    int deviceFrees = 0;
    int pinnedFrees = 0;
    void* device = nullptr;
    void* pinned = nullptr;
} hip;

hipError_t recordCall() {
    ++hip.total;
    return hip.forbidden ? hipErrorNotInitialized : hipSuccess;
}

hipError_t allocate(void** result, void*& allocation) {
    if (recordCall() != hipSuccess) return hipErrorNotInitialized;
    allocation = std::malloc(1);
    *result = allocation;
    return allocation ? hipSuccess : hipErrorOutOfMemory;
}

hipError_t release(void* pointer, void*& allocation, bool& fail, int& calls) {
    ++calls;
    if (recordCall() != hipSuccess) return hipErrorNotInitialized;
    if (fail) {
        fail = false;
        return hipErrorInvalidValue;
    }
    // Reject duplicate releases, including calls on an already-cleared pointer.
    if (!allocation || pointer != allocation) return hipErrorInvalidValue;
    std::free(allocation);
    allocation = nullptr;
    return hipSuccess;
}
}  // namespace

extern "C" hipError_t hipInit(unsigned int) {
    return recordCall();
}
extern "C" hipError_t hipGetDevice(int* device) {
    *device = 0;
    return recordCall();
}
extern "C" hipError_t hipMalloc(void** pointer, size_t) {
    return allocate(pointer, hip.device);
}
extern "C" hipError_t hipHostMalloc(void** pointer, size_t, unsigned int) {
    return allocate(pointer, hip.pinned);
}
extern "C" hipError_t hipFree(void* pointer) {
    return release(pointer, hip.device, hip.failDeviceFree, hip.deviceFrees);
}
extern "C" hipError_t hipHostFree(void* pointer) {
    return release(pointer, hip.pinned, hip.failPinnedFree, hip.pinnedFrees);
}
extern "C" const char* hipGetErrorString(hipError_t) {
    ++hip.total;
    return "injected HIP error";
}
// Cover the other HIP entry points used by handle_hip.cpp as well. The lifecycle
// does not need them, so an unexpected call must fail instead of reaching a GPU.
extern "C" hipError_t hipMemGetInfo(size_t*, size_t*) {
    ++hip.total;
    return hipErrorNotInitialized;
}
extern "C" hipError_t hipDeviceGetAttribute(int*, hipDeviceAttribute_t, int) {
    ++hip.total;
    return hipErrorNotInitialized;
}
extern "C" hipError_t hipDeviceTotalMem(size_t*, hipDevice_t) {
    ++hip.total;
    return hipErrorNotInitialized;
}
extern "C" hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t*, int) {
    ++hip.total;
    return hipErrorNotInitialized;
}
extern "C" hipError_t hipEventElapsedTime(float*, hipEvent_t, hipEvent_t) {
    ++hip.total;
    return hipErrorNotInitialized;
}
#endif

static bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const char* scenario = argv[1];
    if (std::strcmp(scenario, "null") == 0)
        return check(rppDestroy(nullptr) == rppStatusBadParm, "null handle rejected") ? 0 : 1;

    const bool isHost = std::strncmp(scenario, "host_", 5) == 0;
    rppHandle_t handle = nullptr;
#if RPP_BACKEND_HIP
    hip.forbidden = isHost;
#else
    if (!isHost) return 1;
#endif
    const auto backend = isHost ? RPP_HOST_BACKEND : RPP_HIP_BACKEND;
    if (!check(rppCreate(&handle, 1, 0, nullptr, backend) == rppStatusSuccess && handle,
               "create handle"))
        return 1;
    size_t batchSize = 0;
    if (!check(rppGetBatchSize(handle, &batchSize) == rppStatusSuccess && batchSize == 1,
               "handle remains usable"))
        return 1;

    if (std::strcmp(scenario, "host_invalid") == 0) {
        if (!check(rppDestroy(handle, static_cast<RppBackend>(99)) == rppStatusNotImplemented,
                   "invalid backend rejected without destroying handle"))
            return 1;
    }

#if RPP_BACKEND_HIP
    if (!isHost) {
        if (!check(hip.device && hip.pinned, "HIP allocation interceptors were exercised"))
            return 1;
        hip.failDeviceFree = std::strcmp(scenario, "hip_device_retry") == 0;
        hip.failPinnedFree = std::strcmp(scenario, "hip_pinned_retry") == 0;
        if (hip.failDeviceFree || hip.failPinnedFree) {
            const bool deviceFailure = hip.failDeviceFree;
            if (!check(rppDestroy(handle, RPP_HIP_BACKEND) != rppStatusSuccess,
                       "injected release failure is reported"))
                return 1;
            if (!check(
                    hip.pinned && (deviceFailure ? hip.device != nullptr : hip.device == nullptr),
                    "only successfully released allocations are discarded"))
                return 1;
            if (!check(rppGetBatchSize(handle, &batchSize) == rppStatusSuccess && batchSize == 1,
                       "failed destruction retains the handle"))
                return 1;
        }
    }
#endif

    rppStatus_t status;
    if (std::strstr(scenario, "default"))
        status = rppDestroy(handle);
    else if (std::strstr(scenario, "mismatch"))
        status = rppDestroy(handle, isHost ? RPP_HIP_BACKEND : RPP_HOST_BACKEND);
    else
        status = rppDestroy(handle, backend);
    if (!check(status == rppStatusSuccess, "destroy succeeds")) return 1;
#if RPP_BACKEND_HIP
    if (isHost) {
        if (!check(hip.total == 0, "HOST lifecycle must make zero HIP calls")) return 1;
    } else {
        if (!check(!hip.device && !hip.pinned, "all HIP allocations released")) return 1;
        const int expectedDeviceFrees = std::strcmp(scenario, "hip_device_retry") == 0 ? 2 : 1;
        const int expectedPinnedFrees = std::strcmp(scenario, "hip_pinned_retry") == 0 ? 2 : 1;
        if (!check(hip.deviceFrees == expectedDeviceFrees && hip.pinnedFrees == expectedPinnedFrees,
                   "retry never repeats a successful release"))
            return 1;
    }
#endif
    std::printf("PASS: %s\n", scenario);
    return 0;
}

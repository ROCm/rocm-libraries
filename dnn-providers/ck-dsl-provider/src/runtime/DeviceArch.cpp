// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceArch.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace ck_dsl_provider {

namespace {

/// Resolve the device ordinal to query: the stream's device when a
/// stream is supplied and the query succeeds, otherwise the current
/// default device. Returns nullopt only when even the default-device
/// query fails (no usable device).
std::optional<int> resolveDevice(hipStream_t stream) {
    if (stream != nullptr) {
        hipDevice_t streamDevice = 0;
        if (hipStreamGetDevice(stream, &streamDevice) == hipSuccess && streamDevice >= 0) {
            return static_cast<int>(streamDevice);
        }
        // Stream query failed -- fall through to the current device
        // rather than giving up; the current device is the right answer
        // in the common single-GPU case. Warn so that a silent
        // resolve-to-current-device is observable: in a multi-GPU
        // process where the stream is bound to a non-default device,
        // this would target the current device's arch instead.
        HIPDNN_PLUGIN_LOG_WARN(
            "detectDeviceArch: hipStreamGetDevice failed for the supplied stream; "
            "falling back to the current default device for arch detection");
    }

    int currentDevice = -1;
    if (hipGetDevice(&currentDevice) == hipSuccess && currentDevice >= 0) {
        return currentDevice;
    }
    return std::nullopt;
}

/// Process-wide memo of device ordinal -> bare gfx token. A device's
/// arch is immutable for the process lifetime, so a successful detection
/// is cached and the expensive hipGetDeviceProperties query runs at most
/// once per device. Mutex-guarded for the multi-threaded plan-finding
/// path. The two accessors share one cache via this single instance.
struct ArchCache {
    std::mutex mutex;
    std::unordered_map<int, std::string> map;
};

ArchCache& archCache() {
    static ArchCache cache;
    return cache;
}

/// Cached bare gfx token for ``device``, or nullopt on a miss.
std::optional<std::string> cachedArchForDevice(int device) {
    ArchCache& cache = archCache();
    const std::lock_guard<std::mutex> lock(cache.mutex);
    const auto it = cache.map.find(device);
    if (it != cache.map.end()) {
        return it->second;
    }
    return std::nullopt;
}

void storeArchForDevice(int device, const std::string& arch) {
    ArchCache& cache = archCache();
    const std::lock_guard<std::mutex> lock(cache.mutex);
    cache.map[device] = arch;
}

}  // namespace

std::string stripArchFeatureSuffix(std::string archName) {
    const std::string::size_type colon = archName.find(':');
    if (colon != std::string::npos) {
        archName.resize(colon);
    }
    return archName;
}

std::optional<std::string> detectDeviceArch(hipStream_t stream) {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        // No HIP device is visible at all (e.g. a host-only CI runner).
        // This is the one benign "no arch" outcome: there is no GPU to
        // mismatch, so callers treat it as "this provider cannot run
        // here" rather than a fault. Every path below runs only once a
        // device is known to exist.
        return std::nullopt;
    }

    // A device IS present from here on, so failing to read its
    // architecture is a genuine fault -- not something to paper over
    // with a default. Guessing an arch silently miscompiles (a kernel
    // built for the wrong target) or fails the later module load with a
    // confusing hipErrorNoBinaryForGpu. Throw instead, so callers fail
    // closed (isApplicable declines) or abort (buildPlan) deliberately.
    std::optional<int> device = resolveDevice(stream);
    if (!device.has_value()) {
        throw DeviceArchDetectionError(
            "detectDeviceArch: a HIP device is present (count=" + std::to_string(deviceCount) +
            ") but the active device ordinal could not be resolved (hipGetDevice failed)");
    }

    // The device's arch is immutable, so reuse a prior successful
    // detection for this ordinal instead of re-running the expensive
    // hipGetDeviceProperties query on every plan-resolution call.
    if (std::optional<std::string> cached = cachedArchForDevice(*device)) {
        return cached;
    }

    hipDeviceProp_t props{};
    const hipError_t propsErr = hipGetDeviceProperties(&props, *device);
    if (propsErr != hipSuccess) {
        throw DeviceArchDetectionError(
            "detectDeviceArch: hipGetDeviceProperties failed for device " +
            std::to_string(*device) + " (" + hipGetErrorString(propsErr) + ")");
    }

    std::string archName = props.gcnArchName;  // e.g. "gfx950:sramecc+:xnack-"
    if (archName.empty()) {
        throw DeviceArchDetectionError("detectDeviceArch: device " + std::to_string(*device) +
                                       " reported an empty gcnArchName");
    }

    // Strip the ROCm feature suffix: the DSL's known_arches() are bare
    // gfx tokens, and ArchTarget.from_gfx() rejects the suffixed form.
    std::string arch = stripArchFeatureSuffix(std::move(archName));
    storeArchForDevice(*device, arch);
    return arch;
}

}  // namespace ck_dsl_provider

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceArch.hpp"

#include <string>

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
        // in the common single-GPU case.
    }

    int currentDevice = -1;
    if (hipGetDevice(&currentDevice) == hipSuccess && currentDevice >= 0) {
        return currentDevice;
    }
    return std::nullopt;
}

}  // namespace

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

    hipDeviceProp_t props{};
    const hipError_t propsErr = hipGetDeviceProperties(&props, *device);
    if (propsErr != hipSuccess) {
        throw DeviceArchDetectionError(
            "detectDeviceArch: hipGetDeviceProperties failed for device " +
            std::to_string(*device) + " (" + hipGetErrorString(propsErr) + ")");
    }

    std::string arch = props.gcnArchName;  // e.g. "gfx950:sramecc+:xnack-"
    if (arch.empty()) {
        throw DeviceArchDetectionError("detectDeviceArch: device " + std::to_string(*device) +
                                       " reported an empty gcnArchName");
    }

    // Strip the ROCm feature suffix: the DSL's known_arches() are bare
    // gfx tokens, and ArchTarget.from_gfx() rejects the suffixed form.
    const std::string::size_type colon = arch.find(':');
    if (colon != std::string::npos) {
        arch.resize(colon);
    }
    return arch;
}

}  // namespace ck_dsl_provider

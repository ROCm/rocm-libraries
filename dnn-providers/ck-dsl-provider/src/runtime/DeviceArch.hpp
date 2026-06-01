// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace ck_dsl_provider {

/// Thrown by ``detectDeviceArch`` when a HIP device is present but its
/// architecture cannot be determined. A distinct type -- rather than a
/// shared plugin status code -- so callers can react to *this* fault
/// specifically (e.g. ``isApplicable`` logs it loudly and declines)
/// without misclassifying an unrelated failure that happens to carry the
/// same status. At the plugin C-API boundary ``tryCatch`` maps any
/// ``std::exception`` to ``HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR``, so the
/// surfaced status is correct without this type having to assert one.
class DeviceArchDetectionError : public std::runtime_error {
   public:
    explicit DeviceArchDetectionError(const std::string& message) : std::runtime_error(message) {}
};

/// Best-effort detection of the target GPU architecture as a bare gfx
/// token (e.g. ``"gfx950"``), suitable for passing to the CK DSL's
/// arch-aware entry points (``ck_dsl.core.arch.known_arches()`` keys
/// are bare gfx tokens).
///
/// Resolution order:
///   * the device backing ``stream`` (via ``hipStreamGetDevice``) when
///     ``stream`` is non-null -- so a compile targets the device the
///     kernel will actually launch on;
///   * otherwise the current default device (``hipGetDevice``).
///
/// The raw ``gcnArchName`` carries ROCm feature suffixes (e.g.
/// ``"gfx950:sramecc+:xnack-"``); this strips everything from the first
/// ``':'`` so the result matches the DSL's bare-token catalog.
///
/// Returns ``std::nullopt`` in exactly one case: no HIP device is
/// visible at all (``hipGetDeviceCount`` reports zero or fails), e.g. a
/// host-only CI runner. There is no GPU to target, so callers decline
/// gracefully -- ``isApplicable`` returns false, ``buildPlan`` aborts.
///
/// Throws ``DeviceArchDetectionError`` when a device IS present but its
/// architecture cannot be read (the device ordinal won't resolve,
/// ``hipGetDeviceProperties`` fails, or ``gcnArchName`` is empty). With a
/// GPU in hand we must get an arch; guessing a default would silently
/// miscompile for the wrong target or fail the later module load
/// confusingly, so this fails loudly instead.
std::optional<std::string> detectDeviceArch(hipStream_t stream);

}  // namespace ck_dsl_provider

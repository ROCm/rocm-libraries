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

/// Strip the ROCm feature suffix from a raw ``gcnArchName``, returning
/// the bare gfx token: ``"gfx950:sramecc+:xnack-"`` -> ``"gfx950"``,
/// ``"gfx942"`` -> ``"gfx942"``. The DSL's ``known_arches()`` keys are
/// bare tokens and ``ArchTarget.from_gfx()`` rejects the suffixed form.
/// Everything from the first ``':'`` onward is removed; an empty input
/// stays empty. Exposed (rather than kept internal) so the pure string
/// behaviour can be unit-tested without a HIP device.
std::string stripArchFeatureSuffix(std::string archName);

/// Best-effort detection of the target GPU architecture as a bare gfx
/// token (e.g. ``"gfx950"``), suitable for passing to the CK DSL's
/// arch-aware entry points (``ck_dsl.core.arch.known_arches()`` keys
/// are bare gfx tokens).
///
/// **Security invariant**: the returned token originates from the HIP
/// runtime's device-reported ``gcnArchName`` (trusted hardware), never
/// from a graph payload field or an external caller. The provider's
/// injection-free posture (the token only becomes a comgr API argument,
/// an in-memory cache key, and a log string) depends on this. If a
/// future milestone lets a caller name an arbitrary target arch, that
/// path must be re-evaluated separately -- do not route caller-supplied
/// arch strings through here.
///
/// TODO(arch-detection-consolidation): this is the fourth in-tree copy
/// of "stream -> bare gfx token". The others are
/// ``hip-kernel-provider/include/hip_kernel_provider_common/HipDeviceUtils.hpp``
/// (``getDeviceString``),
/// ``hip-kernel-provider/src/CurrentDevicePropertyProvider.hpp``, and
/// ``integration-tests/src/harness/DeviceArch.hpp``. The plugin SDK has
/// no shared device helper, so each provider rolls its own and the
/// strip/fallback/fail-policy conventions can drift. This (the only copy
/// that distinguishes nullopt-vs-throw) is the natural seed for a shared
/// ``hipdnn_plugin_sdk`` helper; consolidate when one is introduced.
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
///
/// A device's architecture is immutable for the process lifetime, so a
/// successful lookup is memoized per device ordinal: the expensive
/// ``hipGetDeviceProperties`` query runs once per device, and the
/// repeated calls down the plan-resolution path (``isApplicable`` runs
/// several times per finalize, then ``buildPlan``) reuse the cached
/// token. The cheap ordinal-resolution queries still run each call. The
/// cache is process-wide and mutex-guarded; only successful detections
/// are cached (nullopt and the throwing faults are re-evaluated).
std::optional<std::string> detectDeviceArch(hipStream_t stream);

}  // namespace ck_dsl_provider

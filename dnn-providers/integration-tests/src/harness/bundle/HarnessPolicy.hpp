// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Where the memory a variant pack points at actually lives.
///
/// Production is always DEVICE — both registration sites are. HOST exists so the
/// harness's own unit tests can drive the whole body on a machine with no GPU:
/// ITensor::rawDeviceData() hipMallocs lazily, so a HOST run makes no HIP call at
/// all. It is one field of the policy below, not a second harness.
enum class TensorPlacement
{
    HOST,
    DEVICE,
};

/// What this run does with support claims.
///
/// One ordered value rather than a report bool and an enforce bool: those two carry
/// a rule -- enforcing without querying is nothing to enforce -- and so have a fourth
/// state no caller should be able to construct. All three of these are legal, and
/// "at least observing" is a `>=` rather than a second predicate.
enum class ClaimMode : std::uint8_t
{
    OFF, ///< never open a sidecar
    REPORT, ///< query and publish; a broken claim is counted and printed, never failed
    ENFORCE, ///< query and publish; a broken claim fails the test
};

/// Everything about the run the harness needs and cannot work out for itself.
///
/// A value, not an interface. Every field is a plain answer that cannot change
/// during a test, so a struct beats a mock: a test states the environment it wants
/// by filling one in, and nothing downstream has to be told how to answer. The one
/// place TestConfig is consulted is productionPolicy() in ProductionPolicy.hpp.
struct HarnessPolicy
{
    VerificationMode mode = VerificationMode::AUTO;
    ClaimMode claims = ClaimMode::OFF;
    TensorPlacement placement = TensorPlacement::DEVICE;

    /// Full arch token as detected, e.g. "gfx942:sramecc+:xnack-". Empty when
    /// detection failed; the metadata guard treats that as "nothing to check".
    std::string arch;
    std::string platform;
    std::size_t deviceVramMb = 0;

    bool useDevice() const
    {
        return placement == TensorPlacement::DEVICE;
    }
};

} // namespace hipdnn_integration_tests::bundle

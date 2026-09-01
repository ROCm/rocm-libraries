// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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

/// Everything about the run the harness needs and cannot work out for itself.
///
/// A value, not an interface. Every field is a plain answer that cannot change
/// during a test, so a struct beats a mock: a test states the environment it wants
/// by filling one in, and nothing downstream has to be told how to answer. The one
/// place TestConfig is consulted is productionPolicy() in HarnessDependencies.hpp.
struct HarnessPolicy
{
    VerificationMode mode = VerificationMode::AUTO;
    bool enforceSupportClaims = false;
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

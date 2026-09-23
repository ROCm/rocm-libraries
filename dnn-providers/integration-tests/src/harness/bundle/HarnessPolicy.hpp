// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
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

/// What a broken claim costs this run.
///
/// Both modes read the sidecar and publish every verdict: a lane that cannot afford to
/// go red over drift still needs to see the drift, and a summary that only appears when
/// it is also a gate is a summary nobody can use to decide whether to turn the gate on.
/// The mode decides exactly one thing -- whether a broken claim fails the test.
enum class ClaimMode : std::uint8_t
{
    WARN, ///< query and publish; a broken claim is reported, not failed
    ENFORCE, ///< query and publish; a broken claim fails the test
};

/// The value spelled after --enforce-support-claims=. Only the two words are accepted:
/// anything else throws rather than falling back to a default, because a value the
/// parser does not recognise is exactly how an opt-out ends up silently enforcing.
inline bool parseEnforceClaimsValue(const std::string& value)
{
    if(value == "true")
    {
        return true;
    }
    if(value == "false")
    {
        return false;
    }
    throw std::invalid_argument("--enforce-support-claims expects 'true' or 'false', got '" + value
                                + "'.");
}

/// The command-line inputs that decide the claim mode.
///
/// Named fields rather than positional arguments: every combination is legal to
/// write, so a transposed argument would compile and quietly pick the wrong mode.
struct ClaimModeRequest
{
    /// --enforce-support-claims as typed: empty when it was not typed at all. Kept
    /// apart from "typed true" because enforcement defaults on, and only a lane that
    /// asked for it is owed an error when it cannot be delivered.
    std::optional<bool> enforce;
    bool writing = false; ///< --write-support-claims
    bool hasEngine = false; ///< --test-engine named one
};

struct ClaimModeResolution
{
    ClaimMode mode = ClaimMode::WARN;
    /// Set when the flags contradict each other; the run must not start.
    std::optional<std::string> error;
};

/// Enforcement needs something to check and something to check it against: a write
/// run skips every graph before the check is reached, and without --test-engine there
/// is no engine to hold to a claim. Inheriting the default into either case is not a
/// request to enforce, so it is dropped to WARN; typing the flag is, and is refused.
/// A wrong answer here does not crash -- it quietly disables the gate -- which is why
/// this is a pure function with every combination pinned by a test.
inline ClaimModeResolution resolveClaimMode(const ClaimModeRequest& request)
{
    const bool enforceAsked = request.enforce.value_or(false);
    if(enforceAsked && request.writing)
    {
        return {ClaimMode::WARN,
                "--write-support-claims is mutually exclusive with "
                "--enforce-support-claims.\n"};
    }
    // Silently degrading an asked-for enforcement to "enforced nothing, exit 0" is
    // the exact failure --enforce-support-claims exists to prevent.
    if(enforceAsked && !request.hasEngine)
    {
        return {ClaimMode::WARN,
                "Error: --enforce-support-claims requires --test-engine; there is no "
                "engine to\n"
                "       check sidecar claims against.\n"};
    }

    const bool enforce = request.enforce.value_or(true) && !request.writing && request.hasEngine;
    return {enforce ? ClaimMode::ENFORCE : ClaimMode::WARN, std::nullopt};
}

/// Everything about the run the harness needs and cannot work out for itself.
///
/// A value, not an interface. Every field is a plain answer that cannot change
/// during a test, so a struct beats a mock: a test states the environment it wants
/// by filling one in, and nothing downstream has to be told how to answer. The one
/// place TestConfig is consulted is productionPolicy() in ProductionPolicy.hpp.
struct HarnessPolicy
{
    VerificationMode mode = VerificationMode::AUTO;
    ClaimMode claims = ClaimMode::WARN;
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

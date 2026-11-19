// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace hipdnn_integration_tests
{

/// Where a comparison runs, for either validator kind (allclose or RMS).
///
/// A GPU reference leaves its output on the device, next to the engine's, so by
/// default the comparison runs there too. Golden data is loaded on the host and a CPU
/// reference writes to the host, so by default those compare on the host. The mismatch
/// report is built on the host whichever site made the pass/fail call.
enum class ValidationSite
{
    HOST,
    DEVICE,
};

/// Which validator the run asked for: --validator / HIPDNN_TEST_VALIDATOR.
///
///   AUTO — follow the reference (the default): see ValidationSite.
///   CPU  — always the host validators, even for a GPU reference's output.
///   GPU  — always the device validators, even for golden data or a CPU reference's
///          output; those are copied to the device first, and a device is required.
enum class ValidatorDevice
{
    AUTO,
    CPU,
    GPU,
};

/// Parse a --validator value (case-insensitive). Throws std::runtime_error on anything
/// else. Shared by the CLI flag and the env-var fallback so both accept exactly the same
/// spellings.
inline ValidatorDevice parseValidatorDevice(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if(value == "auto")
    {
        return ValidatorDevice::AUTO;
    }
    if(value == "cpu")
    {
        return ValidatorDevice::CPU;
    }
    if(value == "gpu")
    {
        return ValidatorDevice::GPU;
    }
    throw std::runtime_error("Invalid validator '" + value + "'; expected 'auto', 'cpu', or 'gpu'");
}

/// The site one comparison runs at: the requested validator, or — under AUTO — where
/// the expected values already live. The single place the two are combined.
inline ValidationSite resolveValidationSite(ValidatorDevice requested, ValidationSite referenceSite)
{
    switch(requested)
    {
    case ValidatorDevice::CPU:
        return ValidationSite::HOST;
    case ValidatorDevice::GPU:
        return ValidationSite::DEVICE;
    case ValidatorDevice::AUTO:
        return referenceSite;
    default:
        throw std::invalid_argument("resolveValidationSite: unhandled ValidatorDevice");
    }
}

} // namespace hipdnn_integration_tests

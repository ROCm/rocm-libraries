// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ProductionPolicy.hpp"

#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

ClaimMode claimMode(const TestConfig& config)
{
    return config.enforceSupportClaims() ? ClaimMode::ENFORCE : ClaimMode::WARN;
}

ClaimMode claimMode()
{
    return claimMode(TestConfig::get());
}

HarnessPolicy productionPolicy(TensorPlacement placement, const TestConfig& config)
{
    HarnessPolicy policy;
    policy.mode = config.getVerificationMode();
    policy.validator = config.getValidatorDevice();
    policy.claims = claimMode(config);
    policy.skipGraphValidation = config.skipGraphValidation();
    policy.writeSupportClaims = config.writeSupportClaims();
    policy.placement = placement;
    policy.arch = config.getCurrentArch();
    policy.platform = currentPlatform();
    policy.deviceVramMb = config.getCurrentDeviceVramMb();
    policy.benchmark = config.benchmarkSettings();
    policy.testSettings = config.testSettings();
    policy.graphCapture = config.graphCaptureSettings();
    policy.benchmarkReference = config.benchmarkReference();
    return policy;
}

HarnessPolicy productionPolicy(TensorPlacement placement)
{
    return productionPolicy(placement, TestConfig::get());
}

} // namespace hipdnn_integration_tests::bundle

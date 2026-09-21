// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ProductionPolicy.hpp"

#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

ClaimMode claimMode()
{
    if(TestConfig::get().enforceSupportClaims())
    {
        return ClaimMode::ENFORCE;
    }
    if(TestConfig::get().reportSupportClaims())
    {
        return ClaimMode::REPORT;
    }
    return ClaimMode::OFF;
}

HarnessPolicy productionPolicy(TensorPlacement placement)
{
    HarnessPolicy policy;
    policy.mode = TestConfig::get().getVerificationMode();
    policy.claims = claimMode();
    policy.placement = placement;
    policy.arch = TestConfig::get().getCurrentArch();
    policy.platform = currentPlatform();
    policy.deviceVramMb = TestConfig::get().getCurrentDeviceVramMb();
    return policy;
}

} // namespace hipdnn_integration_tests::bundle

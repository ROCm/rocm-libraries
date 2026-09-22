// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ProductionPolicy.hpp"

#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

ClaimMode claimMode(bool enforce, bool report)
{
    if(enforce)
    {
        return ClaimMode::ENFORCE;
    }
    if(report)
    {
        return ClaimMode::REPORT;
    }
    return ClaimMode::OFF;
}

ClaimMode claimMode()
{
    return claimMode(TestConfig::get().enforceSupportClaims(),
                     TestConfig::get().reportSupportClaims());
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ProductionPolicy.hpp"

#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

HarnessPolicy productionPolicy(TensorPlacement placement)
{
    HarnessPolicy policy;
    policy.mode = TestConfig::get().getVerificationMode();
    policy.enforceSupportClaims = TestConfig::get().enforceSupportClaims();
    policy.placement = placement;
    policy.arch = TestConfig::get().getCurrentArch();
    policy.platform = currentPlatform();
    policy.deviceVramMb = TestConfig::get().getCurrentDeviceVramMb();
    return policy;
}

} // namespace hipdnn_integration_tests::bundle

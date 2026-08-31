// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/HarnessDependencies.hpp"

#include "common/PlatformUtils.hpp"
#include "harness/ReferenceExecutorPool.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/FrontendGraphEngineRunner.hpp"

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

HarnessDependencies productionDependencies(TensorPlacement placement)
{
    HarnessDependencies deps;
    deps.engineRunner = std::make_shared<FrontendGraphEngineRunner>();
    deps.referenceExecutors = sharedReferenceExecutors();
    deps.claimObserver = std::make_shared<DefaultSupportClaimObserver>();
    deps.reporter = std::make_shared<GlobalVerificationReporter>();
    deps.policy = productionPolicy(placement);
    return deps;
}

} // namespace hipdnn_integration_tests::bundle

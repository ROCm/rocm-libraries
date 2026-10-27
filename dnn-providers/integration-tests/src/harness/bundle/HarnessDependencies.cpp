// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/HarnessDependencies.hpp"

#include "harness/ReferenceExecutorPool.hpp"
#include "harness/bundle/FrontendGraphEngineRunner.hpp"
#include "harness/bundle/ProductionPolicy.hpp"

namespace hipdnn_integration_tests::bundle
{

HarnessDependencies productionDependencies(TensorPlacement placement,
                                           std::shared_ptr<DeviceInputFiller> deviceFiller)
{
    HarnessDependencies deps;
    deps.engineRunner = std::make_shared<FrontendGraphEngineRunner>();
    deps.referenceExecutors = sharedReferenceExecutors();
    deps.claimObserver = std::make_shared<DefaultSupportClaimObserver>();
    deps.reporter = std::make_shared<GlobalVerificationReporter>();
    deps.policy = productionPolicy(placement);
    deps.deviceFiller = std::move(deviceFiller);
    return deps;
}

} // namespace hipdnn_integration_tests::bundle

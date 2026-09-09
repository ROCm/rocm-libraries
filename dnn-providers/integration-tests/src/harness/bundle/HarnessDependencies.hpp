// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "harness/IReferenceExecutors.hpp"
#include "harness/bundle/HarnessPolicy.hpp"
#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/ISupportClaimObserver.hpp"
#include "harness/bundle/IVerificationReporter.hpp"
#include "harness/bundle/ProductionPolicy.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Everything IntegrationBundleVerificationHarness talks to that it does not own
/// the decision for.
///
/// Four collaborators and a policy value — not a container, and not a registry.
/// They are the four things that need a GPU, a handle, a loaded engine plugin, or
/// process-wide state; everything else the harness does is a decision over values
/// and stays inside it.
struct HarnessDependencies
{
    std::shared_ptr<IGraphEngineRunner> engineRunner;
    std::shared_ptr<IReferenceExecutors> referenceExecutors;
    std::shared_ptr<ISupportClaimObserver> claimObserver;
    std::shared_ptr<IVerificationReporter> reporter;
    HarnessPolicy policy;
};

/// The real collaborators, with the process-wide reference-executor pool.
///
/// Defined in HarnessDependencies.cpp, which the unit-test binary deliberately does
/// not compile: calling this from a unit test is a link error rather than a test
/// that quietly needs a GPU.
HarnessDependencies productionDependencies(TensorPlacement placement);

} // namespace hipdnn_integration_tests::bundle

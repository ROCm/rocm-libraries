// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/ReferenceLaneVerdict.hpp"

#include <stdexcept>

#include "harness/reference-validation/ReferenceOpCoverage.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

// A lane accounts for a bundle when it puts a test under the bundle's name: one
// that validates, one that asserts a known gap, or one that skips naming why
// nothing could validate it.
bool laneAccountsFor(ReferenceLaneVerdict verdict)
{
    return verdict == ReferenceLaneVerdict::REGISTERED
           || verdict == ReferenceLaneVerdict::UNCOVERED_ON_COST;
}

} // namespace

const char* toString(ReferenceLaneVerdict verdict)
{
    switch(verdict)
    {
    case ReferenceLaneVerdict::NO_GOLDEN_OUTPUTS:
        return "carries no golden outputs";
    case ReferenceLaneVerdict::OUTSIDE_OP_SET:
        return "outside its supported-op set";
    case ReferenceLaneVerdict::EXCLUDED_ON_COST:
        return "excluded on cost (see referenceShapeIsAffordable)";
    case ReferenceLaneVerdict::UNCOVERED_ON_COST:
        return "excluded on cost, registered as a skipping test";
    case ReferenceLaneVerdict::REGISTERED:
        return "registered";
    default:
        return "unknown";
    }
}

namespace detail
{

ReferenceLaneVerdict referenceLaneVerdict(const IntegrationTestBundle& bundle,
                                          const std::string& bundleId,
                                          ReferenceExecutorType referenceType,
                                          bool gpuLaneWillRun)
{
    if(!bundle.hasGoldenOutputs)
    {
        return ReferenceLaneVerdict::NO_GOLDEN_OUTPUTS;
    }
    if(!referenceCoversGraph(referenceType, bundle.graphBuffer.data(), bundle.graphBuffer.size()))
    {
        return ReferenceLaneVerdict::OUTSIDE_OP_SET;
    }
    // Deliberate cost exclusion. It applies unconditionally: these are the shapes
    // the scalar CPU reference needs tens of minutes for -- a 4096-token GQA bundle
    // measured 19.6 min on a CI runner and blew the ctest timeout -- so running
    // them is never the right answer, whatever else is or isn't covering them.
    if(!referenceShapeIsAffordable(
           referenceType, bundleId, bundle.graphBuffer.data(), bundle.graphBuffer.size()))
    {
        return gpuLaneWillRun ? ReferenceLaneVerdict::EXCLUDED_ON_COST
                              : ReferenceLaneVerdict::UNCOVERED_ON_COST;
    }
    return ReferenceLaneVerdict::REGISTERED;
}

std::vector<size_t> bundlesNoLaneAccountsFor(const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                             const std::vector<ReferenceLaneVerdict>& gpuVerdicts)
{
    if(cpuVerdicts.size() != gpuVerdicts.size())
    {
        throw std::invalid_argument("bundlesNoLaneAccountsFor: lanes judged different bundle sets");
    }

    std::vector<size_t> unaccounted;
    for(size_t i = 0; i < cpuVerdicts.size(); ++i)
    {
        // Both lanes read golden-ness off the same loaded bundle, so either view will do.
        if(cpuVerdicts[i] == ReferenceLaneVerdict::NO_GOLDEN_OUTPUTS)
        {
            continue;
        }
        if(!laneAccountsFor(cpuVerdicts[i]) && !laneAccountsFor(gpuVerdicts[i]))
        {
            unaccounted.push_back(i);
        }
    }
    return unaccounted;
}

std::vector<FailedLoad>
    unaccountedGoldenBundleFailures(const std::vector<LoadedBundle>& bundles,
                                    const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                    const std::vector<ReferenceLaneVerdict>& gpuVerdicts)
{
    if(bundles.size() != cpuVerdicts.size())
    {
        throw std::invalid_argument(
            "unaccountedGoldenBundleFailures: verdicts do not match the bundles");
    }

    std::vector<FailedLoad> failures;
    for(const size_t index : bundlesNoLaneAccountsFor(cpuVerdicts, gpuVerdicts))
    {
        const auto& bundle = bundles[index];
        failures.push_back(FailedLoad{
            bundle.suiteName + "_Unvalidated",
            bundle.testName,
            std::string("This bundle carries golden data, but no reference lane registered a test "
                        "for it, so nothing validated it. CpuRef: ")
                + toString(cpuVerdicts[index]) + "; GpuRef: " + toString(gpuVerdicts[index])
                + ".\n  bundle: " + bundle.jsonPath.string()});
    }
    return failures;
}

} // namespace detail

} // namespace hipdnn_integration_tests::bundle

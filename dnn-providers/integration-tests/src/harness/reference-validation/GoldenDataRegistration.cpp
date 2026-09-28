// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/GoldenDataRegistration.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>

#include "harness/ReferenceExecutorPool.hpp"
#include "harness/reference-validation/BundleReferenceValidationHarness.hpp"
#include "harness/reference-validation/GoldenOutputProbe.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

// Registers a synthetic skipping test for a bundle no reference validated this
// run. Same shape as registerFailedBundleLoad(), different verdict: this is a
// declared coverage gap, not a broken bundle.
void registerUncoveredBundle(const std::string& suiteName,
                             const std::string& testName,
                             const std::string& message)
{
    ::testing::RegisterTest(
        suiteName.c_str(),
        testName.c_str(),
        nullptr,
        nullptr,
        __FILE__,
        __LINE__,
        [message]() -> ::testing::Test* { return new detail::UncoveredBundleTest(message); });
}

} // namespace

std::optional<std::vector<detail::LoadedBundle>> loadGoldenDataBundles()
{
    auto discovered = detail::discoverDataDirBundles();
    if(!discovered.has_value())
    {
        return std::nullopt;
    }

    auto& candidates = discovered->bundles;
    const size_t discoveredCount = candidates.size();
    detail::GoldenOutputProbe goldenProbe;
    candidates.erase(std::remove_if(candidates.begin(),
                                    candidates.end(),
                                    [&goldenProbe](const DiscoveredBundle& disc) {
                                        return !goldenProbe.mayCarryGoldenOutputs(disc);
                                    }),
                     candidates.end());
    const size_t skippedWithoutGolden = discoveredCount - candidates.size();

    auto bundles = detail::loadDiscoveredBundles(*discovered,
                                                 /*countFound=*/false,
                                                 /*countClaims=*/false);

    if(skippedWithoutGolden > 0)
    {
        // "loaded" rather than "with golden data": these are the bundles that passed
        // the pre-load probe, which is conservative and lets through anything it
        // cannot decide. Whether they really carry golden outputs is settled during
        // the load, and reported per lane.
        std::cerr << "Bundle discovery: " << (bundles.has_value() ? bundles->size() : 0)
                  << " bundle(s) loaded, " << skippedWithoutGolden
                  << " skipped as carrying no golden data\n";
    }

    return bundles;
}

// Registers one validation test per bundle this reference is *required* to handle
// and for which golden data exists. Both conditions are checked here rather than in
// the body precisely so the harness has no skip path: if a test exists, it must run
// and pass.
//
// Bundles that fall outside the reference's supported-op set are absent from this
// lane's suite, and the count is logged so the gap is visible. Whether another lane
// picked them up is checked across lanes by registerUnvalidatedGoldenDataFailures().
//
// Returns this lane's verdict for each bundle, in the order of `bundles`.
std::vector<ReferenceLaneVerdict>
    registerGoldenDataValidationTests(const std::vector<detail::LoadedBundle>& bundles,
                                      ReferenceExecutorType referenceType,
                                      bool gpuLaneWillRun)
{
    const char* label = BundleReferenceValidationHarness::referenceLabel(referenceType);

    std::vector<ReferenceLaneVerdict> verdicts;
    verdicts.reserve(bundles.size());
    size_t registered = 0;
    size_t noGolden = 0;
    size_t uncovered = 0;
    size_t knownGaps = 0;
    size_t tooCostly = 0;
    size_t uncoveredOnCost = 0;
    std::set<std::string> uncoveredOps;

    for(const auto& bundle : bundles)
    {
        const std::string bundleId = bundle.suiteName + "." + bundle.testName;
        const auto verdict
            = detail::referenceLaneVerdict(*bundle.bundle, bundleId, referenceType, gpuLaneWillRun);
        verdicts.push_back(verdict);

        switch(verdict)
        {
        case ReferenceLaneVerdict::NO_GOLDEN_OUTPUTS:
            // Belt and braces: loadGoldenDataBundles() already filtered on this, so a
            // bundle without golden outputs reaching here is a filter bug, not data.
            // Counted rather than assumed away so it surfaces instead of skewing the
            // registered-of-total line below.
            noGolden++;
            break;
        case ReferenceLaneVerdict::OUTSIDE_OP_SET:
            uncovered++;
            // Name the ops responsible, not just the tally. The op set is a
            // commitment (see ReferenceOpCoverage.hpp): "7 bundles excluded" says a
            // gap exists, "7 excluded: ConvolutionBwdData, Reduction" says which one
            // to close.
            for(auto& nodeType : uncoveredNodeTypes(referenceType,
                                                    bundle.bundle->graphBuffer.data(),
                                                    bundle.bundle->graphBuffer.size()))
            {
                uncoveredOps.insert(std::move(nodeType));
            }
            break;
        case ReferenceLaneVerdict::EXCLUDED_ON_COST:
            tooCostly++;
            break;
        case ReferenceLaneVerdict::UNCOVERED_ON_COST:
            // `--reference cpu`, or a device-less runner where the GPU harness
            // SKIP_IF_NO_DEVICES()s in SetUp(): nothing validates this bundle, so it
            // gets a skipping test under its own name rather than vanishing into a
            // counter.
            tooCostly++;
            uncoveredOnCost++;
            registerUncoveredBundle(
                bundle.suiteName + "_" + label,
                bundle.testName,
                std::string("Excluded from the ") + label
                    + " lane on cost (see referenceShapeIsAffordable), and no GpuRef lane "
                      "runs this session to cover it -- so nothing validated this bundle "
                      "against its golden data.\n  bundle: "
                    + bundle.jsonPath.string());
            break;
        case ReferenceLaneVerdict::REGISTERED:
            if(findKnownReferenceGap(referenceType, bundleId) != nullptr)
            {
                knownGaps++;
            }
            ::testing::RegisterTest(
                (bundle.suiteName + "_" + label).c_str(),
                bundle.testName.c_str(),
                nullptr,
                nullptr,
                __FILE__,
                __LINE__,
                [loaded = bundle.bundle,
                 path = bundle.jsonPath,
                 bundleId,
                 referenceType,
                 validator = TestConfig::get().getValidatorDevice()]() -> ::testing::Test* {
                    // Only the GPU reference — or a forced GPU validator — touches a
                    // device. Passing true for a plain CPU lane made SetUp() run
                    // SKIP_IF_NO_DEVICES() on work that reads and writes host memory, so
                    // CPU golden-data validation silently skipped on any runner without a
                    // GPU.
                    const bool requiresDevice = referenceType == ReferenceExecutorType::GPU
                                                || validator == ValidatorDevice::GPU;
                    auto* test = new BundleReferenceValidationHarness(
                        referenceType, requiresDevice, sharedReferenceExecutors(), validator);
                    test->setBundle(loaded, path, bundleId);
                    return test;
                });
            registered++;
            break;
        default:
            throw std::logic_error("registerGoldenDataValidationTests: unhandled lane verdict");
        }
    }

    // knownGaps is a subset of registered, not a third bucket: those tests still run,
    // they just assert the reference declines. Printing it separately keeps "38
    // registered" from reading as "38 validated against golden data".
    std::cerr << "Golden-data validation (" << label << "): " << registered << " of "
              << bundles.size() << " golden-bearing bundle(s) registered, " << uncovered
              << " outside this reference's supported-op set" << formatUncoveredOps(uncoveredOps);
    if(tooCostly > 0)
    {
        std::cerr << "\n       " << tooCostly
                  << " excluded as too costly for this reference (see "
                     "referenceShapeIsAffordable); ";
        if(uncoveredOnCost > 0)
        {
            std::cerr << uncoveredOnCost
                      << " of those are covered by NO reference this session (no GpuRef lane) "
                         "and are registered as skipping tests naming the gap";
        }
        else
        {
            std::cerr << "the GpuRef lane runs this session and must cover them (any it does "
                         "not fails as <bundle>_Unvalidated)";
        }
    }
    if(knownGaps > 0)
    {
        std::cerr << "\n       " << knownGaps << " of the " << registered
                  << " registered are known reference gaps (see knownReferenceGaps()): they assert "
                     "the reference declines the graph, and are NOT validated against golden data";
    }
    if(noGolden > 0)
    {
        std::cerr << "\n       " << noGolden
                  << " loaded bundle(s) turned out to carry no golden outputs: the pre-load probe "
                     "cannot always tell, and on a tree where `dvc pull` has not run every bundle "
                     "lands here";
    }
    std::cerr << "\n";

    return verdicts;
}

void registerUnvalidatedGoldenDataFailures(const std::vector<detail::LoadedBundle>& bundles,
                                           const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                           const std::vector<ReferenceLaneVerdict>& gpuVerdicts)
{
    for(const auto& failure :
        detail::unaccountedGoldenBundleFailures(bundles, cpuVerdicts, gpuVerdicts))
    {
        detail::registerFailedBundleLoad(failure.suiteName, failure.testName, failure.message);
    }
}

} // namespace hipdnn_integration_tests::bundle

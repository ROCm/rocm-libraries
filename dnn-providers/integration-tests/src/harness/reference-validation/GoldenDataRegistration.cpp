// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/GoldenDataRegistration.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "harness/ReferenceExecutorPool.hpp"
#include "harness/TestConfig.hpp"
#include "harness/reference-validation/BundleReferenceSmokeHarness.hpp"
#include "harness/reference-validation/BundleReferenceValidationHarness.hpp"
#include "harness/reference-validation/GoldenOutputProbe.hpp"

namespace hipdnn_integration_tests::bundle
{

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

void registerGoldenDataPlan(const GoldenDataPlan& plan,
                            const std::vector<detail::LoadedBundle>& bundles)
{
    const auto validator = TestConfig::get().getValidatorDevice();

    for(const auto& planned : plan.tests)
    {
        const auto& bundle = bundles.at(planned.bundleIndex);

        switch(planned.kind)
        {
        case PlannedKind::VALIDATE:
        case PlannedKind::EXPECT_DECLINE:
            ::testing::RegisterTest(
                planned.suiteName.c_str(),
                planned.testName.c_str(),
                nullptr,
                nullptr,
                __FILE__,
                __LINE__,
                [loaded = bundle.bundle,
                 path = bundle.jsonPath,
                 referenceType = planned.reference.value(),
                 expectedGap = planned.gap,
                 validator]() -> ::testing::Test* {
                    // Only the GPU reference — or a forced GPU validator — touches a
                    // device. Passing true for a plain CPU lane made SetUp() run
                    // SKIP_IF_NO_DEVICES() on work that reads and writes host memory, so
                    // CPU golden-data validation silently skipped on any runner without a
                    // GPU.
                    const bool requiresDevice = referenceType == ReferenceExecutorType::GPU
                                                || validator == ValidatorDevice::GPU;
                    auto* test = new BundleReferenceValidationHarness(
                        referenceType, requiresDevice, sharedReferenceExecutors(), validator);
                    test->setBundle(loaded, path, expectedGap);
                    return test;
                });
            break;
        case PlannedKind::SKIP:
            detail::registerSyntheticBundleTest(planned.suiteName,
                                                planned.testName,
                                                detail::SyntheticOutcome::SKIP,
                                                planned.message);
            break;
        case PlannedKind::FAIL:
            detail::registerSyntheticBundleTest(planned.suiteName,
                                                planned.testName,
                                                detail::SyntheticOutcome::FAIL,
                                                planned.message);
            break;
        default:
            throw std::logic_error("registerGoldenDataPlan: unhandled planned test kind");
        }
    }
}

size_t registerReferenceSmokeTests(ReferenceExecutorType reference)
{
    auto discovered = detail::discoverDataDirBundles();
    if(!discovered.has_value())
    {
        return 0;
    }

    const std::string suffix
        = std::string("_") + BundleReferenceValidationHarness::referenceLabel(reference) + "Smoke";
    const std::string filter = GTEST_FLAG_GET(filter);
    const size_t discoveredCount = discovered->bundles.size();
    auto& candidates = discovered->bundles;
    candidates.erase(std::remove_if(candidates.begin(),
                                    candidates.end(),
                                    [&](const DiscoveredBundle& disc) {
                                        return !gtestFilterSelects(
                                            filter, disc.suiteName + suffix + "." + disc.testName);
                                    }),
                     candidates.end());
    std::cerr << "--smoke: " << candidates.size() << " of " << discoveredCount
              << " discovered bundle(s) selected by --gtest_filter\n";
    if(candidates.empty())
    {
        return 0;
    }

    const auto bundles = detail::loadDiscoveredBundles(*discovered,
                                                       /*countFound=*/false,
                                                       /*countClaims=*/false);
    if(!bundles.has_value())
    {
        return 0;
    }

    for(const auto& bundle : *bundles)
    {
        std::optional<KnownReferenceGap> gap;
        if(const auto* entry = findKnownReferenceGap(reference, bundle.fullName()))
        {
            gap = *entry;
        }
        ::testing::RegisterTest(
            (bundle.suiteName + suffix).c_str(),
            bundle.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [loaded = bundle.bundle, path = bundle.jsonPath, reference, gap]() -> ::testing::Test* {
                auto* test = new BundleReferenceSmokeHarness(reference, sharedReferenceExecutors());
                test->setBundle(loaded, path, gap);
                return test;
            });
    }
    return bundles->size();
}

} // namespace hipdnn_integration_tests::bundle

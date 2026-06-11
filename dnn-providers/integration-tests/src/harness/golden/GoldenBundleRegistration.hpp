// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include "harness/TestConfig.hpp"
#include "harness/golden/GoldenBundleDiscovery.hpp"
#include "harness/golden/IntegrationGpuGoldenReferenceEngineValidation.hpp"
#include "harness/golden/TestCpuReferenceUsingGoldenValues.hpp"
#include "harness/golden/TestGpuReferenceUsingGoldenValues.hpp"

namespace hipdnn_integration_tests::golden
{

namespace detail
{

// Registers every discovered bundle against one runner subclass. The runner
// suffix is appended to the suite token (e.g. `..._fp32_CpuRef.typical`) so the
// three subclasses produce distinct, filterable GTest names without colliding.
// The verification-mode flag that would otherwise pick a single runner is a
// separate story (RFC 0011 §4.4); until then all three are registered.
template <typename HarnessType>
void registerBundlesForMode(const std::vector<DiscoveredBundle>& bundles,
                            const std::string& runnerSuffix)
{
    for(const auto& bundle : bundles)
    {
        auto suiteName = bundle.suiteName + "_" + runnerSuffix;

        ::testing::RegisterTest(suiteName.c_str(),
                                bundle.testName.c_str(),
                                nullptr,
                                nullptr,
                                __FILE__,
                                __LINE__,
                                [path = bundle.jsonPath]() -> ::testing::Test* {
                                    auto* test = new HarnessType();
                                    test->setBundlePath(path);
                                    return test;
                                });
    }
}

} // namespace detail

inline std::filesystem::path resolveGoldenDataDir()
{
    auto& config = TestConfig::get();
    if(config.hasGoldenDataDir())
    {
        return config.getGoldenDataDir();
    }
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
           / "../lib/golden_reference_data";
}

inline void registerGoldenBundleTests()
{
    if(!TestConfig::get().allowBundles())
    {
        return;
    }

    auto goldenDataDir = resolveGoldenDataDir();
    if(!std::filesystem::exists(goldenDataDir))
    {
        std::cerr << "Warning: --allow-bundles enabled but golden data directory "
                     "does not exist: "
                  << goldenDataDir << '\n';
        return;
    }

    std::vector<DiscoveredBundle> bundles;
    try
    {
        bundles = discoverGoldenBundles(goldenDataDir);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error during golden bundle discovery: " << e.what() << '\n';
        throw;
    }

    if(bundles.empty())
    {
        std::cerr << "Warning: --allow-bundles enabled but no golden bundles found in "
                  << goldenDataDir << '\n';
        return;
    }

    // Register all three runner subclasses against the shared bundle set. Each
    // produces a distinct suite via its runner suffix (see registerBundlesForMode).
    detail::registerBundlesForMode<TestCpuReferenceUsingGoldenValues>(bundles, "CpuRef");
    detail::registerBundlesForMode<TestGpuReferenceUsingGoldenValues>(bundles, "GpuRef");
    detail::registerBundlesForMode<IntegrationGpuGoldenReferenceEngineValidation>(bundles,
                                                                                  "Engine");

    std::cout << "Registered " << bundles.size()
              << " golden bundle(s) across CpuRef, GpuRef, and Engine runners\n";
}

} // namespace hipdnn_integration_tests::golden

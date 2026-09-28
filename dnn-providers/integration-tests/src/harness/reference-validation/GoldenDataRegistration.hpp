// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Test registration for hipdnn_golden_data_tests. Discovery and loading are shared
// with the engine binary (BundleRegistration.hpp); the per-bundle decisions live in
// ReferenceLaneVerdict.hpp and GoldenOutputProbe.hpp, where the unit tests can
// reach them. What is left here registers gtest tests, so it is linked only into
// the golden-data binary.

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/reference-validation/ReferenceLaneVerdict.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

// A GTest test body that immediately skips with a stored diagnostic message.
//
// Used for a bundle the cost gate excluded on a run where no other reference
// lane will cover it. Running it is not an option -- these are the shapes the
// scalar CPU reference needs tens of minutes for, which is what the gate exists
// to avoid -- but dropping it silently is what this whole binary is about not
// doing. A skip carrying the reason lands in the GTest and ctest reports under
// the bundle's own name, so "nothing validated this today" is a line someone can
// read, at no runtime cost.
class UncoveredBundleTest : public ::testing::Test
{
public:
    explicit UncoveredBundleTest(std::string message)
        : _message(std::move(message))
    {
    }

    // Public so a test can run the body directly, like FailedBundleLoadTest.
    void TestBody() override
    {
        GTEST_SKIP() << _message;
    }

private:
    std::string _message;
};

} // namespace detail

/// The bundles hipdnn_golden_data_tests validates: those carrying golden data.
///
/// Loaded once and handed to each reference lane, rather than rediscovered per
/// lane. Discovery plus load dominates this binary's startup, and doing it twice
/// also registered every failing-load test twice under the same name.
///
/// Bundles that cannot carry golden data (see GoldenOutputProbe) are dropped before
/// loading: they are ones this binary would load in full and then discard, and they
/// outnumber the ones it validates by roughly a hundred to one. That also stops this
/// binary reporting load failures for bundles it has no business validating; those
/// keep surfacing in the engine binary, which loads everything.
/// UNVALIDATABLE_GOLDEN_DATA is unaffected, since a bundle carrying golden blobs is
/// never dropped by the probe.
///
/// Returns nullopt when there is nothing to validate; the reason is already on
/// stderr.
std::optional<std::vector<detail::LoadedBundle>> loadGoldenDataBundles();

/// Registers the golden-data validation suite for one reference executor, and
/// returns that lane's verdict for each bundle, in the order of `bundles`.
///
/// Two harnesses, never both: verifying an engine and validating our own golden
/// data are different jobs, so they live in different binaries. This one is the
/// entry point for hipdnn_golden_data_tests, which loads no plugin and creates no
/// handle -- see the binary's main() for why that separation is structural rather
/// than a flag.
///
/// `gpuLaneWillRun` is the caller's answer to "will a GPU reference lane actually
/// execute this session" -- selected by --reference and backed by a device. It
/// decides what the CPU lane's cost exclusion means, not whether it applies.
std::vector<ReferenceLaneVerdict>
    registerGoldenDataValidationTests(const std::vector<detail::LoadedBundle>& bundles,
                                      ReferenceExecutorType referenceType,
                                      bool gpuLaneWillRun);

/// Fails every golden-bearing bundle that neither lane registered a test for.
/// Call after both lanes have registered, with the verdicts each returned.
void registerUnvalidatedGoldenDataFailures(const std::vector<detail::LoadedBundle>& bundles,
                                           const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                           const std::vector<ReferenceLaneVerdict>& gpuVerdicts);

} // namespace hipdnn_integration_tests::bundle

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Test registration for hipdnn_golden_data_tests. Discovery and loading are shared
// with the engine binary (BundleRegistration.hpp), and what to register is decided
// by planGoldenDataValidation() (GoldenDataPlan.hpp), where the unit tests can reach
// it. What is left here registers gtest tests and reaches the shared reference
// executors, so it is linked only into the golden-data binary.

#include <optional>
#include <vector>

#include "harness/bundle/BundleRegistration.hpp"
#include "harness/reference-validation/GoldenDataPlan.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The bundles hipdnn_golden_data_tests validates: those carrying golden data.
///
/// Loaded once and handed to the planner, rather than rediscovered per lane.
/// Discovery plus load dominates this binary's startup, and doing it twice also
/// registered every failing-load test twice under the same name.
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

/// Registers every test in `plan`, which must have been made from `bundles`. The one
/// place golden-data validation registers tests; failing-load tests are registered
/// earlier, by the shared loader.
void registerGoldenDataPlan(const GoldenDataPlan& plan,
                            const std::vector<detail::LoadedBundle>& bundles);

} // namespace hipdnn_integration_tests::bundle

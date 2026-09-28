// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The per-bundle decision each golden-data reference lane makes, and the
// cross-lane check that no golden-bearing bundle ends the run without a test. Pure
// functions: nothing here registers a test or reaches an executor, so the unit
// tests link all of it.

#include <cstddef>
#include <string>
#include <vector>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

// What one reference lane did with one loaded bundle. REGISTERED and
// UNCOVERED_ON_COST put a test under the bundle's name; the other three do not.
// That difference is what bundlesNoLaneAccountsFor() checks across lanes.
enum class ReferenceLaneVerdict
{
    NO_GOLDEN_OUTPUTS, ///< Nothing to validate.
    OUTSIDE_OP_SET, ///< The reference does not implement every op in the graph.
    EXCLUDED_ON_COST, ///< Too costly here; a GPU lane runs this session.
    UNCOVERED_ON_COST, ///< Too costly here and no GPU lane runs: a skipping test stands in.
    REGISTERED, ///< A validation test is registered (known-gap tests included).
};

const char* toString(ReferenceLaneVerdict verdict);

namespace detail
{

// The per-bundle decision behind registerGoldenDataValidationTests(). Split out so
// it registers nothing and reaches no executor.
//
// `gpuLaneWillRun` says whether a GPU reference lane will actually execute in this
// process -- selected by --reference *and* backed by a device. It does not change
// *whether* the cost gate excludes a shape, only what the exclusion leaves behind:
// with that lane running, the GPU lane is expected to validate the bundle (and
// bundlesNoLaneAccountsFor() fails the run if it does not); without it nothing
// validates the bundle, so a skipping test is registered in its place to say so.
ReferenceLaneVerdict referenceLaneVerdict(const IntegrationTestBundle& bundle,
                                          const std::string& bundleId,
                                          ReferenceExecutorType referenceType,
                                          bool gpuLaneWillRun);

// Indices of golden-bearing bundles that neither lane accounts for. Such a bundle
// was loaded, carries golden data, and ends the run with no test under its name
// in either lane -- outside both op sets, or dropped on cost by the CPU lane on
// the assumption that a GPU lane that does not implement it would cover it. It
// shows up only as a stderr counter, which is the silent drop this binary exists
// to prevent.
//
// Cross-lane because no single lane can see it: every bundle a lane is handed
// ends in exactly one verdict, so "this lane registered nothing" is always
// explained by that lane's own counters. Whether *some* lane covered the bundle
// is the only question with a wrong answer.
//
// Both vectors are verdicts for the same bundles, in the same order.
std::vector<size_t> bundlesNoLaneAccountsFor(const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                             const std::vector<ReferenceLaneVerdict>& gpuVerdicts);

// The failing test owed to every golden-bearing bundle no lane accounted for,
// under the bundle's own name so the tier prefix (and with it the ctest category)
// is kept. A FailedLoad, not a skip: nothing validated data we ship, and that
// must turn the run red. Split from the registration so it can be tested --
// ::testing::RegisterTest cannot run once RUN_ALL_TESTS() has started.
std::vector<FailedLoad>
    unaccountedGoldenBundleFailures(const std::vector<LoadedBundle>& bundles,
                                    const std::vector<ReferenceLaneVerdict>& cpuVerdicts,
                                    const std::vector<ReferenceLaneVerdict>& gpuVerdicts);

} // namespace detail

} // namespace hipdnn_integration_tests::bundle

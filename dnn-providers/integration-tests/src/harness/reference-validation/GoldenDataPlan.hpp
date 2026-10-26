// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// What hipdnn_golden_data_tests registers for every loaded bundle: which reference
// lanes validate it, which assert a known gap, which skip, and which bundles nothing
// covers. Decided here in one pass, as data. Nothing in this header registers a test
// or reaches an executor, so the unit tests link all of it; GoldenDataRegistration
// registers the plan this produces.

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

namespace hipdnn_integration_tests::bundle
{

/// What one reference lane makes of one bundle, from the bundle and the reference
/// alone -- nothing about the rest of the session. Checked in this order, and the
/// first that applies wins.
enum class LaneVerdict
{
    NO_GOLDEN_OUTPUTS, ///< Nothing to validate.
    OUTSIDE_OP_SET, ///< The reference is not required to implement every op in the graph.
    TOO_COSTLY, ///< The reference could, but the shape is not worth its time (CPU Sdpa only).
    KNOWN_GAP, ///< Listed in knownReferenceGaps(): the reference is expected to decline it.
    VALIDATE, ///< Validate the reference's output against the golden data.
};

const char* toString(LaneVerdict verdict);

/// `bundleId` is "<suiteName>.<testName>": the cost gate reads its tier prefix, and
/// knownReferenceGaps() is keyed on it.
LaneVerdict judgeLane(const IntegrationTestBundle& bundle,
                      const std::string& bundleId,
                      ReferenceExecutorType reference);

/// The session facts a plan depends on: which lanes --reference selected, and
/// whether the GPU one will actually execute.
struct GoldenDataSession
{
    bool cpuSelected = true;
    bool gpuSelected = true;
    /// A GPU lane with no device SKIP_IF_NO_DEVICES()s every test in SetUp(), so a
    /// selected but device-less GPU lane covers nothing.
    bool gpuHasDevice = false;

    bool gpuLaneRuns() const
    {
        return gpuSelected && gpuHasDevice;
    }
};

enum class PlannedKind
{
    VALIDATE, ///< Run the reference and compare against golden data.
    EXPECT_DECLINE, ///< Assert the reference declines the graph (a known gap).
    SKIP, ///< Stands in for a bundle nothing will validate, naming why.
    FAIL, ///< A golden-bearing bundle no lane put a test under.
};

/// One test to register, under the bundle's own name.
struct PlannedTest
{
    PlannedKind kind;
    /// Index into the bundles the plan was made from.
    size_t bundleIndex;
    /// The lane the test belongs to; nullopt for FAIL, which no lane owns.
    std::optional<ReferenceExecutorType> reference;
    std::string suiteName;
    std::string testName;
    /// Why a SKIP or FAIL test exists; empty otherwise.
    std::string message;
    /// The entry an EXPECT_DECLINE test asserts; empty otherwise.
    std::optional<KnownReferenceGap> gap;
};

/// One lane's counts, for the registration summary.
struct LaneTally
{
    ReferenceExecutorType reference;
    size_t bundles = 0;
    size_t validated = 0;
    size_t knownGaps = 0;
    size_t outsideOpSet = 0;
    size_t tooCostly = 0;
    /// The subset of tooCostly that got a SKIP test because no GPU lane runs.
    size_t skippedOnCost = 0;
    size_t noGoldenOutputs = 0;
    std::set<std::string> uncoveredOps;

    /// Tests under this lane's name that are not SKIP. Known gaps count: those
    /// tests run, they just assert the reference declines.
    size_t registered() const
    {
        return validated + knownGaps;
    }
};

struct GoldenDataPlan
{
    /// In registration order: the CpuRef lane, then the GpuRef lane, each in bundle
    /// order, then the _Unvalidated failures.
    std::vector<PlannedTest> tests;
    /// One per selected lane, CpuRef first.
    std::vector<LaneTally> lanes;
};

/// Every test hipdnn_golden_data_tests registers for `bundles` in `session`.
///
/// A bundle is accounted for when some lane puts a test under its name: VALIDATE,
/// EXPECT_DECLINE, or SKIP. With both lanes selected, a golden-bearing bundle no lane
/// accounts for gets a FAIL test, <suite>_Unvalidated. With one --reference the
/// other lane is out of scope by the caller's choice, so there is nothing to check.
///
/// A bundle that is a known gap in both lanes counts as accounted for, though
/// neither lane compares its golden data: the gap is written down, and its entries
/// fail the day the reference gains support.
GoldenDataPlan planGoldenDataValidation(const std::vector<detail::LoadedBundle>& bundles,
                                        const GoldenDataSession& session);

/// The per-lane registration summary, and a note when both lanes were selected but
/// only one registered anything.
void printPlanSummary(const GoldenDataPlan& plan, std::ostream& out);

} // namespace hipdnn_integration_tests::bundle

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/GoldenDataPlan.hpp"

#include <array>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "harness/reference-validation/BundleReferenceValidationHarness.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

// The lanes this session selected, CpuRef first. Registration order follows it.
std::vector<ReferenceExecutorType> selectedLanes(const GoldenDataSession& session)
{
    std::vector<ReferenceExecutorType> lanes;
    if(session.cpuSelected)
    {
        lanes.push_back(ReferenceExecutorType::CPU);
    }
    if(session.gpuSelected)
    {
        lanes.push_back(ReferenceExecutorType::GPU);
    }
    return lanes;
}

void printLaneSummary(const LaneTally& lane, std::ostream& out)
{
    const char* label = BundleReferenceValidationHarness::referenceLabel(lane.reference);

    // knownGaps is a subset of registered, not a third bucket: those tests still run,
    // they just assert the reference declines. Printing it separately keeps "38
    // registered" from reading as "38 validated against golden data".
    out << "Golden-data validation (" << label << "): " << lane.registered() << " of "
        << lane.bundles << " golden-bearing bundle(s) registered, " << lane.outsideOpSet
        << " outside this reference's supported-op set" << formatUncoveredOps(lane.uncoveredOps);
    if(lane.tooCostly > 0)
    {
        out << "\n       " << lane.tooCostly
            << " excluded as too costly for this reference (see "
               "referenceShapeIsAffordable); ";
        if(lane.skippedOnCost > 0)
        {
            out << lane.skippedOnCost
                << " of those are covered by NO reference this session (no GpuRef lane) "
                   "and are registered as skipping tests naming the gap";
        }
        else
        {
            out << "the GpuRef lane runs this session and must cover them (any it does "
                   "not fails as <bundle>_Unvalidated)";
        }
    }
    if(lane.knownGaps > 0)
    {
        out << "\n       " << lane.knownGaps << " of the " << lane.registered()
            << " registered are known reference gaps (see knownReferenceGaps()): they assert "
               "the reference declines the graph, and are NOT validated against golden data";
    }
    if(lane.noGoldenOutputs > 0)
    {
        out << "\n       " << lane.noGoldenOutputs
            << " loaded bundle(s) turned out to carry no golden outputs: the pre-load probe "
               "cannot always tell, and on a tree where `dvc pull` has not run every bundle "
               "lands here";
    }
    out << "\n";
}

} // namespace

const char* toString(LaneVerdict verdict)
{
    switch(verdict)
    {
    case LaneVerdict::NO_GOLDEN_OUTPUTS:
        return "carries no golden outputs";
    case LaneVerdict::OUTSIDE_OP_SET:
        return "outside its supported-op set";
    case LaneVerdict::TOO_COSTLY:
        return "excluded on cost (see referenceShapeIsAffordable)";
    case LaneVerdict::KNOWN_GAP:
        return "a known reference gap";
    case LaneVerdict::VALIDATE:
        return "validated";
    default:
        return "unknown";
    }
}

LaneVerdict judgeLane(const IntegrationTestBundle& bundle,
                      const std::string& bundleId,
                      ReferenceExecutorType reference)
{
    if(!bundle.hasGoldenOutputs)
    {
        return LaneVerdict::NO_GOLDEN_OUTPUTS;
    }
    if(!referenceCoversGraph(reference, bundle.graphBuffer.data(), bundle.graphBuffer.size()))
    {
        return LaneVerdict::OUTSIDE_OP_SET;
    }
    // Deliberate cost exclusion: these are the shapes the scalar CPU reference needs
    // tens of minutes for -- a 4096-token GQA bundle measured 19.6 min on a CI runner
    // and blew the ctest timeout -- so running them is never the right answer. What
    // the exclusion leaves behind depends on the session, which is the planner's
    // business, not this function's.
    if(!referenceShapeIsAffordable(
           reference, bundleId, bundle.graphBuffer.data(), bundle.graphBuffer.size()))
    {
        return LaneVerdict::TOO_COSTLY;
    }
    // Checked last, so a gap entry only changes the outcome for a bundle this
    // reference would otherwise validate.
    if(findKnownReferenceGap(reference, bundleId) != nullptr)
    {
        return LaneVerdict::KNOWN_GAP;
    }
    return LaneVerdict::VALIDATE;
}

GoldenDataPlan planGoldenDataValidation(const std::vector<detail::LoadedBundle>& bundles,
                                        const GoldenDataSession& session)
{
    const auto lanes = selectedLanes(session);
    const bool crossCheck = session.cpuSelected && session.gpuSelected;

    GoldenDataPlan plan;
    for(const auto reference : lanes)
    {
        LaneTally tally;
        tally.reference = reference;
        tally.bundles = bundles.size();
        plan.lanes.push_back(std::move(tally));
    }

    // Built per lane and joined at the end, so registration keeps its order: every
    // CpuRef test, then every GpuRef test, then the failures.
    std::vector<std::vector<PlannedTest>> laneTests(lanes.size());
    std::vector<PlannedTest> unvalidated;

    for(size_t index = 0; index < bundles.size(); ++index)
    {
        const auto& bundle = bundles[index];
        const std::string bundleId = bundle.fullName();

        bool accounted = false;
        std::array<LaneVerdict, 2> verdictByReference{LaneVerdict::NO_GOLDEN_OUTPUTS,
                                                      LaneVerdict::NO_GOLDEN_OUTPUTS};

        for(size_t lane = 0; lane < lanes.size(); ++lane)
        {
            const auto reference = lanes[lane];
            const std::string suiteName
                = bundle.suiteName + "_"
                  + BundleReferenceValidationHarness::referenceLabel(reference);
            auto& tally = plan.lanes[lane];

            const auto verdict = judgeLane(*bundle.bundle, bundleId, reference);
            verdictByReference.at(reference == ReferenceExecutorType::CPU ? 0 : 1) = verdict;

            switch(verdict)
            {
            case LaneVerdict::NO_GOLDEN_OUTPUTS:
                // Belt and braces: loadGoldenDataBundles() already filtered on this, so
                // a bundle without golden outputs reaching here is a filter bug, not
                // data. Counted rather than assumed away so it surfaces instead of
                // skewing the registered-of-total line.
                tally.noGoldenOutputs++;
                break;
            case LaneVerdict::OUTSIDE_OP_SET:
                tally.outsideOpSet++;
                // Name the ops responsible, not just the tally. The op set is a
                // commitment (see ReferenceOpCoverage.hpp): "7 bundles excluded" says
                // a gap exists, "7 excluded: ConvolutionBwdData, Reduction" says which
                // one to close.
                for(auto& nodeType : uncoveredNodeTypes(reference,
                                                        bundle.bundle->graphBuffer.data(),
                                                        bundle.bundle->graphBuffer.size()))
                {
                    tally.uncoveredOps.insert(std::move(nodeType));
                }
                break;
            case LaneVerdict::TOO_COSTLY:
                tally.tooCostly++;
                // With a GPU lane running, that lane owes the coverage and the
                // cross-lane check below holds it to that. Without one -- `--reference
                // cpu`, or a runner with no device -- nothing validates this bundle,
                // so a skipping test says so under its own name rather than letting
                // it vanish into a counter.
                if(!session.gpuLaneRuns())
                {
                    tally.skippedOnCost++;
                    accounted = true;
                    laneTests[lane].push_back(PlannedTest{
                        PlannedKind::SKIP,
                        index,
                        reference,
                        suiteName,
                        bundle.testName,
                        std::string("Excluded from the ")
                            + BundleReferenceValidationHarness::referenceLabel(reference)
                            + " lane on cost (see referenceShapeIsAffordable), and no GpuRef "
                              "lane runs this session to cover it -- so nothing validated this "
                              "bundle against its golden data.\n  bundle: "
                            + bundle.jsonPath.string(),
                        std::nullopt});
                }
                break;
            case LaneVerdict::KNOWN_GAP:
                tally.knownGaps++;
                accounted = true;
                // judgeLane() just found this entry, so the lookup cannot come back empty.
                laneTests[lane].push_back(PlannedTest{PlannedKind::EXPECT_DECLINE,
                                                      index,
                                                      reference,
                                                      suiteName,
                                                      bundle.testName,
                                                      {},
                                                      *findKnownReferenceGap(reference, bundleId)});
                break;
            case LaneVerdict::VALIDATE:
                tally.validated++;
                accounted = true;
                laneTests[lane].push_back(PlannedTest{PlannedKind::VALIDATE,
                                                      index,
                                                      reference,
                                                      suiteName,
                                                      bundle.testName,
                                                      {},
                                                      std::nullopt});
                break;
            default:
                throw std::logic_error("planGoldenDataValidation: unhandled lane verdict");
            }
        }

        // No single lane can see this: every bundle a lane is handed ends in exactly
        // one verdict, so "this lane registered nothing" is always explained by its
        // own counters. Whether *some* lane covered the bundle is the only question
        // with a wrong answer, and a bundle nobody covered would otherwise show up
        // only as a stderr counter -- the silent drop this binary exists to prevent.
        if(crossCheck && bundle.bundle->hasGoldenOutputs && !accounted)
        {
            unvalidated.push_back(PlannedTest{
                PlannedKind::FAIL,
                index,
                std::nullopt,
                bundle.suiteName + "_Unvalidated",
                bundle.testName,
                std::string("This bundle carries golden data, but no reference lane registered a "
                            "test for it, so nothing validated it. CpuRef: ")
                    + toString(verdictByReference[0]) + "; GpuRef: "
                    + toString(verdictByReference[1]) + ".\n  bundle: " + bundle.jsonPath.string(),
                std::nullopt});
        }
    }

    for(auto& tests : laneTests)
    {
        plan.tests.insert(plan.tests.end(),
                          std::make_move_iterator(tests.begin()),
                          std::make_move_iterator(tests.end()));
    }
    plan.tests.insert(plan.tests.end(),
                      std::make_move_iterator(unvalidated.begin()),
                      std::make_move_iterator(unvalidated.end()));
    return plan;
}

void printPlanSummary(const GoldenDataPlan& plan, std::ostream& out)
{
    for(const auto& lane : plan.lanes)
    {
        printLaneSummary(lane, out);
    }

    // Per reference, not just per run. A lane that registered nothing while its
    // sibling registered plenty is invisible in the binary-wide total. Every bundle it
    // skipped is either covered by the sibling or already a failing
    // <bundle>_Unvalidated test; this line is the human-readable counterpart, printed
    // before the run so it frames the results that follow.
    if(plan.lanes.size() == 2
       && (plan.lanes[0].registered() == 0) != (plan.lanes[1].registered() == 0))
    {
        out << "NOTE: only one reference lane registered any golden-data validation tests (CpuRef: "
            << plan.lanes[0].registered() << ", GpuRef: " << plan.lanes[1].registered()
            << "). The empty lane verified nothing this run.\n";
    }
}

} // namespace hipdnn_integration_tests::bundle

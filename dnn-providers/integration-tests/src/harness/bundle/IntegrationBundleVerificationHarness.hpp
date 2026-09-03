// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/HarnessDependencies.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/VerificationOutcome.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests::bundle
{

// OutputTensors and ExpectedTensorLookup come from OutputComparison.hpp, which owns
// the comparison this harness drives.

// detail::buildVariantPack() lives in VariantPackBuilder.hpp -- both harnesses use
// it, so it is not this one's to own.

/// Runs one bundle against the engine under test and decides what that says.
///
/// Fallback chain: golden → GPU ref → CPU ref → SKIP (RFC 0010 §4.4). Inputs are
/// read-only (shared); outputs are separate allocations per executor.
///
/// **This class has no virtual members.** Everything that needs a GPU, a handle, a
/// loaded engine plugin, or process-wide state lives behind one of the four
/// collaborators in HarnessDependencies, so a unit test constructs the real harness
/// with mocks rather than subclassing it to stub methods out. What is left here is
/// the part worth testing: which oracle to try, how far the run got, and what that
/// means for the graph's support claim.
///
/// Graph initialisation is still duplicated between this harness and
/// BundleReferenceValidationHarness; unifying the two is future work.
class IntegrationBundleVerificationHarness : public ::testing::Test
{
public:
    explicit IntegrationBundleVerificationHarness(HarnessDependencies dependencies,
                                                  std::optional<LoadedEngine> engineUnderTest = {})
        : _deps(std::move(dependencies))
        , _engineUnderTest(std::move(engineUnderTest))
    {
    }

    void setBundle(std::shared_ptr<IntegrationTestBundle> bundle,
                   std::filesystem::path path,
                   SupportClaimLocator claimLocator = {})
    {
        _bundle = std::move(bundle);
        _bundlePath = std::move(path);
        _claimLocator = std::move(claimLocator);

        if(_bundle != nullptr && _bundle->metadata.seed.has_value())
        {
            _inputFillRecipes.setGlobalSeed(static_cast<unsigned int>(*_bundle->metadata.seed));
        }

        if(_bundle != nullptr && _bundle->metadata.inputs.has_value())
        {
            _inputFillRecipes.loadFromJson(*_bundle->metadata.inputs);
        }
    }

    // Public rather than protected: the unit tests drive a real harness directly
    // instead of subclassing it, so they need to call these. GTest calls them
    // through the base class either way.
    //
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        if(_deps.policy.useDevice())
        {
            SKIP_IF_NO_DEVICES();
        }

        if(_bundle == nullptr)
        {
            GTEST_SKIP() << "No bundle set";
        }

        if(auto reason = checkTomlSkip(currentTestName()))
        {
            GTEST_SKIP() << "[arch " << _deps.policy.arch << "] " << *reason;
        }

        applyMetadataGuards();
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override
    {
        // One from_binary, one ranked query, one applicability answer. Everything
        // below takes the session as an argument, so nothing re-derives it and
        // nothing caches it on the harness.
        GraphSession session = openGraph();

        // Phase 1: read the claim facts before anything can cut the test short. This
        // has to sit above runComparison(): every mode has an early return that would
        // otherwise leave the graph's claims undecided while the run exited 0.
        const auto observation = checkSupportClaims(session);
        recordClaimCoverage(observation);

        // Phase 2: either a claim already failed, or this bundle gets run.
        VerificationOutcome outcome;
        try
        {
            if(const auto blocked = claimBlocked(observation))
            {
                // A broken claim means the engine will not take the graph, so there is
                // nothing to compare. Running anyway would execute nothing, leave the
                // NaN sentinel outputs untouched, and pile a tensor diff on the real
                // message.
                outcome = *blocked;
            }
            else
            {
                outcome = runComparison(session);

                // Kept as a live check because "the test did nothing and went green"
                // is the failure this harness exists to catch. Only asked on this
                // path: a blocked claim never reached the depth, and is already a
                // failure.
                const VerificationDepth required = bundleRequiredDepth();
                EXPECT_FALSE(outcome.status == OutcomeStatus::PASSED && outcome.depth < required)
                    << "test passed without reaching " << toString(required) << " for "
                    << _bundlePath;
            }
        }
        catch(const std::exception& e)
        {
            // This graph was already counted as queried, so a verdict that never
            // lands leaves the summary short a row and reconciles against nothing.
            // HARNESS at NOT_REACHED because a throw in here is our bug and proves
            // nothing about the engine: it must not demote the claim, and it must
            // not confirm it either.
            outcome = VerificationOutcome::failed(
                VerificationDepth::NOT_REACHED, FailureOrigin::HARNESS, e.what());
        }

        // Phase 3: one verdict, then one pass/fail/skip, both from the same outcome.
        commitClaims(observation.results, outcome);
        reportOutcome(outcome);
    }

    /// Exposed so a test can pin the fill recipes a bundle would otherwise carry in
    /// its metadata.
    InputFillRecipes& inputFillRecipes()
    {
        return _inputFillRecipes;
    }

private:
    // The one place a graph is built and the ranked list is asked for.
    GraphSession openGraph();

    void applyMetadataGuards() const;

    SupportObservation checkSupportClaims(const GraphSession& session);

    // Applies the coverage rules to the run counters, and fails this test if a
    // sidecar exists that the query somehow did not reach.
    void recordClaimCoverage(const SupportObservation& observation);

    // Publishes every verdict, promoting the engine-under-test's accepted claim by
    // what the run actually achieved. Called exactly once per test.
    void commitClaims(const std::vector<SupportResult>& results,
                      const VerificationOutcome& outcome);

    // The only place a gtest disposition is issued. Called exactly once per test,
    // last, because GTEST_SKIP() and FAIL() both return. Static because the
    // disposition is a pure function of the outcome.
    static void reportOutcome(const VerificationOutcome& outcome);

    // Records the bundle as unverifiable and yields the skip outcome for TestBody()
    // to issue.
    VerificationOutcome unverifiable(const std::string& reason,
                                     VerificationDepth reached = VerificationDepth::NOT_REACHED);

    // The single definition of "this graph's claims must be checked": a sidecar
    // exists, enforcement is on, and an engine was named to check against. Checked
    // in the same order everywhere so a harness with no injected engine never asks
    // about the sidecar.
    bool shouldEnforceClaims() const
    {
        return _engineUnderTest.has_value() && !_claimLocator.sidecarPath.empty()
               && std::filesystem::exists(_claimLocator.sidecarPath)
               && _deps.policy.enforceSupportClaims;
    }

    VerificationDepth bundleRequiredDepth() const
    {
        return _bundle != nullptr ? requiredDepth(_bundle->metadata.enforcementLevel)
                                  : VerificationDepth::VERIFIED;
    }

    enum class RefStatus
    {
        RAN,
        CAPABILITY_MISS,
        RUNTIME_ERROR,
    };
    struct RefRunResult
    {
        RefStatus status;
        std::string message;
    };

    enum class EngineStatus
    {
        RAN, ///< the engine executed the graph; `outputs` holds what it wrote
        DECLINED, ///< the engine refused the graph
        ERRORED, ///< the engine broke while compiling or executing
    };
    struct EngineRunResult
    {
        EngineStatus status = EngineStatus::DECLINED;
        /// Plans compiled before the engine stopped, so the run reached BUILDABLE
        /// even though it did not execute.
        bool plansBuilt = false;
        std::string message;
        OutputTensors outputs;
    };

    // Verifies the bundle at the depth its enforcement_level asks for. Only
    // APPLICABILITY and BUILDABLE come here; FULL takes the comparison path.
    VerificationOutcome enforceAtLevel(EnforcementLevel level, GraphSession& session);

    VerificationOutcome runComparison(GraphSession& session);
    VerificationOutcome runGoldenMode(GraphSession& session);
    VerificationOutcome runExplicitRefMode(GraphSession& session, ReferenceExecutorType type);
    VerificationOutcome runAutoMode(GraphSession& session);

    // nullopt when the inputs are ready; otherwise the outcome to return.
    std::optional<VerificationOutcome> prepareInputs();
    std::optional<VerificationOutcome> fillBundleInputs();

    OutputTensors allocateSentinelOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const;
    EngineRunResult runEngine(GraphSession& session);
    VerificationOutcome engineDidNotRun(const EngineRunResult& run) const;

    RefRunResult runReferenceCapturingOutputs(ReferenceExecutorType type,
                                              OutputTensors& refOutputs);
    void markOutputsModified(OutputTensors& outputs) const;

    VerificationOutcome compareAgainstGolden(OutputTensors& engineOutputs);
    VerificationOutcome compareOutputs(OutputTensors& engineOutputs, OutputTensors& expected);

    // Resolves tolerances, runs bundle::compareOutputs(), and turns each mismatch it
    // returns into one failure. The comparison itself owns no gtest state.
    VerificationOutcome compareAgainst(OutputTensors& engineOutputs,
                                       const ExpectedTensorLookup& expectedFor);

    // VERIFIED either way: the oracle ran and the outputs were examined. A mismatch
    // carries no message because compareAgainst() has already put one failure per
    // drifted tensor on the record — the only place in this harness where that is
    // true, and so the only caller of alreadyReportedFailure().
    static VerificationOutcome comparisonOutcome(bool allMatched)
    {
        return allMatched ? VerificationOutcome::passed(VerificationDepth::VERIFIED)
                          : VerificationOutcome::alreadyReportedFailure(VerificationDepth::VERIFIED,
                                                                        FailureOrigin::COMPARISON);
    }

    void recordRefError(const std::string& reason);
    static std::string refLabel(ReferenceExecutorType type);

    HarnessDependencies _deps;
    std::optional<LoadedEngine> _engineUnderTest;
    std::filesystem::path _bundlePath;
    SupportClaimLocator _claimLocator;
    std::shared_ptr<IntegrationTestBundle> _bundle;
    InputFillRecipes _inputFillRecipes;
};

} // namespace hipdnn_integration_tests::bundle

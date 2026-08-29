// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/VerificationOutcome.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests::bundle
{

using OutputTensors
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

namespace detail
{
std::unordered_map<int64_t, void*> buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice);
}

// Fallback chain: golden → GPU ref → CPU ref → SKIP (RFC 0010 §4.4).
// Inputs are read-only (shared); outputs are separate allocations per executor.
//
// TODO(ALMIOPEN-1969 follow-up): Unify graph-init with the non-golden harness.
class IntegrationBundleVerificationHarness : public ::testing::Test
{
public:
    explicit IntegrationBundleVerificationHarness(bool requiresDevice,
                                                  std::optional<LoadedEngine> engineUnderTest = {})
        : _requiresDevice(requiresDevice)
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

protected:
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        if(_requiresDevice)
        {
            SKIP_IF_NO_DEVICES();
        }

        if(_bundle == nullptr)
        {
            GTEST_SKIP() << "No bundle set";
        }

        if(auto reason = checkTomlSkip(currentTestName()))
        {
            GTEST_SKIP() << "[arch " << TestConfig::get().getCurrentArch() << "] " << *reason;
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
        const auto observation = adjudicateClaims(session);
        recordClaimCoverage(observation);

        // Phase 2: run as far as this bundle asks for. A broken claim means the
        // engine will not take the graph, so the comparison has nothing to run and
        // would only pile a sentinel-buffer diff on top of the real message — it is
        // reported as an engine failure without being tried.
        const std::optional<VerificationOutcome> blocked = claimBlocked(observation);
        const VerificationOutcome outcome = blocked ? *blocked : runComparison(session);

        // Kept as a live check because "the test did nothing and went green" is the
        // failure this harness exists to catch.
        const VerificationDepth required = bundleRequiredDepth();
        EXPECT_FALSE(outcome.status == OutcomeStatus::PASSED && outcome.depth < required)
            << "test passed without reaching " << toString(required) << " for " << _bundlePath;

        // Phase 3: one verdict, then one pass/fail/skip, both from the same outcome.
        commitClaims(observation.results, outcome);
        reportOutcome(outcome);
    }

    virtual void executeGraphThroughEngine(GraphSession& session,
                                           std::unordered_map<int64_t, void*>& variantPack);
    virtual void runReferenceExecutor(ReferenceExecutorType type,
                                      std::unordered_map<int64_t, void*>& variantPack);
    virtual std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType type);
    virtual VerificationMode getVerificationMode() const;
    virtual bool isEnforcingSupportClaims() const;
    virtual void applyMetadataGuards() const;

    // The one place a graph is built and the ranked list is asked for. Virtual
    // because both need a handle: the deviceless unit harnesses return a session
    // with a canned `engines` and no graph.
    virtual GraphSession openGraph();

    // Virtual so a deviceless harness can inject verdicts without writing a sidecar
    // for each one. The real implementation is a pure decision over the session.
    virtual SupportObservation adjudicateClaims(const GraphSession& session);

    // Virtual so deviceless tests can observe the non-FULL routing decision without
    // reaching a real graph. The real implementation compiles plans.
    virtual VerificationOutcome enforceAtLevel(EnforcementLevel level, GraphSession& session);

    // Protected so a stubbed enforceAtLevel() can exit the same way the real one
    // does when it cannot verify: records the bundle as unverifiable and yields the
    // skip outcome for TestBody() to issue.
    VerificationOutcome unverifiable(const std::string& reason,
                                     VerificationDepth reached = VerificationDepth::NOT_REACHED);

    InputFillRecipes& inputFillRecipes()
    {
        return _inputFillRecipes;
    }

    // The single definition of "this graph's claims must be adjudicated": a sidecar
    // exists, enforcement is on, and an engine was named to adjudicate against.
    // Checked in the same order everywhere so a deviceless harness with no injected
    // engine never reaches TestConfig.
    bool shouldEnforceClaims() const
    {
        return _engineUnderTest.has_value() && !_claimLocator.sidecarPath.empty()
               && std::filesystem::exists(_claimLocator.sidecarPath) && isEnforcingSupportClaims();
    }

private:
    // Applies the coverage rules to the process-wide counters, and fails this test
    // if a sidecar exists that the query somehow did not reach.
    void recordClaimCoverage(const SupportObservation& observation);

    // Publishes every verdict, promoting the engine-under-test's accepted claim by
    // what the run actually achieved. Called exactly once per test.
    void commitClaims(const std::vector<SupportResult>& results,
                      const VerificationOutcome& outcome);

    // The only place a gtest disposition is issued. Called exactly once per test,
    // last, because GTEST_SKIP() and FAIL() both return.
    void reportOutcome(const VerificationOutcome& outcome);

    VerificationDepth bundleRequiredDepth() const
    {
        return _bundle != nullptr ? requiredDepth(_bundle->metadata.enforcementLevel)
                                  : VerificationDepth::VERIFIED;
    }

    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    SupportClaimLocator _claimLocator;
    std::shared_ptr<IntegrationTestBundle> _bundle;
    InputFillRecipes _inputFillRecipes;
    std::optional<LoadedEngine> _engineUnderTest;

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
        DECLINED, ///< EngineNotApplicableError — the engine refused the graph
        ERRORED, ///< the executor raised a fatal assertion, already on the record
    };
    struct EngineRunResult
    {
        EngineStatus status = EngineStatus::DECLINED;
        std::string message;
        OutputTensors outputs;
    };

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
    static void markOutputsModifiedFor(OutputTensors& outputs, bool device);

    VerificationOutcome compareAgainstGolden(OutputTensors& engineOutputs);
    VerificationOutcome compareOutputs(OutputTensors& engineOutputs, OutputTensors& expected);

    // VERIFIED either way: the oracle ran and the outputs were examined. A mismatch
    // carries no message because compareOutputTensor() already printed the diff.
    static VerificationOutcome comparisonOutcome(bool allMatched)
    {
        return allMatched ? VerificationOutcome::passed(VerificationDepth::VERIFIED)
                          : VerificationOutcome::failed(
                                VerificationDepth::VERIFIED, FailureOrigin::COMPARISON, {});
    }

    template <typename ExpectedLookup>
    bool compareEach(OutputTensors& engineOutputs, ExpectedLookup expectedFor);

    bool compareOutputTensor(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             hipdnn_data_sdk::utilities::ITensor& actual,
                             float atol,
                             float rtol) const;

    void recordRefError(const std::string& reason);
    static std::string refLabel(ReferenceExecutorType type);

    static std::string
        labelFor(int64_t uid, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);
};

} // namespace hipdnn_integration_tests::bundle

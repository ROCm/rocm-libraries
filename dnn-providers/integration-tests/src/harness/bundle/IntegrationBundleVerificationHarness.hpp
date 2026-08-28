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
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/VerificationOutcome.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_frontend::graph
{
class Graph;
}

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
        // Phase 1: read the claim facts before anything can cut the test short. This
        // has to sit above runComparison(): every mode has an early return that would
        // otherwise leave the graph's claims undecided while the run exited 0.
        const auto observation = observeSupportForBundle();
        recordClaimCoverage(observation);

        // Phase 2: run as far as this bundle asks for. A broken claim means the
        // engine will not take the graph, so the comparison has nothing to run and
        // would only pile a sentinel-buffer diff on top of the real message — it is
        // reported as an engine failure without being tried.
        const std::optional<VerificationOutcome> blocked = claimBlocked(observation);
        const VerificationOutcome outcome = blocked ? *blocked : runComparison();

        // Kept as a live check because "the test did nothing and went green" is the
        // failure this harness exists to catch.
        const VerificationDepth required = bundleRequiredDepth();
        EXPECT_FALSE(outcome.status == OutcomeStatus::PASSED && outcome.depth < required)
            << "test passed without reaching " << toString(required) << " for " << _bundlePath;

        // Phase 3: one verdict, then one pass/fail/skip, both from the same outcome.
        commitClaims(observation.results, outcome);
        reportOutcome(outcome);
    }

    virtual void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack);
    virtual void runReferenceExecutor(ReferenceExecutorType type,
                                      std::unordered_map<int64_t, void*>& variantPack);
    virtual std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType type);
    virtual VerificationMode getVerificationMode() const;
    virtual bool isEnforcingSupportClaims() const;
    virtual void applyMetadataGuards() const;

    // Virtual so deviceless tests can observe the non-FULL routing decision without
    // reaching getSharedHandle(). The real implementation needs a device.
    virtual VerificationOutcome enforceAtLevel(EnforcementLevel level);

    // Virtual for the same reason: the real implementation needs a handle to build
    // the graph and ask for the ranked engine list. Deviceless harnesses return an
    // empty observation.
    virtual SupportObservation observeSupportForBundle();

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
    // Bumps the process-wide coverage counters, and fails this test if a sidecar
    // exists that the query somehow did not reach.
    void recordClaimCoverage(const SupportObservation& observation);

    // nullopt when the claims held and the comparison may run; otherwise the outcome
    // that stands in for it, carrying every failing verdict's message.
    std::optional<VerificationOutcome> claimBlocked(const SupportObservation& observation) const;

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

    // Ranked-engine query result, populated by observeSupportForBundle() and reused
    // by the executor and the enforcement rungs so one test makes one heuristic
    // query instead of two.
    struct RankedQuery
    {
        bool ran = false;
        hipdnn_frontend::ErrorCode statusCode = hipdnn_frontend::ErrorCode::OK;
        std::string statusMessage;
        std::vector<int64_t> rankedIds;
    };

    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    SupportClaimLocator _claimLocator;
    std::shared_ptr<IntegrationTestBundle> _bundle;
    InputFillRecipes _inputFillRecipes;
    std::optional<LoadedEngine> _engineUnderTest;
    RankedQuery _query;

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

    VerificationOutcome runComparison();
    VerificationOutcome runGoldenMode();
    VerificationOutcome runExplicitRefMode(ReferenceExecutorType type);
    VerificationOutcome runAutoMode();

    // Builds the bundle's graph on the shared handle. Returns the frontend's message
    // on failure, empty on success — callers choose how loud to be.
    std::string buildGraph(hipdnn_frontend::graph::Graph& graph) const;

    // One heuristic query per test. The ranked list is a property of (graph, handle),
    // so a second call fans out to every plugin again for the same answer.
    const RankedQuery& rankedEngineIds(hipdnn_frontend::graph::Graph& graph);

    // nullopt when the inputs are ready; otherwise the outcome to return.
    std::optional<VerificationOutcome> prepareInputs();
    std::optional<VerificationOutcome> fillBundleInputs();

    OutputTensors allocateSentinelOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const;
    EngineRunResult runEngine();
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

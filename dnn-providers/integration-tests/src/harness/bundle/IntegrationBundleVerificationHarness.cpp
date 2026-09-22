// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"

#include <algorithm>
#include <ostream>
#include <set>
#include <sstream>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::bundle
{

// ---- the one graph, the one query ------------------------------------------

GraphSession IntegrationBundleVerificationHarness::openGraph()
{
    if(_bundle == nullptr)
    {
        return GraphSession{};
    }
    return _deps.engineRunner->openGraph(*_bundle, _engineUnderTest);
}

void IntegrationBundleVerificationHarness::applyMetadataGuards() const
{
    if(auto reason = checkVramRequirement(_bundle->metadata, _deps.policy.deviceVramMb))
    {
        GTEST_SKIP() << *reason;
    }

    if(auto reason = checkArchCompatibility(_bundle->metadata, _deps.policy.arch))
    {
        GTEST_SKIP() << *reason;
    }
}

// ---- support claims --------------------------------------------------------

SupportObservation
    IntegrationBundleVerificationHarness::checkSupportClaims(const GraphSession& session)
{
    if(_bundle == nullptr || !shouldEnforceClaims())
    {
        return {};
    }

    if(!session.buildError.empty())
    {
        // Silent on purpose: runComparison() reports this same build failure as the
        // test's outcome, and one fault deserves one message. NOT_QUERIED rather
        // than the default NONE so the coverage rules can tell "the query was
        // impossible" from "a sidecar sat there and nothing looked at it" — the
        // second is a harness bug, this is not.
        return SupportObservation{SidecarState::NOT_QUERIED, {}};
    }

    return _deps.claimObserver->observe(session.engines,
                                        _claimLocator,
                                        *_engineUnderTest,
                                        baseArchToken(_deps.policy.arch),
                                        _deps.policy.platform);
}

void IntegrationBundleVerificationHarness::recordClaimCoverage(
    const SupportObservation& observation)
{
    const CoverageUpdate update = coverageFor(observation, shouldEnforceClaims());

    _deps.reporter->recordCoverage(update);

    if(update.missedQuery)
    {
        ADD_FAILURE() << "support claims exist for " << _bundlePath
                      << " but were never queried; enforcement would have passed "
                         "without checking them";
    }
}

// The decision is finalizeClaims(); this only publishes what it returns. results is
// only ever non-empty when an engine was injected, so there is no engine-less case
// to handle here.
void IntegrationBundleVerificationHarness::commitClaims(const std::vector<SupportResult>& results,
                                                        const VerificationOutcome& outcome)
{
    if(!_engineUnderTest.has_value())
    {
        return;
    }

    for(const auto& record :
        finalizeClaims(results, _engineUnderTest->name, outcome, bundleRequiredDepth()))
    {
        _deps.reporter->recordVerdict(record);
    }
}

// The single place a test is marked passed, failed or skipped. Everything above
// returns a value, which is what keeps the claim verdict and the test result read
// off the same facts instead of off each other.
void IntegrationBundleVerificationHarness::reportOutcome(const VerificationOutcome& outcome)
{
    switch(outcome.status)
    {
    case OutcomeStatus::PASSED:
        return;
    case OutcomeStatus::SKIPPED:
        GTEST_SKIP() << outcome.message;
    case OutcomeStatus::FAILED:
        if(!outcome.message.empty())
        {
            FAIL() << outcome.message;
        }
        // A silent return is honest only for the one producer that promises the
        // detail is already on the record. Anything else reaching here is a failure
        // with nothing to say, and a bare failure beats a green run.
        if(!outcome.alreadyReported)
        {
            FAIL() << "verification failed at " << toString(outcome.depth) << " ("
                   << toString(outcome.origin) << ") with no message";
        }
        return;
    default:
        FAIL() << "Unknown outcome status";
        return;
    }
}

// ---- top-level dispatch ----------------------------------------------------

VerificationOutcome IntegrationBundleVerificationHarness::enforceAtLevel(EnforcementLevel level,
                                                                         GraphSession& session)
{
    if(level == EnforcementLevel::FULL)
    {
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED,
            FailureOrigin::HARNESS,
            "enforceAtLevel() handles APPLICABILITY/BUILDABLE only; FULL uses the normal path");
    }

    // The rung itself needs a named engine to check applicability against; claim
    // enforcement already ran in TestBody() and is not affected by this return.
    if(!_engineUnderTest.has_value())
    {
        return unverifiable("enforcement_level requires --test-engine");
    }

    const std::string rung
        = level == EnforcementLevel::APPLICABILITY ? "applicability" : "buildable";

    // Same applicability answer the claim verdict and the executor read.
    if(!session.engines.accepted)
    {
        return unverifiable("Engine " + _engineUnderTest->name
                            + " does not support this graph (enforcement_level=" + rung + ")");
    }

    if(level == EnforcementLevel::APPLICABILITY)
    {
        return VerificationOutcome::passed(VerificationDepth::APPLICABLE);
    }

    // BUILDABLE: additionally compile plans. A rung failure is the engine's doing,
    // and says so through the outcome rather than through a bare assertion.
    const EngineOpResult built = _deps.engineRunner->buildPlans(session, _engineUnderTest);

    // A provider is allowed to decline later than the ranked list suggested. That
    // is the same answer as `!accepted` above, just arrived at later, so it lands
    // in the same place rather than being blamed on the engine as a break.
    if(built.declined)
    {
        return unverifiable("Engine " + _engineUnderTest->name
                            + " declined this graph while compiling plans (enforcement_level="
                            + rung + "): " + built.message);
    }
    if(!built.ok)
    {
        return VerificationOutcome::failed(VerificationDepth::APPLICABLE,
                                           FailureOrigin::ENGINE,
                                           "[rung=buildable] " + built.message);
    }
    return VerificationOutcome::passed(VerificationDepth::BUILDABLE);
}

VerificationOutcome IntegrationBundleVerificationHarness::runComparison(GraphSession& session)
{
    // A graph that would not load is the engine's problem, at every level, and it is
    // the reason nothing below can run. Checked once, here, so the rungs and the
    // modes can all assume a usable session.
    if(!session.buildError.empty())
    {
        return VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                           FailureOrigin::ENGINE,
                                           "from_binary failed: " + session.buildError);
    }

    if(_bundle->metadata.enforcementLevel != EnforcementLevel::FULL)
    {
        return enforceAtLevel(_bundle->metadata.enforcementLevel, session);
    }

    if(_bundle->outputTensorUids.empty())
    {
        return unverifiable("bundle has no output tensors to compare");
    }

    if(auto unavailable = prepareInputs())
    {
        return *unavailable;
    }

    switch(_deps.policy.mode)
    {
    case VerificationMode::GOLDEN:
        return runGoldenMode(session);
    case VerificationMode::GPU:
        return runExplicitRefMode(session, ReferenceExecutorType::GPU);
    case VerificationMode::CPU:
        return runExplicitRefMode(session, ReferenceExecutorType::CPU);
    case VerificationMode::AUTO:
        return runAutoMode(session);
    default:
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED, FailureOrigin::HARNESS, "Unknown verification mode");
    }
}

VerificationOutcome
    IntegrationBundleVerificationHarness::engineDidNotRun(const EngineRunResult& run) const
{
    // The rung the engine actually cleared. A bundle that only has to compile still
    // gets credit for compiling, even though what came next did not work.
    const VerificationDepth reached
        = run.plansBuilt ? VerificationDepth::BUILDABLE : VerificationDepth::NOT_REACHED;

    if(run.status == EngineStatus::DECLINED)
    {
        std::ostringstream msg;
        msg << "Engine could not execute bundle " << _bundlePath;
        if(!run.message.empty())
        {
            msg << ": " << run.message;
        }
        return VerificationOutcome::skipped(reached, msg.str());
    }

    // ERRORED: the engine broke while compiling or executing, and the runner handed
    // back the frontend's own message. The engine is at fault.
    //
    // Named unconditionally: an empty message on a FAILED outcome means "the failure
    // is already on the gtest record", which only the comparison can promise. A
    // silent green test is the exact shape this harness exists to rule out.
    std::ostringstream msg;
    msg << "Engine failed on bundle " << _bundlePath;
    if(!run.message.empty())
    {
        msg << ": " << run.message;
    }
    return VerificationOutcome::failed(reached, FailureOrigin::ENGINE, msg.str());
}

VerificationOutcome IntegrationBundleVerificationHarness::runGoldenMode(GraphSession& session)
{
    // An explicit --verification-mode=golden is a demand for a specific oracle, not
    // a preference. Skipping when that oracle is absent means the run did not do
    // what it was asked and still went green — use `auto` if a fallback chain is
    // what you want.
    if(!_bundle->hasGoldenOutputs)
    {
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED,
            FailureOrigin::HARNESS,
            "verification-mode=golden was requested but this bundle has no golden "
            "data; run `dvc pull` for it, or use --verification-mode=auto");
    }

    auto engine = runEngine(session);
    if(engine.status != EngineStatus::RAN)
    {
        return engineDidNotRun(engine);
    }
    return compareAgainstGolden(engine.outputs);
}

VerificationOutcome
    IntegrationBundleVerificationHarness::runExplicitRefMode(GraphSession& session,
                                                             ReferenceExecutorType type)
{
    auto engine = runEngine(session);
    if(engine.status != EngineStatus::RAN)
    {
        return engineDidNotRun(engine);
    }

    OutputTensors refOutputs;
    const RefRunResult result = runReferenceCapturingOutputs(type, refOutputs);
    switch(result.status)
    {
    case RefStatus::CAPABILITY_MISS:
        return unverifiable(refLabel(type) + " cannot run this op: " + result.message,
                            VerificationDepth::EXECUTED);
    case RefStatus::RUNTIME_ERROR:
        recordRefError(refLabel(type) + " errored: " + result.message);
        return VerificationOutcome::failed(VerificationDepth::EXECUTED,
                                           FailureOrigin::ORACLE,
                                           refLabel(type) + " errored (verification-mode="
                                               + refLabel(type) + "): " + result.message);
    case RefStatus::RAN:
        return compareOutputs(engine.outputs, refOutputs);
    default:
        return VerificationOutcome::failed(
            VerificationDepth::EXECUTED, FailureOrigin::HARNESS, "Unknown RefStatus");
    }
}

VerificationOutcome IntegrationBundleVerificationHarness::runAutoMode(GraphSession& session)
{
    auto engine = runEngine(session);
    if(engine.status != EngineStatus::RAN)
    {
        return engineDidNotRun(engine);
    }

    if(_bundle->hasGoldenOutputs)
    {
        return compareAgainstGolden(engine.outputs);
    }

    // GPU ref (non-final): capability miss or runtime error -> fall through.
    bool gpuRefErrored = false;
    {
        OutputTensors refOutputs;
        const RefRunResult gpu
            = runReferenceCapturingOutputs(ReferenceExecutorType::GPU, refOutputs);
        if(gpu.status == RefStatus::RAN)
        {
            return compareOutputs(engine.outputs, refOutputs);
        }
        if(gpu.status == RefStatus::RUNTIME_ERROR)
        {
            gpuRefErrored = true;
            recordRefError("GPU reference errored (auto mode, falling through to CPU): "
                           + gpu.message);
        }
    }

    // CPU ref (final): capability miss -> unverifiable; runtime error -> FAIL.
    {
        OutputTensors refOutputs;
        const RefRunResult cpu
            = runReferenceCapturingOutputs(ReferenceExecutorType::CPU, refOutputs);
        switch(cpu.status)
        {
        case RefStatus::CAPABILITY_MISS:
            return unverifiable(
                gpuRefErrored ? "no usable reference (golden absent; GPU ref errored, CPU ref "
                                "cannot run this op; see reference-error report): "
                                    + cpu.message
                              : "no reference available (golden absent; GPU and CPU ref "
                                "cannot run this op): "
                                    + cpu.message,
                VerificationDepth::EXECUTED);
        case RefStatus::RUNTIME_ERROR:
            recordRefError("CPU reference errored (auto mode, last resort): " + cpu.message);
            return VerificationOutcome::failed(VerificationDepth::EXECUTED,
                                               FailureOrigin::ORACLE,
                                               "CPU reference errored (auto mode, last resort): "
                                                   + cpu.message);
        case RefStatus::RAN:
            return compareOutputs(engine.outputs, refOutputs);
        default:
            return VerificationOutcome::failed(
                VerificationDepth::EXECUTED, FailureOrigin::HARNESS, "Unknown RefStatus");
        }
    }
}

// ---- inputs ----------------------------------------------------------------

std::optional<VerificationOutcome> IntegrationBundleVerificationHarness::prepareInputs()
{
    if(_bundle->tensors.has_value())
    {
        return std::nullopt;
    }
    return fillBundleInputs();
}

std::optional<VerificationOutcome> IntegrationBundleVerificationHarness::fillBundleInputs()
{
    const auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();
    const std::set<int64_t> outputUids(_bundle->outputTensorUids.begin(),
                                       _bundle->outputTensorUids.end());

    InputTensorMap inputs;
    std::vector<int64_t> leafInputUids;
    for(const auto& [uid, attrs] : tensorAttrMap)
    {
        if(attrs->virtual_() || outputUids.count(uid) != 0)
        {
            continue;
        }
        inputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*attrs);
        leafInputUids.push_back(uid);
    }

    auto fillResult = hipdnn_integration_tests::fillInputs(
        wrapper.getGraph(), inputs, leafInputUids, _inputFillRecipes);
    if(!fillResult.filled)
    {
        return unverifiable(fillResult.reason);
    }

    _bundle->tensors = std::move(inputs);
    return std::nullopt;
}

// ---- engine + reference runs -----------------------------------------------

OutputTensors IntegrationBundleVerificationHarness::allocateSentinelOutputs() const
{
    const auto wrapper = _bundle->graphWrapper();
    return detail::allocateSentinelOutputs(wrapper.getTensorMap(), _bundle->outputTensorUids);
}

std::unordered_map<int64_t, void*>
    IntegrationBundleVerificationHarness::buildVariantPack(OutputTensors& outputs,
                                                           bool useDevice) const
{
    const auto wrapper = _bundle->graphWrapper();
    return detail::buildVariantPack(
        *_bundle->tensors, outputs, wrapper.getTensorMap(), _bundle->outputTensorUids, useDevice);
}

IntegrationBundleVerificationHarness::EngineRunResult
    IntegrationBundleVerificationHarness::runEngine(GraphSession& session)
{
    EngineRunResult run;

    // Decided once, in openGraph(), so applicability is one fact rather than three
    // opinions. A test states what it is simulating by setting engines.accepted on
    // the session it hands back.
    if(!session.engines.accepted)
    {
        const auto summary
            = std::to_string(_bundle->outputTensorUids.size()) + " output tensor(s), "
              + std::to_string(session.engines.rankedIds.size()) + " ranked engine(s)";
        run.status = EngineStatus::DECLINED;
        run.message = _engineUnderTest.has_value()
                          ? "Engine " + _engineUnderTest->name + " does not support this graph ("
                                + summary + ")"
                          : "No engine supports this graph (" + summary + ")";
        return run;
    }

    run.outputs = allocateSentinelOutputs();
    auto variantPack = buildVariantPack(run.outputs, _deps.policy.useDevice());

    // The runner reports rather than asserts, so "the engine broke" is a value here
    // instead of a disposition read back off GTest, which cannot tell an engine
    // assertion apart from any other fatal failure in the same test.
    const EngineOpResult result
        = _deps.engineRunner->execute(session, _engineUnderTest, variantPack);

    run.plansBuilt = result.plansBuilt;

    if(result.declined)
    {
        run.status = EngineStatus::DECLINED;
        run.message = result.message;
        return run;
    }
    if(!result.ok)
    {
        run.status = EngineStatus::ERRORED;
        run.message = result.message;
        return run;
    }

    markOutputsModified(run.outputs);
    run.status = EngineStatus::RAN;
    return run;
}

IntegrationBundleVerificationHarness::RefRunResult
    IntegrationBundleVerificationHarness::runReferenceCapturingOutputs(ReferenceExecutorType type,
                                                                       OutputTensors& refOutputs)
{
    refOutputs = allocateSentinelOutputs();

    // Only an executor that asks for device pointers gets them. Handing host memory
    // to an executor that wants device memory — or the reverse — is a silent crash,
    // not an error, and the executor is the one that knows which it needs.
    bool useDevice = false;

    try
    {
        IReferenceGraphExecutor& executor = _deps.referenceExecutors->get(type);
        useDevice = _deps.policy.useDevice() && executor.requiresDeviceMemory();
        auto variantPack = buildVariantPack(refOutputs, useDevice);

        if(!executor.isApplicable(_bundle->graphBuffer.data(), _bundle->graphBuffer.size()))
        {
            return {RefStatus::CAPABILITY_MISS,
                    refLabel(type) + " is not applicable for this graph"};
        }
        executor.execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        return {RefStatus::CAPABILITY_MISS, e.what()};
    }
    catch(const std::exception& e)
    {
        return {RefStatus::RUNTIME_ERROR, e.what()};
    }

    detail::markOutputsModified(refOutputs, useDevice);
    return {RefStatus::RAN, {}};
}

void IntegrationBundleVerificationHarness::markOutputsModified(OutputTensors& outputs) const
{
    detail::markOutputsModified(outputs, _deps.policy.useDevice());
}

// ---- comparison ------------------------------------------------------------

VerificationOutcome
    IntegrationBundleVerificationHarness::compareAgainstGolden(OutputTensors& engineOutputs)
{
    return compareAgainst(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *_bundle->tensors->at(uid);
    });
}

VerificationOutcome
    IntegrationBundleVerificationHarness::compareOutputs(OutputTensors& engineOutputs,
                                                         OutputTensors& expected)
{
    return compareAgainst(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *expected.at(uid);
    });
}

VerificationOutcome
    IntegrationBundleVerificationHarness::compareAgainst(OutputTensors& engineOutputs,
                                                         const ExpectedTensorLookup& expectedFor)
{
    auto wrapper = _bundle->graphWrapper();

    const auto tomlOverride = TestConfig::get().findToleranceOverride(currentTestName());
    if(tomlOverride)
    {
        HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for " << currentTestName()
                                                                 << ": atol=" << tomlOverride->atol
                                                                 << " rtol=" << tomlOverride->rtol);
    }

    const auto toleranceFor = [&](hipdnn_flatbuffers_sdk::data_objects::DataType dataType) {
        ComparisonTolerance tolerance;
        tolerance::resolveTolerance(
            wrapper, dataType, currentTestName(), tolerance.atol, tolerance.rtol);
        return tolerance;
    };

    const auto mismatches = bundle::compareOutputs(wrapper,
                                                   _bundle->outputTensorUids,
                                                   engineOutputs,
                                                   expectedFor,
                                                   toleranceFor,
                                                   "Bundle: " + _bundlePath.string());

    // Reported one per tensor so each diff lands next to the tensor it describes;
    // the outcome carries no message because of it.
    for(const auto& mismatch : mismatches)
    {
        ADD_FAILURE() << mismatch.report;
    }

    return comparisonOutcome(mismatches.empty());
}

// ---- reporting helpers -----------------------------------------------------

VerificationOutcome IntegrationBundleVerificationHarness::unverifiable(const std::string& reason,
                                                                       VerificationDepth reached)
{
    _deps.reporter->recordUnverifiable(_bundlePath.string(), reason);
    return VerificationOutcome::skipped(
        reached, "Unverifiable: " + reason + " (" + _bundlePath.string() + ")");
}

void IntegrationBundleVerificationHarness::recordRefError(const std::string& reason)
{
    _deps.reporter->recordReferenceError(_bundlePath.string(), reason);
}

std::string IntegrationBundleVerificationHarness::refLabel(ReferenceExecutorType type)
{
    return type == ReferenceExecutorType::GPU ? "GPU reference" : "CPU reference";
}

} // namespace hipdnn_integration_tests::bundle

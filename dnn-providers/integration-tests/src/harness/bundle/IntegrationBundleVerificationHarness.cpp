// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"

#include <algorithm>
#include <ostream>
#include <set>
#include <sstream>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "common/PlatformUtils.hpp"
#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/EngineNotApplicableError.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/UnverifiableBundleReport.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::bundle
{

// ---- the one graph, the one query ------------------------------------------

GraphSession IntegrationBundleVerificationHarness::openGraph()
{
    GraphSession session;
    if(_bundle == nullptr)
    {
        return session;
    }

    auto handle = getSharedHandle();
    session.graph = std::make_unique<hipdnn_frontend::graph::Graph>();

    const std::vector<uint8_t> graphBytes(
        _bundle->graphBuffer.data(), _bundle->graphBuffer.data() + _bundle->graphBuffer.size());

    if(auto err = session.graph->from_binary(handle, graphBytes); !err.is_good())
    {
        session.buildError = err.get_message();
        return session;
    }

    // The only heuristic query this test makes. It is a pure read on the graph — the
    // executor calls create_execution_plans() on this same object afterwards, which
    // is what the unpinned path has always done.
    std::vector<int64_t> ids;
    auto status = session.graph->get_ranked_engine_ids(ids);
    session.engines.status = status.get_code();
    session.engines.statusMessage = status.get_message();
    session.engines.rankedIds = std::move(ids);
    session.engines.accepted
        = enginesAccept(session.engines.status, session.engines.rankedIds, _engineUnderTest);

    return session;
}

// ---- virtual defaults ------------------------------------------------------

void IntegrationBundleVerificationHarness::executeGraphThroughEngine(
    GraphSession& session, std::unordered_map<int64_t, void*>& variantPack)
{
    auto handle = getSharedHandle();

    ASSERT_NE(session.graph, nullptr) << "openGraph() produced no graph to execute";
    auto& graph = *session.graph;

    // Applicability was settled once, in openGraph(); runEngine() has already turned
    // a decline into a skip, so reaching here means the engine takes this graph.
    if(_engineUnderTest.has_value())
    {
        graph.set_preferred_engine_id_ext(_engineUnderTest->id);
    }

    auto result = graph.create_execution_plans();
    ASSERT_TRUE(result.is_good()) << result.get_message();
    result = graph.check_support();
    ASSERT_TRUE(result.is_good()) << result.get_message();
    result = graph.build_plans();
    ASSERT_TRUE(result.is_good()) << result.get_message();

    int64_t workspaceSize = 0;
    result = graph.get_workspace_size(workspaceSize);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_GE(workspaceSize, 0);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    result = graph.execute(handle, variantPack, workspace.get());
    ASSERT_TRUE(result.is_good()) << result.get_message();
}

void IntegrationBundleVerificationHarness::runReferenceExecutor(
    ReferenceExecutorType type, std::unordered_map<int64_t, void*>& variantPack)
{
    auto executor = makeReferenceExecutor(type);
    if(!executor->isApplicable(_bundle->graphBuffer.data(), _bundle->graphBuffer.size()))
    {
        throw ReferenceCapabilityError(refLabel(type) + " is not applicable for this graph");
    }
    executor->execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
}

std::unique_ptr<IReferenceGraphExecutor>
    IntegrationBundleVerificationHarness::makeReferenceExecutor(ReferenceExecutorType type)
{
    switch(type)
    {
    case ReferenceExecutorType::CPU:
        return std::make_unique<CpuReferenceGraphExecutorAdapter>();
    case ReferenceExecutorType::GPU:
        return std::make_unique<gpu_graph_executor::GpuReferenceGraphExecutor>();
    default:
        throw std::runtime_error("Unknown reference executor type");
    }
}

// ---- top-level dispatch ----------------------------------------------------

VerificationMode IntegrationBundleVerificationHarness::getVerificationMode() const
{
    return TestConfig::get().getVerificationMode();
}

bool IntegrationBundleVerificationHarness::isEnforcingSupportClaims() const
{
    return TestConfig::get().enforceSupportClaims();
}

// ---- support claims --------------------------------------------------------

SupportObservation
    IntegrationBundleVerificationHarness::adjudicateClaims(const GraphSession& session)
{
    if(_bundle == nullptr || !shouldEnforceClaims())
    {
        return {};
    }

    if(!session.buildError.empty())
    {
        // ADD_FAILURE rather than ASSERT: this runs before runComparison(), which
        // reports the same build failure as an outcome; asserting here would hide it.
        ADD_FAILURE() << "from_binary failed: " << session.buildError;
        return {};
    }

    return observeSupport(session.engines,
                          _claimLocator,
                          *_engineUnderTest,
                          baseArchToken(TestConfig::get().getCurrentArch()),
                          currentPlatform());
}

void IntegrationBundleVerificationHarness::recordClaimCoverage(
    const SupportObservation& observation)
{
    const CoverageUpdate update = coverageFor(observation, shouldEnforceClaims());

    if(update.queried)
    {
        supportClaimCoverage().graphsQueried++;
    }
    if(update.noApplicableClaim)
    {
        supportClaimCoverage().graphsWithNoApplicableClaim++;
    }
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
        SupportClaimVerdicts::get().record(record);
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
        // An empty message means the failure is already on the record with more
        // detail than this could add — per-tensor diffs, or an ASSERT inside the
        // executor. Restating it would double-report one defect.
        if(!outcome.message.empty())
        {
            FAIL() << outcome.message;
        }
        return;
    default:
        FAIL() << "Unknown outcome status";
        return;
    }
}

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
    auto& graph = *session.graph;
    graph.set_preferred_engine_id_ext(_engineUnderTest->id);

    const auto rungFailure = [](const hipdnn_frontend::Error& error) {
        return VerificationOutcome::failed(VerificationDepth::APPLICABLE,
                                           FailureOrigin::ENGINE,
                                           "[rung=buildable] " + error.get_message());
    };

    if(auto result = graph.create_execution_plans(); !result.is_good())
    {
        return rungFailure(result);
    }
    if(auto result = graph.check_support(); !result.is_good())
    {
        return rungFailure(result);
    }
    if(auto result = graph.build_plans(); !result.is_good())
    {
        return rungFailure(result);
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

    switch(getVerificationMode())
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
    if(run.status == EngineStatus::DECLINED)
    {
        std::ostringstream msg;
        msg << "Engine could not execute bundle " << _bundlePath;
        if(!run.message.empty())
        {
            msg << ": " << run.message;
        }
        return VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, msg.str());
    }

    // ERRORED: an ASSERT inside the executor is already on the gtest record, with
    // the frontend's own message. Nothing to add, but the engine is at fault.
    return VerificationOutcome::failed(VerificationDepth::NOT_REACHED, FailureOrigin::ENGINE, {});
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

// Sentinel-filled (NaN) so unwritten outputs are caught by allClose.
namespace detail
{
std::unordered_map<int64_t, void*> buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice)
{
    std::unordered_map<int64_t, void*> variantPack;
    const std::set<int64_t> outputUids(outputTensorUids.begin(), outputTensorUids.end());

    for(auto& [uid, tensor] : inputs)
    {
        if(outputUids.count(uid) != 0)
        {
            continue;
        }

        const auto attrIt = tensorAttributes.find(uid);
        const bool isRuntimePassByValue
            = attrIt != tensorAttributes.end() && attrIt->second->is_runtime_pass_by_value();
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, isRuntimePassByValue);
    }

    for(auto& [uid, tensor] : outputs)
    {
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, /*isRuntimePassByValue=*/false);
    }

    return variantPack;
}
}

OutputTensors IntegrationBundleVerificationHarness::allocateSentinelOutputs() const
{
    const auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    OutputTensors outputs;
    for(const int64_t uid : _bundle->outputTensorUids)
    {
        outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttrMap.at(uid));
        outputs[uid]->fillWithSentinelValue();
    }
    return outputs;
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

    // Decided once, in openGraph(). Asking the executor to work this out again is
    // what used to make one fact three different pieces of code. A harness that
    // stubs the executor states what it is simulating by setting engines.accepted.
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
    auto variantPack = buildVariantPack(run.outputs, /*useDevice=*/_requiresDevice);

    try
    {
        executeGraphThroughEngine(session, variantPack);
    }
    catch(const EngineNotApplicableError& e)
    {
        // Still caught: a stubbed executor may raise it, and a provider is free to
        // decline later than the ranked list suggested.
        run.status = EngineStatus::DECLINED;
        run.message = e.what();
        return run;
    }

    // The executor is allowed to ASSERT, which returns from it without unwinding.
    // This is the one place gtest result state is read, and it is read once: an
    // assertion there means the engine failed to build or execute the graph.
    if(::testing::Test::HasFatalFailure())
    {
        run.status = EngineStatus::ERRORED;
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
    const bool useDevice = _requiresDevice && (type == ReferenceExecutorType::GPU);
    auto variantPack = buildVariantPack(refOutputs, useDevice);

    try
    {
        runReferenceExecutor(type, variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        return {RefStatus::CAPABILITY_MISS, e.what()};
    }
    catch(const std::exception& e)
    {
        return {RefStatus::RUNTIME_ERROR, e.what()};
    }

    markOutputsModifiedFor(refOutputs, useDevice);
    return {RefStatus::RAN, {}};
}

void IntegrationBundleVerificationHarness::markOutputsModified(OutputTensors& outputs) const
{
    markOutputsModifiedFor(outputs, _requiresDevice);
}

void IntegrationBundleVerificationHarness::markOutputsModifiedFor(OutputTensors& outputs,
                                                                  bool device)
{
    for(auto& [uid, tensor] : outputs)
    {
        if(device)
        {
            tensor->markDeviceModified();
        }
        else
        {
            tensor->markHostModified();
        }
    }
}

// ---- comparison ------------------------------------------------------------

VerificationOutcome
    IntegrationBundleVerificationHarness::compareAgainstGolden(OutputTensors& engineOutputs)
{
    return comparisonOutcome(
        compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
            return *_bundle->tensors->at(uid);
        }));
}

VerificationOutcome
    IntegrationBundleVerificationHarness::compareOutputs(OutputTensors& engineOutputs,
                                                         OutputTensors& expected)
{
    return comparisonOutcome(
        compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
            return *expected.at(uid);
        }));
}

// Every output tensor is compared even after the first mismatch: one failing test
// should name every tensor that drifted, not just the lowest uid.
template <typename ExpectedLookup>
bool IntegrationBundleVerificationHarness::compareEach(OutputTensors& engineOutputs,
                                                       ExpectedLookup expectedFor)
{
    auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    const auto tomlOverride = TestConfig::get().findToleranceOverride(currentTestName());
    if(tomlOverride)
    {
        HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for " << currentTestName()
                                                                 << ": atol=" << tomlOverride->atol
                                                                 << " rtol=" << tomlOverride->rtol);
    }

    bool allMatched = true;
    for(const int64_t uid : _bundle->outputTensorUids)
    {
        auto& actualTensor = *engineOutputs.at(uid);
        auto& expectedTensor = expectedFor(uid);

        auto* attrs = tensorAttrMap.at(uid);
        const auto dataType = attrs->data_type();

        float atol = 0.0f;
        float rtol = 0.0f;
        tolerance::resolveTolerance(wrapper, dataType, currentTestName(), atol, rtol);

        if(!compareOutputTensor(uid, *attrs, dataType, expectedTensor, actualTensor, atol, rtol))
        {
            allMatched = false;
        }
    }
    return allMatched;
}

// ---- reporting helpers -----------------------------------------------------

VerificationOutcome IntegrationBundleVerificationHarness::unverifiable(const std::string& reason,
                                                                       VerificationDepth reached)
{
    UnverifiableBundleReport::get().record(
        _bundlePath.string(), reason, UnverifiableSeverity::UNVERIFIABLE);
    return VerificationOutcome::skipped(
        reached, "Unverifiable: " + reason + " (" + _bundlePath.string() + ")");
}

void IntegrationBundleVerificationHarness::recordRefError(const std::string& reason)
{
    UnverifiableBundleReport::get().record(
        _bundlePath.string(), reason, UnverifiableSeverity::REF_ERROR);
}

std::string IntegrationBundleVerificationHarness::refLabel(ReferenceExecutorType type)
{
    return type == ReferenceExecutorType::GPU ? "GPU reference" : "CPU reference";
}

// ---- comparison + tolerance machinery --------------------------------------

bool IntegrationBundleVerificationHarness::compareOutputTensor(
    int64_t uid,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
    hipdnn_data_sdk::utilities::ITensor& expected,
    hipdnn_data_sdk::utilities::ITensor& actual,
    float atol,
    float rtol) const
{
    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, atol, rtol);
    const bool passed = validator->allClose(expected, actual);

    if(!passed)
    {
        const auto label = labelFor(uid, attrs);
        hipdnn_test_sdk::utilities::ComparisonContext ctx;
        ctx.contextLine = "Bundle: " + _bundlePath.string();
        ctx.tensorLabel = label + " (UID " + std::to_string(uid) + ", output)";
        ctx.dtypeName = hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
        ctx.atol = atol;
        ctx.rtol = rtol;

        std::ostringstream report;
        report << hipdnn_test_sdk::utilities::formatComparisonHeader(ctx, expected);
        hipdnn_test_sdk::utilities::appendComparisonDiffByDataType(
            report, dataType, label, expected, actual, atol, rtol);
        // Reported here so the diff lands next to the tensor it describes; the
        // outcome carries no message because of it.
        EXPECT_TRUE(false) << report.str();
    }
    return passed;
}

std::string IntegrationBundleVerificationHarness::labelFor(
    int64_t uid, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
{
    const auto* name = attrs.name();
    return (name != nullptr && !name->empty()) ? name->str() : ("uid=" + std::to_string(uid));
}

void IntegrationBundleVerificationHarness::applyMetadataGuards() const
{
    if(auto reason
       = checkVramRequirement(_bundle->metadata, TestConfig::get().getCurrentDeviceVramMb()))
    {
        GTEST_SKIP() << *reason;
    }

    if(auto reason = checkArchCompatibility(_bundle->metadata, TestConfig::get().getCurrentArch()))
    {
        GTEST_SKIP() << *reason;
    }
}

} // namespace hipdnn_integration_tests::bundle

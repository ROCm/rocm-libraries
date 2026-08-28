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

// ---- virtual defaults ------------------------------------------------------

void IntegrationBundleVerificationHarness::executeGraphThroughEngine(
    std::unordered_map<int64_t, void*>& variantPack)
{
    auto handle = getSharedHandle();

    const std::vector<uint8_t> graphBytes(
        _bundle->graphBuffer.data(), _bundle->graphBuffer.data() + _bundle->graphBuffer.size());

    hipdnn_frontend::graph::Graph graph;
    auto err = graph.from_binary(handle, graphBytes);
    ASSERT_TRUE(err.is_good()) << "from_binary failed: " << err.get_message();

    // Reuse the ranked list TestBody() already asked for when enforcement is on;
    // the list is a property of (graph, handle), so a second heuristic query would
    // fan out to every plugin again for the same answer.
    std::vector<int64_t> engineIds;
    hipdnn_frontend::ErrorCode statusCode = hipdnn_frontend::ErrorCode::OK;
    if(_query.ran)
    {
        engineIds = _query.rankedIds;
        statusCode = _query.statusCode;
    }
    else
    {
        auto status = graph.get_ranked_engine_ids(engineIds);
        statusCode = status.get_code();
        _query.ran = true;
        _query.statusCode = statusCode;
        _query.statusMessage = status.get_message();
        _query.rankedIds = engineIds;
    }
    const bool queryFailed = statusCode != hipdnn_frontend::ErrorCode::OK;

    const auto graphSummary = [&] {
        return std::to_string(_bundle->outputTensorUids.size()) + " output tensor(s), "
               + std::to_string(engineIds.size()) + " ranked engine(s)";
    };

    if(_engineUnderTest.has_value())
    {
        const int64_t targetEngineId = _engineUnderTest->id;

        if(queryFailed
           || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
        {
            throw EngineNotApplicableError("Engine " + _engineUnderTest->name
                                           + " does not support this graph (" + graphSummary()
                                           + ")");
        }
        graph.set_preferred_engine_id_ext(targetEngineId);
    }
    else
    {
        if(queryFailed || engineIds.empty())
        {
            throw EngineNotApplicableError("No engine supports this graph (" + graphSummary()
                                           + ")");
        }
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

SupportObservation IntegrationBundleVerificationHarness::observeSupportForBundle()
{
    if(_bundle == nullptr || !shouldEnforceClaims())
    {
        return {};
    }

    auto handle = getSharedHandle();

    const std::vector<uint8_t> graphBytes(
        _bundle->graphBuffer.data(), _bundle->graphBuffer.data() + _bundle->graphBuffer.size());

    hipdnn_frontend::graph::Graph graph;
    auto err = graph.from_binary(handle, graphBytes);
    if(!err.is_good())
    {
        // ADD_FAILURE rather than ASSERT_TRUE: this runs before runComparison(), and
        // the executor's own ASSERT on the same call would otherwise be shadowed.
        ADD_FAILURE() << "from_binary failed: " << err.get_message();
        return {};
    }

    std::vector<int64_t> rankedIds;
    auto status = graph.get_ranked_engine_ids(rankedIds);

    _query.ran = true;
    _query.statusCode = status.get_code();
    _query.statusMessage = status.get_message();
    _query.rankedIds = rankedIds;

    return observeSupport(_query.statusCode,
                          _query.rankedIds,
                          _claimLocator,
                          *_engineUnderTest,
                          baseArchToken(TestConfig::get().getCurrentArch()),
                          currentPlatform(),
                          _query.statusMessage);
}

void IntegrationBundleVerificationHarness::recordClaimCoverage(
    const SupportObservation& observation)
{
    if(observation.sidecarChecked)
    {
        supportClaimCoverage().graphsQueried++;
        if(!observation.hasApplicableClaim())
        {
            // Read in full, but silent about this arch/platform/case. Counted so
            // "we checked and it holds" is distinguishable from "we checked and
            // nobody had said anything" — identical in the verdict tallies, and
            // only one of them means the cell is covered.
            supportClaimCoverage().graphsWithNoApplicableClaim++;
        }
    }

    // Per-graph coverage invariant. The run-level guard only fires when *no* graph
    // anywhere was queried, so a partial gap slips through it. Per graph, a future
    // short-circuit above the query is loud immediately instead of surviving behind
    // one healthy bundle.
    if(shouldEnforceClaims() && !observation.sidecarChecked)
    {
        ADD_FAILURE() << "support claims exist for " << _bundlePath
                      << " but were never queried; enforcement would have passed "
                         "without checking them";
    }
}

std::string IntegrationBundleVerificationHarness::aggregateClaimFailures(
    const std::vector<SupportResult>& results) const
{
    std::string aggregate;
    for(const auto& result : results)
    {
        if(isFailure(result.verdict))
        {
            // Aggregated rather than failed on sight so one FAIL() names every
            // broken engine, instead of the first one hiding the rest.
            aggregate += formatVerdictMessage(result);
        }
    }
    return aggregate;
}

// An accepted claim is an observation about the ranked list, taken before the graph
// was built or run. Publishing it as working support when the same test then failed
// to build, execute, or match would feed a support matrix a cell that demonstrably
// does not work — so the promotion happens here, once, and only for the engine this
// test actually drove.
void IntegrationBundleVerificationHarness::commitClaims(const std::vector<SupportResult>& results,
                                                        bool exercised)
{
    // results is only ever non-empty when an engine was injected, so there is no
    // unpinned case to handle here.
    if(!_engineUnderTest.has_value())
    {
        return;
    }

    const bool passed = !HasFailure();
    const std::string& underTest = _engineUnderTest->name;

    for(auto record : results)
    {
        if(record.verdict == SupportVerdict::CLAIM_ACCEPTED && record.engineName == underTest)
        {
            record.verdict = promoteAcceptedClaim(exercised, passed);
            if(record.verdict != SupportVerdict::CLAIM_ACCEPTED)
            {
                record.detail = record.verdict == SupportVerdict::CLAIM_CONFIRMED
                                    ? "engine in ranked list; graph executed and verified"
                                    : "engine accepted the graph, but the test did not pass";
            }
        }
        SupportClaimVerdicts::get().record(record);
    }
}

void IntegrationBundleVerificationHarness::enforceAtLevel(EnforcementLevel level)
{
    ASSERT_NE(level, EnforcementLevel::FULL)
        << "enforceAtLevel() handles APPLICABILITY/BUILDABLE only; FULL uses the normal path";

    // The rung itself needs a named engine to check applicability against; claim
    // enforcement already ran in TestBody() and is not affected by this return.
    if(!_engineUnderTest.has_value())
    {
        skipUnverifiable("enforcement_level requires --test-engine");
        return;
    }

    auto handle = getSharedHandle();

    const std::vector<uint8_t> graphBytes(
        _bundle->graphBuffer.data(), _bundle->graphBuffer.data() + _bundle->graphBuffer.size());

    hipdnn_frontend::graph::Graph graph;
    auto err = graph.from_binary(handle, graphBytes);
    ASSERT_TRUE(err.is_good()) << "from_binary failed: " << err.get_message();

    std::vector<int64_t> engineIds;
    hipdnn_frontend::ErrorCode statusCode = hipdnn_frontend::ErrorCode::OK;
    if(_query.ran)
    {
        engineIds = _query.rankedIds;
        statusCode = _query.statusCode;
    }
    else
    {
        auto status = graph.get_ranked_engine_ids(engineIds);
        statusCode = status.get_code();
        _query.ran = true;
        _query.statusCode = statusCode;
        _query.statusMessage = status.get_message();
        _query.rankedIds = engineIds;
    }

    const std::string rung
        = level == EnforcementLevel::APPLICABILITY ? "applicability" : "buildable";

    const int64_t targetEngineId = _engineUnderTest->id;

    if(statusCode != hipdnn_frontend::ErrorCode::OK
       || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
    {
        skipUnverifiable("Engine " + _engineUnderTest->name
                         + " does not support this graph (enforcement_level=" + rung + ")");
        return;
    }

    if(level == EnforcementLevel::APPLICABILITY)
    {
        _verified = true;
        return;
    }

    // BUILDABLE: additionally compile plans
    graph.set_preferred_engine_id_ext(targetEngineId);
    auto result = graph.create_execution_plans();
    ASSERT_TRUE(result.is_good()) << "[rung=buildable] " << result.get_message();
    result = graph.check_support();
    ASSERT_TRUE(result.is_good()) << "[rung=buildable] " << result.get_message();
    result = graph.build_plans();
    ASSERT_TRUE(result.is_good()) << "[rung=buildable] " << result.get_message();
    _verified = true;
}

void IntegrationBundleVerificationHarness::runComparison()
{
    if(_bundle->metadata.enforcementLevel != EnforcementLevel::FULL)
    {
        enforceAtLevel(_bundle->metadata.enforcementLevel);
        return;
    }

    if(_bundle->outputTensorUids.empty())
    {
        skipUnverifiable("bundle has no output tensors to compare");
        return;
    }

    if(!ensureInputsAvailable())
    {
        return;
    }

    switch(getVerificationMode())
    {
    case VerificationMode::GOLDEN:
        runGoldenMode();
        return;
    case VerificationMode::GPU:
        runExplicitRefMode(ReferenceExecutorType::GPU);
        return;
    case VerificationMode::CPU:
        runExplicitRefMode(ReferenceExecutorType::CPU);
        return;
    case VerificationMode::AUTO:
        runAutoMode();
        return;
    default:
        FAIL() << "Unknown verification mode";
        return;
    }
}

namespace
{
// GTEST_SKIP() expands to `return;`, so it can only be used from a void-returning
// function. This wrapper records the skip (and its message) and returns from
// itself; the skip state persists for the caller, which then returns nullopt.
void skipEngineCouldNotRun(const std::filesystem::path& bundlePath, const std::string& error)
{
    std::ostringstream msg;
    msg << "Engine could not execute bundle " << bundlePath;
    if(!error.empty())
    {
        msg << ": " << error;
    }
    GTEST_SKIP() << msg.str();
}
} // namespace

std::optional<OutputTensors> IntegrationBundleVerificationHarness::runEngineOrSkip()
{
    std::string error;
    auto engineOutputs = runEngineCapturingOutputs(error);
    if(!engineOutputs && !::testing::Test::HasFatalFailure())
    {
        _verified = true;
        skipEngineCouldNotRun(_bundlePath, error);
    }
    return engineOutputs;
}

void IntegrationBundleVerificationHarness::runGoldenMode()
{
    // An explicit --verification-mode=golden is a demand for a specific oracle, not
    // a preference. Skipping when that oracle is absent means the run did not do
    // what it was asked and still went green — use `auto` if a fallback chain is
    // what you want.
    if(!_bundle->hasGoldenOutputs)
    {
        FAIL() << "verification-mode=golden was requested but this bundle has no golden "
                  "data; run `dvc pull` for it, or use --verification-mode=auto";
        return;
    }
    auto engineOutputs = runEngineOrSkip();
    if(!engineOutputs)
    {
        return;
    }
    compareAgainstGolden(*engineOutputs);
}

void IntegrationBundleVerificationHarness::runExplicitRefMode(ReferenceExecutorType type)
{
    auto engineOutputs = runEngineOrSkip();
    if(!engineOutputs)
    {
        return;
    }

    OutputTensors refOutputs;
    const RefRunResult result = runReferenceCapturingOutputs(type, refOutputs);
    switch(result.status)
    {
    case RefStatus::CAPABILITY_MISS:
        skipUnverifiable(refLabel(type) + " cannot run this op: " + result.message);
        return;
    case RefStatus::RUNTIME_ERROR:
        recordRefError(refLabel(type) + " errored: " + result.message);
        FAIL() << refLabel(type) << " errored (verification-mode=" << refLabel(type)
               << "): " << result.message;
        return;
    case RefStatus::RAN:
        compareOutputs(*engineOutputs, refOutputs);
        return;
    default:
        FAIL() << "Unknown RefStatus";
        return;
    }
}

void IntegrationBundleVerificationHarness::runAutoMode()
{
    auto engineOutputs = runEngineOrSkip();
    if(!engineOutputs)
    {
        return;
    }

    if(_bundle->hasGoldenOutputs)
    {
        compareAgainstGolden(*engineOutputs);
        return;
    }

    // GPU ref (non-final): capability miss or runtime error -> fall through.
    bool gpuRefErrored = false;
    {
        OutputTensors refOutputs;
        const RefRunResult gpu
            = runReferenceCapturingOutputs(ReferenceExecutorType::GPU, refOutputs);
        if(gpu.status == RefStatus::RAN)
        {
            compareOutputs(*engineOutputs, refOutputs);
            return;
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
            skipUnverifiable(gpuRefErrored
                                 ? "no usable reference (golden absent; GPU ref errored, CPU ref "
                                   "cannot run this op; see reference-error report): "
                                       + cpu.message
                                 : "no reference available (golden absent; GPU and CPU ref "
                                   "cannot run this op): "
                                       + cpu.message);
            return;
        case RefStatus::RUNTIME_ERROR:
            recordRefError("CPU reference errored (auto mode, last resort): " + cpu.message);
            FAIL() << "CPU reference errored (auto mode, last resort): " << cpu.message;
            return;
        case RefStatus::RAN:
            compareOutputs(*engineOutputs, refOutputs);
            return;
        default:
            FAIL() << "Unknown RefStatus";
            return;
        }
    }
}

// ---- inputs ----------------------------------------------------------------

bool IntegrationBundleVerificationHarness::ensureInputsAvailable()
{
    if(_bundle->tensors.has_value())
    {
        return true;
    }
    return fillBundleInputs();
}

bool IntegrationBundleVerificationHarness::fillBundleInputs()
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
        skipUnverifiable(fillResult.reason);
        return false;
    }

    _bundle->tensors = std::move(inputs);
    return true;
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

std::optional<OutputTensors>
    IntegrationBundleVerificationHarness::runEngineCapturingOutputs(std::string& error)
{
    OutputTensors engineOutputs = allocateSentinelOutputs();
    auto variantPack = buildVariantPack(engineOutputs, /*useDevice=*/_requiresDevice);

    try
    {
        executeGraphThroughEngine(variantPack);
        // Recorded here, not inferred from skip state in TestBody(). "The engine ran
        // this graph" is the only basis on which an accepted claim may be promoted,
        // and a mode can now fail before ever reaching the engine — attributing that
        // to the engine would brand a missing .bin as a broken implementation.
        _engineRan = true;
    }
    catch(const EngineNotApplicableError& e)
    {
        error = e.what();
        return std::nullopt;
    }

    if(::testing::Test::HasFatalFailure())
    {
        return std::nullopt;
    }

    markOutputsModified(engineOutputs);
    return engineOutputs;
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

void IntegrationBundleVerificationHarness::compareAgainstGolden(OutputTensors& engineOutputs)
{
    compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *_bundle->tensors->at(uid);
    });
}

void IntegrationBundleVerificationHarness::compareOutputs(OutputTensors& engineOutputs,
                                                          OutputTensors& expected)
{
    compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *expected.at(uid);
    });
}

template <typename ExpectedLookup>
void IntegrationBundleVerificationHarness::compareEach(OutputTensors& engineOutputs,
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

    for(const int64_t uid : _bundle->outputTensorUids)
    {
        auto& actualTensor = *engineOutputs.at(uid);
        auto& expectedTensor = expectedFor(uid);

        auto* attrs = tensorAttrMap.at(uid);
        const auto dataType = attrs->data_type();

        float atol = 0.0f;
        float rtol = 0.0f;
        tolerance::resolveTolerance(wrapper, dataType, currentTestName(), atol, rtol);

        compareOutputTensor(uid, *attrs, dataType, expectedTensor, actualTensor, atol, rtol);
    }
}

// ---- reporting helpers -----------------------------------------------------

void IntegrationBundleVerificationHarness::skipUnverifiable(const std::string& reason)
{
    _verified = true;
    UnverifiableBundleReport::get().record(
        _bundlePath.string(), reason, UnverifiableSeverity::UNVERIFIABLE);
    GTEST_SKIP() << "Unverifiable: " << reason << " (" << _bundlePath << ")";
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

void IntegrationBundleVerificationHarness::compareOutputTensor(
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
    _verified = true;

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
        EXPECT_TRUE(false) << report.str();
    }
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

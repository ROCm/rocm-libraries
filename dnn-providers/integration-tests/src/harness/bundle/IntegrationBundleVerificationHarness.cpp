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

    std::vector<int64_t> engineIds;
    auto status = graph.get_ranked_engine_ids(engineIds);

    const auto graphSummary = [&] {
        return std::to_string(_bundle->outputTensorUids.size()) + " output tensor(s), "
               + std::to_string(engineIds.size()) + " ranked engine(s)";
    };

    if(TestConfig::get().enforceSupportClaims())
    {
        const auto allVerdicts = observeAllSupport(status.get_code(),
                                                   engineIds,
                                                   _claimLocator,
                                                   LoadedEngineTable::get().all(),
                                                   status.get_message());
        if(!allVerdicts.empty())
        {
            supportClaimCoverage().graphsQueried++;
        }

        const bool hasPinnedEngine = TestConfig::get().hasEngineName();
        const std::string pinnedName
            = hasPinnedEngine ? std::string(TestConfig::get().getEngineName()) : std::string{};

        std::string failureAggregate;
        for(const auto& v : allVerdicts)
        {
            SupportClaimVerdicts::get().record(v);
            if(isFailure(v.verdict) && (!hasPinnedEngine || v.engineName == pinnedName))
            {
                failureAggregate += formatVerdictMessage(v);
            }
        }
        if(!failureAggregate.empty())
        {
            FAIL() << failureAggregate;
            return;
        }
    }

    if(TestConfig::get().hasEngineName())
    {
        const int64_t targetEngineId = TestConfig::get().getEngineId();

        if(status.is_bad()
           || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
        {
            throw EngineNotApplicableError(
                "Engine " + std::string(TestConfig::get().getEngineName())
                + " does not support this graph (" + graphSummary() + ")");
        }
        graph.set_preferred_engine_id_ext(targetEngineId);
    }
    else
    {
        if(status.is_bad() || engineIds.empty())
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

void IntegrationBundleVerificationHarness::enforceAtLevel(EnforcementLevel level)
{
    ASSERT_NE(level, EnforcementLevel::FULL)
        << "enforceAtLevel() handles APPLICABILITY/BUILDABLE only; FULL uses the normal path";

    if(!TestConfig::get().hasEngineName())
    {
        skipUnverifiable("enforcement requires --test-engine");
        return;
    }

    auto handle = getSharedHandle();

    const std::vector<uint8_t> graphBytes(
        _bundle->graphBuffer.data(), _bundle->graphBuffer.data() + _bundle->graphBuffer.size());

    hipdnn_frontend::graph::Graph graph;
    auto err = graph.from_binary(handle, graphBytes);
    ASSERT_TRUE(err.is_good()) << "from_binary failed: " << err.get_message();

    std::vector<int64_t> engineIds;
    auto status = graph.get_ranked_engine_ids(engineIds);

    const std::string rung
        = level == EnforcementLevel::APPLICABILITY ? "applicability" : "buildable";

    if(isEnforcingSupportClaims())
    {
        const auto allVerdicts = observeAllSupport(status.get_code(),
                                                   engineIds,
                                                   _claimLocator,
                                                   LoadedEngineTable::get().all(),
                                                   status.get_message());
        if(!allVerdicts.empty())
        {
            supportClaimCoverage().graphsQueried++;
        }

        const std::string pinnedName = std::string(TestConfig::get().getEngineName());

        std::string failureAggregate;
        for(const auto& v : allVerdicts)
        {
            SupportClaimVerdicts::get().record(v);
            if(isFailure(v.verdict) && v.engineName == pinnedName)
            {
                failureAggregate += formatVerdictMessage(v);
            }
        }
        if(!failureAggregate.empty())
        {
            FAIL() << "[rung=" << rung << "] " << failureAggregate;
            return;
        }

        if(allVerdicts.empty())
        {
            skipUnverifiable("enforcement_level=" + rung
                             + " but no support claims found for loaded engines");
            return;
        }
    }

    const int64_t targetEngineId = TestConfig::get().getEngineId();

    if(status.is_bad()
       || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
    {
        skipUnverifiable("Engine " + std::string(TestConfig::get().getEngineName())
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
    case VerificationMode::GOLDEN_CHECK:
        runGoldenCheckMode();
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
    if(!_bundle->hasGoldenOutputs)
    {
        skipUnverifiable("no golden data (verification-mode=golden)");
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

void IntegrationBundleVerificationHarness::runGoldenCheckMode()
{
    if(!_bundle->hasGoldenOutputs)
    {
        skipUnverifiable("no golden data (verification-mode=golden-check)");
        return;
    }

    OutputTensors cpuOutputs;
    const RefRunResult result
        = runReferenceCapturingOutputs(ReferenceExecutorType::CPU, cpuOutputs);
    switch(result.status)
    {
    case RefStatus::CAPABILITY_MISS:
        skipUnverifiable("CPU ref cannot run this op (golden-check): " + result.message);
        return;
    case RefStatus::RUNTIME_ERROR:
        recordRefError("CPU ref errored (golden-check): " + result.message);
        FAIL() << "CPU ref errored (golden-check): " << result.message;
        return;
    case RefStatus::RAN:
        compareEach(cpuOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
            return *_bundle->tensors->at(uid);
        });
        return;
    default:
        FAIL() << "Unknown RefStatus";
        return;
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

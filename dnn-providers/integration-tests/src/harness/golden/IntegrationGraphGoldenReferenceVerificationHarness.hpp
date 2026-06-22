// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"
#include "harness/golden/UnverifiableBundleReport.hpp"
#include "harness/golden/input_init/SynthesizeInputs.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::golden
{

// Output tensors, keyed by uid. Used both for the engine's computed "actual"
// outputs and for an expected source (golden from disk, or a reference executor's
// output). Each set is a distinct allocation so engine and reference never write
// the same buffers.
using OutputTensors
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// Verifies a bundle's engine output against an expected source chosen by the
// verification mode (RFC 0010 §4.4):
//
//   actual   = the engine (the system under test), run once into fresh buffers.
//   expected = golden data from disk, OR a reference executor's output.
//
// Memory invariants for running engine + a reference off the same inputs:
//   * INPUT tensors are read-only by both executors and are NEVER mark*Modified().
//     The engine's rawDeviceData() uploads host->device (state becomes BOTH
//     valid); a later CPU-ref rawHostData() therefore sees the host copy still
//     valid and does NOT download — inputs stay intact across both runs.
//   * OUTPUT buffers are separate ITensor objects per executor (engineOutputs vs
//     refOutputs), so the two runs cannot stomp each other. Only output buffers
//     are mark*Modified().
//   * Virtual (inter-node) tensors are allocated internally by each executor; the
//     variant packs we build carry only real (input + output) tensors.
class IntegrationGraphGoldenReferenceVerificationHarness : public ::testing::Test
{
public:
    explicit IntegrationGraphGoldenReferenceVerificationHarness(bool requiresDevice)
        : _requiresDevice(requiresDevice)
    {
    }

    // The bundle is loaded once at registration time and shared into the test's
    // factory; the harness does not load from disk. The path is kept only for
    // diagnostic messages and the unverifiable report.
    void setBundle(std::shared_ptr<IntegrationTestBundle> bundle, std::filesystem::path path)
    {
        _bundle = std::move(bundle);
        _bundlePath = std::move(path);
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

        applyMetadataGuards();
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override
    {
        runComparison();
    }

    // Builds the graph from its serialized bytes, selects an engine (honouring an
    // explicit --engine if given), builds plans, and executes into the variant
    // pack. "Unsupported graph" is signalled by throwing (the harness translates
    // that into a SKIP). Genuine build/execute errors use ASSERT_*.
    virtual void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack)
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

        if(TestConfig::get().hasEngineName())
        {
            int64_t targetEngineId = TestConfig::get().getEngineId();
            if(status.is_bad()
               || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
            {
                throw std::runtime_error("Engine " + std::string(TestConfig::get().getEngineName())
                                         + " does not support this graph (" + graphSummary() + ")");
            }
            graph.set_preferred_engine_id_ext(targetEngineId);
        }
        else
        {
            if(status.is_bad() || engineIds.empty())
            {
                throw std::runtime_error("No engine supports this graph (" + graphSummary() + ")");
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

    // Runs a reference executor (the chosen expected-output source) into the given
    // variant pack. Throws ReferenceCapabilityError when the executor has no plan
    // for the op (capability miss, case A); throws any other exception for a
    // genuine runtime failure (case C). Virtual so unit tests can stub it the same
    // way they stub executeGraphThroughEngine.
    virtual void runReferenceExecutor(ReferenceExecutorType type,
                                      std::unordered_map<int64_t, void*>& variantPack)
    {
        auto executor = makeReferenceExecutor(type);
        executor->execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
    }

    // Factory split out so a stub harness can short-circuit it. Default: the real
    // CPU / GPU reference executors.
    virtual std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType type)
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

private:
    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    std::shared_ptr<IntegrationTestBundle> _bundle;

    // ---- top-level dispatch -------------------------------------------------

    void runComparison()
    {
        if(_bundle->outputTensorUids.empty())
        {
            skipUnverifiable("bundle has no output tensors to compare");
            return;
        }

        if(!ensureInputsAvailable())
        {
            return; // skipUnverifiable already recorded + GTEST_SKIP issued
        }

        switch(TestConfig::get().getVerificationMode())
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

    // golden mode: golden data only.
    void runGoldenMode()
    {
        if(!_bundle->hasGoldenOutputs)
        {
            skipUnverifiable("no golden data (verification-mode=golden)");
            return;
        }
        auto engineOutputs = runEngineCapturingOutputs();
        if(!engineOutputs)
        {
            if(!::testing::Test::HasFatalFailure())
            {
                GTEST_SKIP() << "Engine could not execute bundle " << _bundlePath;
            }
            return;
        }
        compareAgainstGolden(*engineOutputs);
    }

    // explicit gpu / cpu mode: ignore golden; compare against the named reference.
    //   A (capability miss) -> SKIP+report
    //   C (runtime error)   -> FAIL (the user named this reference)
    //   B (mismatch)        -> FAIL
    void runExplicitRefMode(ReferenceExecutorType type)
    {
        auto engineOutputs = runEngineCapturingOutputs();
        if(!engineOutputs)
        {
            if(!::testing::Test::HasFatalFailure())
            {
                GTEST_SKIP() << "Engine could not execute bundle " << _bundlePath;
            }
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

    // auto mode: golden -> GPU ref -> CPU ref -> SKIP+report.
    //   capability miss falls through; a runtime error in a non-final ref is loud
    //   but still falls through (keep verifying the engine); a runtime error in the
    //   final ref (CPU) is a FAIL; a mismatch anywhere is a FAIL (never a second
    //   opinion).
    void runAutoMode()
    {
        auto engineOutputs = runEngineCapturingOutputs();
        if(!engineOutputs)
        {
            if(!::testing::Test::HasFatalFailure())
            {
                GTEST_SKIP() << "Engine could not execute bundle " << _bundlePath;
            }
            return;
        }

        if(_bundle->hasGoldenOutputs)
        {
            compareAgainstGolden(*engineOutputs);
            return;
        }

        // GPU ref (non-final): capability miss or runtime error -> fall through.
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
                // A reference that CAN run the op but failed is a reference bug:
                // loud, but we still fall through to keep verifying the engine.
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
                skipUnverifiable("no reference available (golden absent; GPU and CPU ref "
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

    // ---- inputs -------------------------------------------------------------

    // Ensures _bundle->tensors holds usable input data. tier 1/2: already loaded
    // from disk. tier 3 (tensors == nullopt): try to synthesize inputs from the
    // graph. Returns false (after recording + SKIP) when neither is possible.
    bool ensureInputsAvailable()
    {
        if(_bundle->tensors.has_value())
        {
            return true; // inputs (and maybe golden outputs) loaded from disk
        }
        return synthesizeInputs();
    }

    // tier-3 synthesis: single-node graph whose op has a registered initializer.
    // Builds zeroed input tensors from graph attributes, routes each leaf input to
    // its owning node's initializer, and fills them. Any refusal -> SKIP+report.
    bool synthesizeInputs()
    {
        const auto wrapper = _bundle->graphWrapper();

        if(wrapper.nodeCount() != 1)
        {
            skipUnverifiable("graph-only bundle with no input data: input synthesis supports "
                             "single-node graphs only (this graph has "
                             + std::to_string(wrapper.nodeCount()) + " nodes)");
            return false;
        }

        const auto& node = wrapper.getNode(0);

        // Leaf inputs = non-virtual tensors that are not graph outputs. (For a
        // single-node graph every such tensor is an input to that node.)
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
            inputs[uid]->fillTensorWithValue(0.f);
            leafInputUids.push_back(uid);
        }

        std::mt19937 rng(static_cast<std::mt19937::result_type>(
            _bundle->metadata.seed.value_or(K_DEFAULT_SEED)));

        const FillOutcome outcome = synthesizeNodeInputs(node, leafInputUids, inputs, rng);
        if(!outcome.filled)
        {
            skipUnverifiable(outcome.reason);
            return false;
        }

        _bundle->tensors = std::move(inputs);
        return true;
    }

    // ---- engine + reference runs -------------------------------------------

    // Allocate fresh zeroed output buffers (one ITensor per output uid) from the
    // graph's tensor attributes — no .bin needed.
    OutputTensors allocateZeroedOutputs() const
    {
        const auto wrapper = _bundle->graphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        OutputTensors outputs;
        for(const int64_t uid : _bundle->outputTensorUids)
        {
            outputs[uid]
                = hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttrMap.at(uid));
            outputs[uid]->fillTensorWithValue(0.f);
        }
        return outputs;
    }

    // Build a variant pack: inputs from _bundle->tensors, outputs from `outputs`.
    // useDevice selects device vs host pointers (engine/GPU-ref use device; CPU-ref
    // uses host). Inputs are read but never mark*Modified() (see class invariants).
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const
    {
        std::unordered_map<int64_t, void*> variantPack;
        const std::set<int64_t> outputUids(_bundle->outputTensorUids.begin(),
                                           _bundle->outputTensorUids.end());

        for(auto& [uid, tensor] : *_bundle->tensors)
        {
            if(outputUids.count(uid) != 0)
            {
                continue; // golden output from disk; use the fresh buffer below instead
            }
            variantPack[uid] = useDevice ? tensor->rawDeviceData() : tensor->rawHostData();
        }
        for(auto& [uid, tensor] : outputs)
        {
            variantPack[uid] = useDevice ? tensor->rawDeviceData() : tensor->rawHostData();
        }
        return variantPack;
    }

    // Run the engine into fresh output buffers. Returns nullopt if the engine
    // signalled "unsupported graph" (SKIP already issued) or a fatal assertion
    // fired inside the executor.
    std::optional<OutputTensors> runEngineCapturingOutputs()
    {
        OutputTensors engineOutputs = allocateZeroedOutputs();
        auto variantPack = buildVariantPack(engineOutputs, /*useDevice=*/_requiresDevice);

        // Call the executor directly (not via ASSERT_NO_FATAL_FAILURE, which would
        // `return;` and cannot compile in this value-returning function). A fatal
        // ASSERT_* inside the executor returns from it and sets the fatal-failure
        // flag, which we detect below and surface as nullopt.
        bool threw = false;
        std::string error;
        try
        {
            executeGraphThroughEngine(variantPack);
        }
        catch(const std::exception& e)
        {
            threw = true;
            error = e.what();
        }

        if(::testing::Test::HasFatalFailure())
        {
            return std::nullopt;
        }
        if(threw)
        {
            // GTEST_SKIP contains `return;` which cannot compile in a non-void
            // function. Callers detect nullopt and issue the skip themselves.
            return std::nullopt;
        }

        markOutputsModified(engineOutputs);
        return engineOutputs;
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

    // Run a reference executor into fresh output buffers `refOutputs`.
    //   ReferenceCapabilityError -> CapabilityMiss (case A)
    //   any other std::exception -> RuntimeError   (case C)
    RefRunResult runReferenceCapturingOutputs(ReferenceExecutorType type, OutputTensors& refOutputs)
    {
        refOutputs = allocateZeroedOutputs();
        const bool useDevice = (type == ReferenceExecutorType::GPU);
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

    void markOutputsModified(OutputTensors& outputs) const
    {
        markOutputsModifiedFor(outputs, _requiresDevice);
    }

    static void markOutputsModifiedFor(OutputTensors& outputs, bool device)
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

    // ---- comparison ---------------------------------------------------------

    // Compare engine output against the golden outputs stored in _bundle->tensors.
    void compareAgainstGolden(OutputTensors& engineOutputs)
    {
        compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
            return *_bundle->tensors->at(uid);
        });
    }

    void compareOutputs(OutputTensors& engineOutputs, OutputTensors& expected)
    {
        compareEach(engineOutputs, [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
            return *expected.at(uid);
        });
    }

    template <typename ExpectedLookup>
    void compareEach(OutputTensors& engineOutputs, ExpectedLookup expectedFor)
    {
        auto wrapper = _bundle->graphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(const int64_t uid : _bundle->outputTensorUids)
        {
            auto& actualTensor = *engineOutputs.at(uid);
            auto& expectedTensor = expectedFor(uid);

            auto* attrs = tensorAttrMap.at(uid);
            const auto dataType = attrs->data_type();

            float atol = 0.0f;
            float rtol = 0.0f;
            resolveTolerances(wrapper, dataType, atol, rtol);

            compareOutputTensor(uid, *attrs, dataType, expectedTensor, actualTensor, atol, rtol);
        }
    }

    // ---- reporting helpers --------------------------------------------------

    void skipUnverifiable(const std::string& reason)
    {
        UnverifiableBundleReport::get().record(
            _bundlePath.string(), reason, UnverifiableSeverity::UNVERIFIABLE);
        GTEST_SKIP() << "Unverifiable: " << reason << " (" << _bundlePath << ")";
    }

    void recordRefError(const std::string& reason)
    {
        UnverifiableBundleReport::get().record(
            _bundlePath.string(), reason, UnverifiableSeverity::REF_ERROR);
    }

    static std::string refLabel(ReferenceExecutorType type)
    {
        return type == ReferenceExecutorType::GPU ? "GPU reference" : "CPU reference";
    }

    static constexpr int64_t K_DEFAULT_SEED = 42;

    // ---- comparison + tolerance machinery (unchanged behaviour) -------------

    // Compare one output tensor against its expected reference via the allClose
    // validator. Only on failure do we compute and report the element-wise diff.
    void compareOutputTensor(int64_t uid,
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
            std::ostringstream report;
            report << reportHeader(uid, attrs, dataType, expected, atol, rtol);
            appendTensorDiff(report, uid, attrs, dataType, expected, actual, atol, rtol);
            EXPECT_TRUE(false) << report.str();
        }
    }

    static void
        appendTensorDiff(std::ostream& os,
                         int64_t uid,
                         const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                         hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                         hipdnn_data_sdk::utilities::ITensor& expected,
                         hipdnn_data_sdk::utilities::ITensor& actual,
                         float atol,
                         float rtol)
    {
        using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
        using hipdnn_data_sdk::types::bfloat16;
        using hipdnn_data_sdk::types::half;

        switch(dataType)
        {
        case DT::FLOAT:
            appendFpDiff<float>(os, uid, attrs, expected, actual, atol, rtol);
            return;
        case DT::HALF:
            appendFpDiff<half>(os, uid, attrs, expected, actual, atol, rtol);
            return;
        case DT::BFLOAT16:
            appendFpDiff<bfloat16>(os, uid, attrs, expected, actual, atol, rtol);
            return;
        case DT::DOUBLE:
            appendFpDiff<double>(os, uid, attrs, expected, actual, atol, rtol);
            return;
        default:
            os << "  (no element-wise diff available for this data type)\n";
        }
    }

    template <typename T>
    static void appendFpDiff(std::ostream& os,
                             int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             hipdnn_data_sdk::utilities::ITensor& actual,
                             float atol,
                             float rtol)
    {
        const auto summary
            = hipdnn_test_sdk::utilities::computeTensorDiff<T>(expected, actual, atol, rtol);
        hipdnn_test_sdk::utilities::printTensorDiffSummary(os, labelFor(uid, attrs), summary);
    }

    static std::string labelFor(int64_t uid,
                                const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
    {
        const auto* name = attrs.name();
        return (name != nullptr && !name->empty()) ? name->str() : ("uid=" + std::to_string(uid));
    }

    std::string reportHeader(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             float atol,
                             float rtol) const
    {
        std::ostringstream os;
        os << "\nGolden comparison FAILED\n"
           << "  Bundle: " << _bundlePath << "\n"
           << "  Tensor: " << labelFor(uid, attrs) << " (UID " << uid << ", output)\n"
           << "  Shape:  " << hipdnn_test_sdk::utilities::StreamVec(expected.dims()) << "  "
           << dataTypeName(dataType) << "\n"
           << "  Tolerance: atol=" << atol << " rtol=" << rtol << "\n";
        return os.str();
    }

    static std::string dataTypeName(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        return hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
    }

    static void
        resolveTolerances(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                          hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                          float& atol,
                          float& rtol)
    {
        const float defaultTolerance = deriveDefaultTolerance(wrapper, dataType);
        atol = defaultTolerance;
        rtol = defaultTolerance;
    }

    template <typename T>
    static float
        toleranceForNodeAttributes(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
    {
        using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
        namespace tol = hipdnn_test_sdk::utilities;

        switch(attrType)
        {
        case NA::ConvolutionFwdAttributes:
            return tol::conv::getToleranceFwd<T>();
        case NA::ConvolutionBwdAttributes:
            return tol::conv::getToleranceBwd<T>();
        case NA::ConvolutionWrwAttributes:
            return tol::conv::getToleranceWrw<T>();
        case NA::BatchnormInferenceAttributes:
            return tol::batchnorm::getToleranceInference<T>();
        case NA::BatchnormInferenceAttributesVarianceExt:
            return tol::batchnorm::getToleranceInferenceWithVariance<T>();
        case NA::BatchnormAttributes:
            return tol::batchnorm::getToleranceTraining<T>();
        case NA::BatchnormBackwardAttributes:
            return tol::batchnorm::getToleranceBackward<T>();
        case NA::MatmulAttributes:
            return tol::matmul::getTolerance<T>();
        case NA::ReductionAttributes:
            return tol::reduction::getTolerance<T>();
        case NA::RMSNormAttributes:
            return tol::rmsnorm::getTolerance<T>();
        case NA::PointwiseAttributes:
            return tol::pointwise::getTolerance<T>();
        case NA::LayernormAttributes:
            return tol::layernorm::getTolerance<T>();
        default:
            return 1e-3f;
        }
    }

    // A bundle graph may fuse several ops; each op type has its own tolerance, so
    // the only tolerance that holds for the fused output is the loosest one across
    // all nodes. We therefore take the max over every node.
    static float deriveDefaultTolerance(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        const auto nodeCount = wrapper.nodeCount();

        bool found = false;
        float maxTolerance = 0.0f;
        for(uint32_t i = 0; i < nodeCount; ++i)
        {
            const auto attrType = wrapper.getNode(i).attributes_type();
            const float nodeTolerance = toleranceForDataType(attrType, dataType);
            maxTolerance = found ? std::max(maxTolerance, nodeTolerance) : nodeTolerance;
            found = true;
        }

        return found ? maxTolerance : 1e-3f;
    }

    static float toleranceForDataType(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
        using hipdnn_data_sdk::types::bfloat16;
        using hipdnn_data_sdk::types::half;

        switch(dataType)
        {
        case DT::FLOAT:
            return toleranceForNodeAttributes<float>(attrType);
        case DT::HALF:
            return toleranceForNodeAttributes<half>(attrType);
        case DT::BFLOAT16:
            return toleranceForNodeAttributes<bfloat16>(attrType);
        default:
            return 1e-3f;
        }
    }

    void applyMetadataGuards() const
    {
        if(auto reason = hipdnn_test_sdk::utilities::checkVramRequirement(
               _bundle->metadata, TestConfig::get().getCurrentDeviceVramMb()))
        {
            GTEST_SKIP() << *reason;
        }

        if(auto reason = hipdnn_test_sdk::utilities::checkArchCompatibility(
               _bundle->metadata, TestConfig::get().getCurrentArch()))
        {
            GTEST_SKIP() << *reason;
        }
    }
};

} // namespace hipdnn_integration_tests::golden

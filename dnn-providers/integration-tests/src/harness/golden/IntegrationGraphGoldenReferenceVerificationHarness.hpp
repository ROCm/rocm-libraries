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

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

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
// Auto mode fallback chain: golden → GPU ref → CPU ref → SKIP.
// When golden outputs are present on disk, the comparison uses them directly
// and no reference executor is run at all.
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

    // Builds the graph, selects an engine, and executes. Throws on unsupported graph (→ SKIP).
    virtual void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack);

    // Runs the named reference executor. Throws ReferenceCapabilityError on capability miss.
    virtual void runReferenceExecutor(ReferenceExecutorType type,
                                      std::unordered_map<int64_t, void*>& variantPack);

    // Constructs the executor object (CpuReferenceGraphExecutorAdapter or
    // GpuReferenceGraphExecutor) — does not allocate buffers or run anything.
    // Skipped in auto mode when golden data is present.
    virtual std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType type);

    // Returns the active verification mode. Override in tests to inject a mode
    // without touching the TestConfig singleton.
    virtual VerificationMode getVerificationMode() const;

private:
    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    std::shared_ptr<IntegrationTestBundle> _bundle;

    static constexpr int64_t K_DEFAULT_SEED = 42;

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

    // ── top-level dispatch ────────────────────────────────────────────────
    void runComparison();
    void runGoldenMode();
    void runExplicitRefMode(ReferenceExecutorType type);
    void runAutoMode();

    // ── inputs ──────────────────────────────────────────────────────────
    bool ensureInputsAvailable();
    bool synthesizeInputs();

    // ── buffer allocation + execution ───────────────────────────────────
    // allocateZeroedOutputs / buildVariantPack prepare the buffers;
    // runEngine* / runReference* call the executors and capture results.
    OutputTensors allocateZeroedOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const;
    std::optional<OutputTensors> runEngineCapturingOutputs();
    RefRunResult runReferenceCapturingOutputs(ReferenceExecutorType type,
                                              OutputTensors& refOutputs);
    void markOutputsModified(OutputTensors& outputs) const;
    static void markOutputsModifiedFor(OutputTensors& outputs, bool device);

    // ── comparison ──────────────────────────────────────────────────────
    void compareAgainstGolden(OutputTensors& engineOutputs);
    void compareOutputs(OutputTensors& engineOutputs, OutputTensors& expected);

    template <typename ExpectedLookup>
    void compareEach(OutputTensors& engineOutputs, ExpectedLookup expectedFor);

    void compareOutputTensor(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             hipdnn_data_sdk::utilities::ITensor& actual,
                             float atol,
                             float rtol) const;

    static void
        appendTensorDiff(std::ostream& os,
                         int64_t uid,
                         const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                         hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                         hipdnn_data_sdk::utilities::ITensor& expected,
                         hipdnn_data_sdk::utilities::ITensor& actual,
                         float atol,
                         float rtol);

    template <typename T>
    static void appendFpDiff(std::ostream& os,
                             int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             hipdnn_data_sdk::utilities::ITensor& actual,
                             float atol,
                             float rtol);

    // ── reporting ───────────────────────────────────────────────────────
    void skipUnverifiable(const std::string& reason);
    void recordRefError(const std::string& reason);
    static std::string refLabel(ReferenceExecutorType type);

    static std::string
        labelFor(int64_t uid,
                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);

    std::string reportHeader(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             float atol,
                             float rtol) const;

    static std::string dataTypeName(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

    // ── tolerances ──────────────────────────────────────────────────────
    static void
        resolveTolerances(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                          hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                          float& atol,
                          float& rtol);

    template <typename T>
    static float
        toleranceForNodeAttributes(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType);

    static float deriveDefaultTolerance(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

    static float
        toleranceForDataType(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

    // ── guards ──────────────────────────────────────────────────────────
    void applyMetadataGuards() const;
};

} // namespace hipdnn_integration_tests::golden

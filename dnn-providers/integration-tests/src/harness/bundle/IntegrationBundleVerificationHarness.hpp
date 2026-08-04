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
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/SupportClaimEnforcement.hpp"
#include "harness/bundle/SupportEnforcementReport.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests::bundle
{

// Output tensors, keyed by uid. Used both for the engine's computed "actual"
// outputs and for an expected source (golden from disk, or a reference executor's
// output). Each set is a distinct allocation so engine and reference never write
// the same buffers.
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
//
// TODO(ALMIOPEN-1969 follow-up): Unify graph-init with the non-golden harness.
//   Stage 1 — Route non-golden ops whose initializeBundle() is plain randomize
//             (conv, matmul, BN-inference, reduction, rmsnorm-fwd, layernorm,
//             pointwise) through the fill-inputs switch. Zero behavioral change.
//   Stage 2 — Migrate structured recipes one op at a time: copy the exact
//             ranges/seeds/derivation from each non-golden subclass override
//             into the corresponding fill function, using fillComputed/tensorAt
//             for derived inputs. Delete each override once its fill fn works.
//   Stage 3 — Both harnesses share one init pipeline via fillInputs().
class IntegrationBundleVerificationHarness : public ::testing::Test
{
public:
    explicit IntegrationBundleVerificationHarness(bool requiresDevice)
        : _requiresDevice(requiresDevice)
    {
    }

    void setBundle(std::shared_ptr<IntegrationTestBundle> bundle, std::filesystem::path path)
    {
        _bundle = std::move(bundle);
        _bundlePath = std::move(path);

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
        // RFC 0015 §6: the enforcement ladder is a stacking sequence of stop
        // points. `applicability`/`buildable` bundles stop before any input
        // generation, execution, or comparison (§6.1: never subject to golden
        // mode's no-.bin skip, since they never even reach runComparison()).
        // `full` (the default) runs the pre-existing comparison pipeline,
        // which independently enforces claims at its own engine-selection
        // point (see executeGraphThroughEngine).
        switch(_bundle->metadata.enforcementLevel)
        {
        case hipdnn_integration_tests::EnforcementLevel::Applicability:
            runApplicabilityLevel();
            return;
        case hipdnn_integration_tests::EnforcementLevel::Buildable:
            runBuildableLevel();
            return;
        case hipdnn_integration_tests::EnforcementLevel::Full:
            runComparison();
            return;
        default:
            FAIL() << "Unknown enforcement level";
            return;
        }
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

    // Skips the test when the bundle's metadata is incompatible with the
    // current device (VRAM/arch). Virtual so isolated unit tests that don't
    // exercise hardware guards can override it — production reads from the
    // TestConfig singleton, which is only initialized by the real test main.
    virtual void applyMetadataGuards() const;

    // Data-free applicability query (RFC 0015 §8): deserializes the graph
    // with no tensor data and calls get_ranked_engine_ids(). Bumps the
    // process-wide SupportQueryGuard (a query was observed) every call.
    // Overridable, like executeGraphThroughEngine, so unit tests can stub the
    // backend call.
    struct SupportQueryResult
    {
        hipdnn_frontend::Error status;
        std::vector<int64_t> rankedEngineIds;
    };
    virtual SupportQueryResult queryGraphSupport();

    // Data-free plan-compile check for the "buildable" rung (RFC 0015 §6,
    // §8): create_execution_plans -> check_support -> build_plans, no input
    // generation, no execution. Hard ASSERTs on failure, mirroring
    // executeGraphThroughEngine's existing (always-unconditional) plan-build
    // assertions. Overridable so unit tests can stub it.
    virtual void runPlanBuildOnly();

    // Every engine name currently loaded on the shared handle (RFC 0015
    // §7.3: support enforcement is multi-engine in every mode -- every
    // loaded engine is attributed independently from one query). Overridable
    // so unit tests can substitute a fixed engine list without a real
    // handle/plugin.
    virtual std::vector<std::string> listLoadedEngines() const;

    // Current device arch token (RFC 0015 §5.1: the gcnArchName prefix
    // before the first ':') and lowercase platform name ("linux"/"windows"),
    // used to evaluate support claims. Virtual, like getVerificationMode(),
    // so tests can inject values without depending on the TestConfig
    // singleton being initialized.
    virtual std::string currentArchToken() const;
    virtual std::string currentPlatform() const;

    InputFillRecipes& inputFillRecipes()
    {
        return _inputFillRecipes;
    }

private:
    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    std::shared_ptr<IntegrationTestBundle> _bundle;
    InputFillRecipes _inputFillRecipes;

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

    // ── enforcement ladder (RFC 0015 §6, §7, §8) ────────────────────────
    void runApplicabilityLevel();
    void runBuildableLevel();
    // Evaluates the bundle's support claims (if any) against one
    // already-observed support query outcome, shared by every rung: the
    // applicability/buildable ladder's own query, and full's existing
    // engine-selection query in executeGraphThroughEngine. `rung` names the
    // lowest broken rung in FAIL messages (RFC 0015 §6: "the failure is
    // attributed to the lowest rung that broke"). No-op (aside from the
    // caller's own query-observed bump) when the bundle carries no
    // support.json. Returns true iff a claim-broken or errored-before-assert
    // row was found (and FAILed) -- callers branch on this return value
    // directly rather than GTest's global HasFailure() state, so the
    // ladder's stop-early behavior does not depend on how a caller's result
    // reporter is wired (production run vs a test double capturing results).
    bool enforceSupportClaims(const char* rung,
                              const hipdnn_frontend::Error& status,
                              const std::vector<int64_t>& rankedEngineIds);

    // ── top-level dispatch ────────────────────────────────────────────────
    void runComparison();
    void runGoldenMode();
    void runExplicitRefMode(ReferenceExecutorType type);
    void runAutoMode();

    // ── inputs ──────────────────────────────────────────────────────────
    bool ensureInputsAvailable();

    // Fills leaf input tensors for the graph when no golden data exists.
    //
    // Phase 1 — allocate: walks the graph's tensor list, skips virtual
    //   (inter-node) and output tensors, allocates a CPU-side buffer for
    //   each remaining leaf input tensor (shape/dtype from TensorAttributes).
    //
    // Phase 2 — fill: calls fillInputs(), which registers each op's default
    //   fill recipes into _inputFillRecipes and then fills every leaf input
    //   as FREE (random values), STRUCTURED (needs specific format), or
    //   DERIVED (needs another op's output).
    //
    // Phase 3 — verify: checks _inputFillRecipes.unfilled() so that every
    //   leaf input was accounted for and none were refused (STRUCTURED/
    //   DERIVED). Returns false and SKIPs the test if any leaf was missed
    //   or refused.
    //
    // On success, moves the filled tensors into the bundle so downstream
    // executors (engine, GPU ref, CPU ref) can upload them to the GPU.
    bool fillBundleInputs();

    // ── buffer allocation + execution ───────────────────────────────────
    // allocateSentinelOutputs / buildVariantPack prepare the buffers;
    // runEngine* / runReference* call the executors and capture results.
    // Outputs are sentinel-filled (NaN) so an unwritten output element is
    // caught by allClose rather than masquerading as a computed zero.
    OutputTensors allocateSentinelOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const;
    // Runs the engine into fresh output buffers. Returns nullopt if the
    // engine threw (its message is written to `error`) or raised a fatal
    // GTest failure (in which case `error` is left empty).
    std::optional<OutputTensors> runEngineCapturingOutputs(std::string& error);

    // Runs the engine and returns its outputs, or nullopt if it could not
    // run. On nullopt the caller must simply return: this has already
    // issued the appropriate verdict (a fatal failure propagates as-is,
    // otherwise the test is SKIPped). Shared preamble for all three modes.
    std::optional<OutputTensors> runEngineOrSkip();

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
    // Records the bundle path + reason in the process-wide
    // UnverifiableBundleReport (printed as a summary after all tests),
    // then GTEST_SKIP()s this test. The reason is a flat human-readable
    // string — per-tensor details are concatenated into it by the caller
    // (e.g., fillBundleInputs()), not stored as structured data.
    void skipUnverifiable(const std::string& reason);
    void recordRefError(const std::string& reason);
    static std::string refLabel(ReferenceExecutorType type);

    static std::string
        labelFor(int64_t uid, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);

    std::string reportHeader(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             float atol,
                             float rtol) const;

    static std::string dataTypeName(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

    // ── tolerances ──────────────────────────────────────────────────────
    // Default derivation (max-across-nodes, per-op/per-dtype lookup) and the
    // TOML per-test override are shared with the graph harness via
    // harness/tolerance/ToleranceResolver.hpp (tolerance::resolveTolerance),
    // called directly from compareEach.
};

} // namespace hipdnn_integration_tests::bundle

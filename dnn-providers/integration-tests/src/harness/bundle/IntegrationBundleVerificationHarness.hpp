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
#include "harness/bundle/SupportClaims.hpp"
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
    explicit IntegrationBundleVerificationHarness(bool requiresDevice)
        : _requiresDevice(requiresDevice)
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
        runComparison();
        if(!HasFatalFailure() && !IsSkipped())
        {
            EXPECT_TRUE(_verified)
                << "test completed without verifying anything for " << _bundlePath;
        }
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
    virtual void enforceAtLevel(EnforcementLevel level);

    // Protected so a stubbed enforceAtLevel() can exit the same way the real one does
    // when it cannot verify: marks the bundle accounted for, then skips.
    void skipUnverifiable(const std::string& reason);

    InputFillRecipes& inputFillRecipes()
    {
        return _inputFillRecipes;
    }

private:
    bool _requiresDevice;
    mutable bool _verified = false;
    std::filesystem::path _bundlePath;
    SupportClaimLocator _claimLocator;
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

    void runComparison();
    void runGoldenMode();
    void runExplicitRefMode(ReferenceExecutorType type);
    void runAutoMode();
    void runGoldenCheckMode();

    bool ensureInputsAvailable();
    bool fillBundleInputs();

    OutputTensors allocateSentinelOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs,
                                                        bool useDevice) const;
    std::optional<OutputTensors> runEngineCapturingOutputs(std::string& error);
    std::optional<OutputTensors> runEngineOrSkip();

    RefRunResult runReferenceCapturingOutputs(ReferenceExecutorType type,
                                              OutputTensors& refOutputs);
    void markOutputsModified(OutputTensors& outputs) const;
    static void markOutputsModifiedFor(OutputTensors& outputs, bool device);

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

    void recordRefError(const std::string& reason);
    static std::string refLabel(ReferenceExecutorType type);

    static std::string
        labelFor(int64_t uid, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs);
};

} // namespace hipdnn_integration_tests::bundle

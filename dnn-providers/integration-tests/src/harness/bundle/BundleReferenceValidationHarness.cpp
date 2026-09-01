// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/BundleReferenceValidationHarness.hpp"

#include <string>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/IReferenceExecutors.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/bundle/ReferenceOpCoverage.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::bundle
{

IReferenceGraphExecutor& BundleReferenceValidationHarness::referenceExecutor() const
{
    return _referenceExecutors->get(_referenceType);
}

void BundleReferenceValidationHarness::SetUp()
{
    if(_requiresDevice)
    {
        SKIP_IF_NO_DEVICES();
    }

    ASSERT_NE(_bundle, nullptr) << "No bundle set";

    // Machine capability, not engine opinion: a bundle that wants more VRAM than
    // this card has, or an arch it was never meant for, cannot be validated here by
    // anyone. Checked before the no-skip contract, which is about verification.
    //
    // The TOML skip list is deliberately NOT consulted. It lives in an engine's own
    // config file, so it says which graphs that engine may sit out — no engine is
    // involved here, and our golden data does not get to opt out.
    if(auto reason
       = checkVramRequirement(_bundle->metadata, TestConfig::get().getCurrentDeviceVramMb()))
    {
        GTEST_SKIP() << *reason;
    }
    if(auto reason = checkArchCompatibility(_bundle->metadata, TestConfig::get().getCurrentArch()))
    {
        GTEST_SKIP() << *reason;
    }

    // Registration only creates a test when both hold, so a violation here is a
    // registration bug rather than a property of the data.
    ASSERT_TRUE(_bundle->hasGoldenOutputs)
        << "reference validation registered for a bundle with no golden data: " << _bundlePath;
    ASSERT_TRUE(_bundle->tensors.has_value())
        << "reference validation registered for a bundle with no tensor data: " << _bundlePath;
}

OutputTensors BundleReferenceValidationHarness::allocateOutputs() const
{
    auto wrapper = _bundle->graphWrapper();
    return detail::allocateSentinelOutputs(wrapper.getTensorMap(), _bundle->outputTensorUids);
}

// Only an executor that actually wants device pointers gets them; the enum a
// harness was registered with says nothing about what the executor it was
// actually handed needs.
bool BundleReferenceValidationHarness::useDevice() const
{
    return _requiresDevice && referenceExecutor().requiresDeviceMemory();
}

std::unordered_map<int64_t, void*>
    BundleReferenceValidationHarness::buildVariantPack(OutputTensors& outputs) const
{
    auto wrapper = _bundle->graphWrapper();
    return detail::buildVariantPack(
        *_bundle->tensors, outputs, wrapper.getTensorMap(), _bundle->outputTensorUids, useDevice());
}

void BundleReferenceValidationHarness::TestBody()
{
    auto referenceOutputs = allocateOutputs();
    auto variantPack = buildVariantPack(referenceOutputs);

    IReferenceGraphExecutor& executor = referenceExecutor();

    // No skip path by design. This bundle's node types are all inside this
    // reference's required-op set (ReferenceOpCoverage.hpp), so an inapplicable or
    // throwing reference is a gap in the reference, not a property of the bundle.
    try
    {
        ASSERT_TRUE(executor.isApplicable(_bundle->graphBuffer.data(), _bundle->graphBuffer.size()))
            << referenceLabel(_referenceType)
            << " is required to support this graph (its node types are in the reference's "
               "supported-op set) but reports it is not applicable: "
            << _bundlePath;

        executor.execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        FAIL() << referenceLabel(_referenceType)
               << " is required to support this graph but reported a capability miss: " << e.what()
               << "\n  bundle: " << _bundlePath;
    }
    catch(const std::exception& e)
    {
        FAIL() << referenceLabel(_referenceType) << " errored on " << _bundlePath << ": "
               << e.what();
    }

    // Tell each tensor which side now holds the fresh data, or the comparison reads
    // the stale copy.
    detail::markOutputsModified(referenceOutputs, useDevice());

    auto wrapper = _bundle->graphWrapper();

    // Golden data is the expectation here, and the reference output is what is being
    // judged — the opposite assignment from the engine harness, where golden is the
    // oracle. Same comparison either way, so it is the same code.
    const ExpectedTensorLookup goldenFor
        = [this](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *_bundle->tensors->at(uid);
    };

    // defaultTolerance(), never resolveTolerance(): a TOML override belongs to an
    // engine and must not loosen the gate on our own data.
    const auto toleranceFor = [&wrapper](hipdnn_flatbuffers_sdk::data_objects::DataType dataType) {
        const float value = tolerance::defaultTolerance(wrapper, dataType);
        return ComparisonTolerance{value, value};
    };

    const std::string contextLine = "Golden data validation ("
                                    + std::string(referenceLabel(_referenceType))
                                    + "): " + _bundlePath.string();

    for(const auto& mismatch : bundle::compareOutputs(wrapper,
                                                      _bundle->outputTensorUids,
                                                      referenceOutputs,
                                                      goldenFor,
                                                      toleranceFor,
                                                      contextLine))
    {
        ADD_FAILURE() << mismatch.report;
    }
}

} // namespace hipdnn_integration_tests::bundle

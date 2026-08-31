// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/BundleReferenceValidationHarness.hpp"

#include <sstream>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>

#include "harness/IReferenceExecutors.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/ReferenceOpCoverage.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::bundle
{

IReferenceGraphExecutor& BundleReferenceValidationHarness::referenceExecutor()
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
    const auto& tensorAttrMap = wrapper.getTensorMap();

    OutputTensors outputs;
    for(const int64_t uid : _bundle->outputTensorUids)
    {
        outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttrMap.at(uid));
        outputs[uid]->fillWithSentinelValue();
    }
    return outputs;
}

// The CPU reference reads and writes host memory; only the GPU one wants device
// pointers. Handing a CPU executor device pointers is a silent crash, not an error.
bool BundleReferenceValidationHarness::useDevice() const
{
    return _requiresDevice && _referenceType == ReferenceExecutorType::GPU;
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
    for(auto& [uid, tensor] : referenceOutputs)
    {
        static_cast<void>(uid);
        if(useDevice())
        {
            tensor->markDeviceModified();
        }
        else
        {
            tensor->markHostModified();
        }
    }

    auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    for(const int64_t uid : _bundle->outputTensorUids)
    {
        const auto* attrs = tensorAttrMap.at(uid);
        const auto dataType = attrs->data_type();

        float atol = 0.0f;
        float rtol = 0.0f;
        tolerance::resolveTolerance(wrapper, dataType, currentTestName(), atol, rtol);

        auto& computed = *referenceOutputs.at(uid);
        auto& golden = *_bundle->tensors->at(uid);

        auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, atol, rtol);
        if(validator->allClose(golden, computed))
        {
            continue;
        }

        const auto* name = attrs->name();
        const std::string label
            = (name != nullptr && !name->empty()) ? name->str() : ("uid=" + std::to_string(uid));

        hipdnn_test_sdk::utilities::ComparisonContext ctx;
        ctx.contextLine = "Golden data validation (" + std::string(referenceLabel(_referenceType))
                          + "): " + _bundlePath.string();
        ctx.tensorLabel = label + " (UID " + std::to_string(uid) + ", output)";
        ctx.dtypeName = hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
        ctx.atol = atol;
        ctx.rtol = rtol;

        std::ostringstream report;
        report << hipdnn_test_sdk::utilities::formatComparisonHeader(ctx, golden);
        hipdnn_test_sdk::utilities::appendComparisonDiffByDataType(
            report, dataType, label, golden, computed, atol, rtol);
        ADD_FAILURE() << report.str();
    }
}

} // namespace hipdnn_integration_tests::bundle

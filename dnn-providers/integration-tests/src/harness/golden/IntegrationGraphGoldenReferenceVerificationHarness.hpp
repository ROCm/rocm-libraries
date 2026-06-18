// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/TestConfig.hpp"
#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::golden
{

// Saved expected output tensors, keyed by output tensor UID. Extracted from a
// loaded bundle's output tensors just before execution: the harness keeps these
// as the golden reference and zeroes the live tensors so the runner computes
// into clean buffers.
using GoldenOutputs
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

class IntegrationGraphGoldenReferenceVerificationHarness : public ::testing::Test
{
public:
    using ExecuteFunc = std::function<void(IntegrationTestBundle&)>;

    IntegrationGraphGoldenReferenceVerificationHarness(ExecuteFunc executor, bool requiresDevice)
        : _executeFunc(std::move(executor))
        , _requiresDevice(requiresDevice)
    {
    }

    // The bundle is loaded once at registration time and shared into the test's
    // factory; the harness does not load from disk. The path is kept only for
    // diagnostic messages.
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

        // A graph-only bundle (no tensor data on disk, or .bin not pulled via
        // DVC) cannot be executed or compared -> SKIP.
        if(!_bundle->tensors.has_value())
        {
            GTEST_SKIP() << "Tensor data not available (graph-only bundle or DVC not pulled?): "
                         << _bundlePath;
        }

        applyMetadataGuards();
    }

    // Save each output tensor's loaded data as the golden reference, then zero
    // the live tensor so the runner computes into a clean buffer. Returns the
    // golden map keyed by output UID.
    GoldenOutputs extractGolden(TensorMap& tensorMap) const
    {
        GoldenOutputs golden;
        const auto wrapper = _bundle->graphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(const int64_t uid : _bundle->outputTensorUids)
        {
            const auto dataType = tensorAttrMap.at(uid)->data_type();
            auto& livePtr = tensorMap.at(uid);

            auto zeroed = std::visit(
                [&](auto nativeType) {
                    using DataType = decltype(nativeType);
                    auto tensorPtr = std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>(
                        new hipdnn_data_sdk::utilities::Tensor<DataType>(livePtr->dims(),
                                                                         livePtr->strides()));
                    tensorPtr->fillTensorWithValue(0.f);
                    return tensorPtr;
                },
                hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));

            std::swap(zeroed, livePtr); // live map now holds the zero buffer
            golden[uid] = std::move(zeroed); // golden holds the original data
        }
        return golden;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override
    {
        runGoldenComparison();
    }

private:
    ExecuteFunc _executeFunc;
    bool _requiresDevice;
    std::filesystem::path _bundlePath;
    std::shared_ptr<IntegrationTestBundle> _bundle;

    void runGoldenComparison()
    {
        // SetUp guarantees we only reach TestBody with tensor data present.
        auto& tensorMap = *_bundle->tensors;

        // A bundle with no output tensors has nothing to compare against —
        // executing it would only exercise the graph path. Treat as a skip.
        if(_bundle->outputTensorUids.empty())
        {
            GTEST_SKIP() << "Bundle has no output tensors to compare: " << _bundlePath;
        }

        // Save the loaded output values as golden and zero the live outputs so the
        // runner computes into clean buffers.
        const auto golden = extractGolden(tensorMap);

        // The executor runs the bundle in place (it reads the graph + input
        // tensors and writes the computed outputs back into the bundle's tensor
        // map). It signals "I cannot run this graph" by throwing (e.g. an op with
        // no GPU reference plan, or an engine that does not support the graph).
        // GTEST_SKIP() cannot be issued from inside the executor lambda because it
        // only returns from the lambda, not from TestBody — control would fall
        // through to the comparison below and fail zeroed outputs against golden
        // data. So the executor throws and the harness, which is inside TestBody,
        // translates "unsupported" into a real skip here.
        //
        // ASSERT_NO_FATAL_FAILURE still wraps the call so that a genuine GTest
        // assertion inside the executor (e.g. a build/execute error) FAILs and
        // stops the test rather than continuing into the comparison. It does not
        // catch C++ exceptions, which is why the try/catch is also required.
        bool executorThrew = false;
        std::string executorError;
        try
        {
            ASSERT_NO_FATAL_FAILURE(_executeFunc(*_bundle));
        }
        catch(const std::exception& e)
        {
            executorThrew = true;
            executorError = e.what();
        }

        if(executorThrew)
        {
            GTEST_SKIP() << "Executor could not run bundle " << _bundlePath << ": "
                         << executorError;
        }

        auto wrapper = _bundle->graphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(auto uid : _bundle->outputTensorUids)
        {
            auto& actualTensor = *tensorMap.at(uid);
            auto& expectedTensor = *golden.at(uid);

            auto* attrs = tensorAttrMap.at(uid);
            auto dataType = attrs->data_type();

            float atol = 0.0f;
            float rtol = 0.0f;
            resolveTolerances(wrapper, dataType, atol, rtol);

            compareOutputTensor(uid, *attrs, dataType, expectedTensor, actualTensor, atol, rtol);
        }
    }

    // Compare one output tensor against its golden reference and report mismatches.
    //
    // Floating-point dtypes go through computeTensorDiff(), which walks the tensor
    // once and returns a structured summary (mismatch count, worst-element index,
    // expected/actual/abs-diff) — that single pass IS the pass/fail decision and
    // the failure report, no second walk. Integer dtypes have no computeTensorDiff
    // specialization, so they keep the boolean allClose validator path.
    void compareOutputTensor(int64_t uid,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             hipdnn_data_sdk::utilities::ITensor& expected,
                             hipdnn_data_sdk::utilities::ITensor& actual,
                             float atol,
                             float rtol) const
    {
        using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
        using hipdnn_data_sdk::types::bfloat16;
        using hipdnn_data_sdk::types::half;

        switch(dataType)
        {
        case DT::FLOAT:
            compareFp<float>(uid, attrs, dataType, expected, actual, atol, rtol);
            return;
        case DT::HALF:
            compareFp<half>(uid, attrs, dataType, expected, actual, atol, rtol);
            return;
        case DT::BFLOAT16:
            compareFp<bfloat16>(uid, attrs, dataType, expected, actual, atol, rtol);
            return;
        case DT::DOUBLE:
            compareFp<double>(uid, attrs, dataType, expected, actual, atol, rtol);
            return;
        default:
            break;
        }

        // Integer (and any non-FP) dtypes: boolean validator fallback.
        auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, atol, rtol);
        EXPECT_TRUE(validator->allClose(expected, actual))
            << reportHeader(uid, attrs, dataType, expected, atol, rtol)
            << "  (no element-wise diff available for this data type)\n";
    }

    template <typename T>
    void compareFp(int64_t uid,
                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                   hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                   hipdnn_data_sdk::utilities::ITensor& expected,
                   hipdnn_data_sdk::utilities::ITensor& actual,
                   float atol,
                   float rtol) const
    {
        const auto summary
            = hipdnn_test_sdk::utilities::computeTensorDiff<T>(expected, actual, atol, rtol);

        const bool passed = summary.mismatchCount == 0;
        std::ostringstream report;
        if(!passed)
        {
            report << reportHeader(uid, attrs, dataType, expected, atol, rtol);
            hipdnn_test_sdk::utilities::printTensorDiffSummary(
                report, labelFor(uid, attrs), summary);
        }
        EXPECT_TRUE(passed) << report.str();
    }

    // The human-readable label for an output tensor: its name if it has one,
    // otherwise "uid=N".
    static std::string labelFor(int64_t uid,
                                const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
    {
        const auto* name = attrs.name();
        return (name != nullptr && !name->empty()) ? name->str() : ("uid=" + std::to_string(uid));
    }

    // Common header for a failed comparison (RFC 0011 §4.3 "What a failure looks
    // like"): bundle path, tensor UID/name, shape + dtype, and tolerance. The
    // per-element diff (worst index, expected/actual/abs-diff, mismatch count) is
    // appended by the caller from the TensorDiffSummary it already computed.
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

    // A bundle graph may fuse several ops (e.g. Convolution + Pointwise
    // activation). Each op type has its own numerical tolerance, so the only
    // tolerance that holds for the fused output is the loosest one across all
    // nodes: a tolerance tight enough for Conv (e.g. 1e-3) would wrongly fail an
    // activation output that legitimately needs 1e-2. We therefore take the max
    // tolerance over every node rather than picking a single "root" node.
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

    // Dispatch a single node's tolerance lookup on the bundle's data type.
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
        // metadata is mandatory, so a loaded bundle always has it (a bundle with
        // no .meta.json fails to load and never reaches here). Individual fields
        // (VRAM, arch) are still optional within BundleMetadata; the guards below
        // no-op when their field is absent, so they can be called unconditionally.
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

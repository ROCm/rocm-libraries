// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/BundleLoadCheck.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::golden
{

class IntegrationGraphGoldenReferenceVerificationHarness : public ::testing::Test
{
public:
    using ExecuteFunc = std::function<void(hipdnn_test_sdk::utilities::GraphAndTensorMap&)>;

    IntegrationGraphGoldenReferenceVerificationHarness(ExecuteFunc executor, bool requiresDevice)
        : _executeFunc(std::move(executor))
        , _requiresDevice(requiresDevice)
    {
    }

    void setBundlePath(std::filesystem::path path)
    {
        _bundlePath = std::move(path);
    }

    static void runReferenceExecutor(IReferenceGraphExecutor& executor,
                                     hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors)
    {
        const bool usesDevice = executor.requiresDeviceMemory();
        const auto variantPack
            = usesDevice ? deviceVariantPack(graphAndTensors) : graphAndTensors.hostBufferMap();

        executor.execute(
            graphAndTensors.graphBuffer.data(), graphAndTensors.graphBuffer.size(), variantPack);

        if(usesDevice)
        {
            for(const auto uid : graphAndTensors.outputTensorUids)
            {
                graphAndTensors.tensorMap.at(uid)->markDeviceModified();
            }
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

        if(_bundlePath.empty())
        {
            GTEST_SKIP() << "No bundle path set";
        }

        if(!std::filesystem::exists(_bundlePath))
        {
            GTEST_SKIP() << "Bundle file missing (DVC not pulled?): " << _bundlePath;
        }

        applyMetadataGuards();

        // A malformed graph .json is an authoring error and must FAIL, but
        // missing .bin tensor data is an environment issue (DVC not pulled)
        // and should SKIP. loadGraphAndTensors() throws std::exception for both,
        // so disambiguate up front with a single parse of the graph .json:
        // parse failure -> FAIL, referenced .bin files absent -> SKIP.
        const auto preload = checkBundlePreload(_bundlePath);
        if(!preload.graphJsonParses)
        {
            FAIL() << "Unparseable bundle graph JSON: " << _bundlePath;
        }
        if(!preload.tensorDataPresent)
        {
            GTEST_SKIP() << "Tensor data not available (DVC not pulled?): " << _bundlePath;
        }

        // Remaining loader failures (schema mismatch, tensor size mismatch) are
        // authoring/data errors, not missing data -> FAIL.
        try
        {
            _bundle = loadIntegrationTestBundle(_bundlePath);
        }
        catch(const std::exception& e)
        {
            FAIL() << "Failed to load bundle " << _bundlePath << ": " << e.what();
        }
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
    IntegrationTestBundle _bundle;

    void runGoldenComparison()
    {
        // The executor signals "I cannot run this graph" by throwing (e.g. an op
        // with no GPU reference plan, or an engine that does not support the
        // graph). GTEST_SKIP() cannot be issued from inside the executor lambda
        // because it only returns from the lambda, not from TestBody — control
        // would fall through to the comparison below and fail zeroed outputs
        // against golden data. So the executor throws and the harness, which is
        // inside TestBody, translates "unsupported" into a real skip here.
        //
        // ASSERT_NO_FATAL_FAILURE still wraps the call so that a genuine GTest
        // assertion inside the executor (e.g. a build/execute error) FAILs and
        // stops the test rather than continuing into the comparison. It does not
        // catch C++ exceptions, which is why the try/catch is also required.
        try
        {
            ASSERT_NO_FATAL_FAILURE(_executeFunc(_bundle.graphAndTensors));
        }
        catch(const std::exception& e)
        {
            GTEST_SKIP() << "Executor could not run bundle " << _bundlePath << ": " << e.what();
        }

        // A bundle without golden output data has nothing to compare against —
        // executing it above was the whole point (it exercises the graph path).
        if(!_bundle.goldenOutputs.has_value())
        {
            GTEST_SKIP() << "Bundle has no golden output data to compare: " << _bundlePath;
        }
        const auto& goldenOutputs = *_bundle.goldenOutputs;

        auto wrapper = _bundle.graphAndTensors.createGraphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(auto uid : _bundle.graphAndTensors.outputTensorUids)
        {
            auto& actualTensor = *_bundle.graphAndTensors.tensorMap.at(uid);
            auto& expectedTensor = *goldenOutputs.at(uid);

            auto* attrs = tensorAttrMap.at(uid);
            auto dataType = attrs->data_type();

            float atol = 0.0f;
            float rtol = 0.0f;
            resolveTolerances(wrapper, dataType, atol, rtol);

            auto validator
                = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, atol, rtol);
            EXPECT_TRUE(validator->allClose(expectedTensor, actualTensor)) << buildFailureReport(
                uid, *attrs, dataType, expectedTensor, actualTensor, atol, rtol);
        }
    }

    // Rich, developer-facing failure report (RFC 0011 §4.3 "What a failure looks
    // like"): bundle path, tensor UID/name, shape + dtype, max abs/rel error vs
    // tolerance, worst-element index with expected/actual, and mismatch count.
    std::string
        buildFailureReport(int64_t uid,
                           const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                           hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                           hipdnn_data_sdk::utilities::ITensor& expected,
                           hipdnn_data_sdk::utilities::ITensor& actual,
                           float atol,
                           float rtol) const
    {
        const auto* name = attrs.name();
        const std::string tensorLabel
            = (name != nullptr && !name->empty()) ? name->str() : ("uid=" + std::to_string(uid));

        std::ostringstream os;
        os << "\nGolden comparison FAILED\n"
           << "  Bundle: " << _bundlePath << "\n"
           << "  Tensor: " << tensorLabel << " (UID " << uid << ", output)\n"
           << "  Shape:  " << hipdnn_test_sdk::utilities::StreamVec(expected.dims()) << "  "
           << dataTypeName(dataType) << "\n"
           << "  Tolerance: atol=" << atol << " rtol=" << rtol << "\n";
        appendTensorDiff(os, dataType, tensorLabel, expected, actual, atol, rtol);
        return os.str();
    }

    static void appendTensorDiff(std::ostream& os,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                 const std::string& tensorLabel,
                                 hipdnn_data_sdk::utilities::ITensor& expected,
                                 hipdnn_data_sdk::utilities::ITensor& actual,
                                 float atol,
                                 float rtol)
    {
        using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
        using hipdnn_data_sdk::types::bfloat16;
        using hipdnn_data_sdk::types::half;
        namespace util = hipdnn_test_sdk::utilities;

        switch(dataType)
        {
        case DT::FLOAT:
            util::printTensorDiff<float>(os, tensorLabel, expected, actual, atol, rtol);
            break;
        case DT::HALF:
            util::printTensorDiff<half>(os, tensorLabel, expected, actual, atol, rtol);
            break;
        case DT::BFLOAT16:
            util::printTensorDiff<bfloat16>(os, tensorLabel, expected, actual, atol, rtol);
            break;
        case DT::DOUBLE:
            util::printTensorDiff<double>(os, tensorLabel, expected, actual, atol, rtol);
            break;
        default:
            os << "  (no element-wise diff available for this data type)\n";
            break;
        }
    }

    static std::string dataTypeName(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        return hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
    }

    static std::unordered_map<int64_t, void*>
        deviceVariantPack(hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors)
    {
        std::unordered_map<int64_t, void*> pack;
        for(auto& [uid, tensor] : graphAndTensors.tensorMap)
        {
            pack[uid] = tensor->rawDeviceData();
        }
        return pack;
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

    void applyMetadataGuards()
    {
        auto meta = hipdnn_test_sdk::utilities::loadBundleMetadata(_bundlePath);
        if(!meta.has_value())
        {
            return;
        }

        if(auto reason = hipdnn_test_sdk::utilities::checkVramRequirement(
               *meta, TestConfig::get().getCurrentDeviceVramMb()))
        {
            GTEST_SKIP() << *reason;
        }

        if(auto reason = hipdnn_test_sdk::utilities::checkArchCompatibility(
               *meta, TestConfig::get().getCurrentArch()))
        {
            GTEST_SKIP() << *reason;
        }
    }
};

} // namespace hipdnn_integration_tests::golden

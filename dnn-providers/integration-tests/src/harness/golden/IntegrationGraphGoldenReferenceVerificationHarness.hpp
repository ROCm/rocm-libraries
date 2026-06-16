// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/GoldenBundleDiscovery.hpp"

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

        try
        {
            _graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(_bundlePath);
        }
        catch(const std::exception& e)
        {
            GTEST_SKIP() << "Tensor data not available (DVC not pulled?): " << e.what();
        }
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
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
    hipdnn_test_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        _referenceOutputTensors;

    void runGoldenComparison()
    {
        ASSERT_NO_FATAL_FAILURE(_executeFunc(_graphAndTensors));

        auto wrapper = _graphAndTensors.createGraphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(auto uid : _graphAndTensors.outputTensorUids)
        {
            auto& actualTensor = *_graphAndTensors.tensorMap.at(uid);
            auto& expectedTensor = *_referenceOutputTensors.at(uid);

            auto* attrs = tensorAttrMap.at(uid);
            auto dataType = attrs->data_type();

            float atol = 0.0f;
            float rtol = 0.0f;
            resolveTolerances(wrapper, dataType, atol, rtol);

            auto validator
                = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, atol, rtol);
            ASSERT_TRUE(validator->allClose(expectedTensor, actualTensor))
                << "Mismatch in output tensor uid=" << uid << " for bundle " << _bundlePath;
        }
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

    static float deriveDefaultTolerance(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
        using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

        auto nodeCount = wrapper.nodeCount();
        NA rootAttrType = NA::NONE;
        for(uint32_t i = 0; i < nodeCount; ++i)
        {
            auto& node = wrapper.getNode(i);
            auto at = node.attributes_type();
            if(at != NA::PointwiseAttributes)
            {
                rootAttrType = at;
            }
        }

        if(rootAttrType == NA::NONE)
        {
            return 1e-3f;
        }

        using hipdnn_data_sdk::types::bfloat16;
        using hipdnn_data_sdk::types::half;

        switch(dataType)
        {
        case DT::FLOAT:
            return toleranceForNodeAttributes<float>(rootAttrType);
        case DT::HALF:
            return toleranceForNodeAttributes<half>(rootAttrType);
        case DT::BFLOAT16:
            return toleranceForNodeAttributes<bfloat16>(rootAttrType);
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Visitor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/GoldenBundleDiscovery.hpp"
#include "harness/golden/GoldenTensorComparator.hpp"

namespace hipdnn_integration_tests::golden
{

class IntegrationGraphGoldenReferenceVerificationHarness : public ::testing::Test
{
public:
    void setBundlePath(std::filesystem::path path)
    {
        _bundlePath = std::move(path);
    }

protected:
    std::filesystem::path _bundlePath;
    hipdnn_test_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        _referenceOutputTensors;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
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

    virtual void executeUnderTest(hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors) = 0;

    // GTest entry point. The base owns the load/execute/compare flow; the only
    // per-runner variation is executeUnderTest (CPU ref / GPU ref / engine).
    // This keeps the base abstract (executeUnderTest is pure) while making each
    // subclass a concrete, registrable test.
    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override
    {
        runGoldenComparison();
    }

    // Runs a reference-graph executor (the ALMIOPEN-1944 IReferenceGraphExecutor
    // port) against the loaded bundle. The interface drives host-vs-device
    // variant-pack selection via requiresDeviceMemory(), so the CPU and GPU
    // golden subclasses no longer hand-roll that branching. Device executors
    // write to GPU memory, so outputs are marked device-modified to trigger a
    // device-to-host sync when the comparator reads them.
    static void runReferenceExecutor(
        IReferenceGraphExecutor& executor,
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors)
    {
        const bool usesDevice = executor.requiresDeviceMemory();
        const auto variantPack
            = usesDevice ? deviceVariantPack(graphAndTensors) : graphAndTensors.hostBufferMap();

        executor.execute(graphAndTensors.graphBuffer.data(),
                         graphAndTensors.graphBuffer.size(),
                         variantPack);

        if(usesDevice)
        {
            for(const auto uid : graphAndTensors.outputTensorUids)
            {
                graphAndTensors.tensorMap.at(uid)->markDeviceModified();
            }
        }
    }

    void runGoldenComparison()
    {
        ASSERT_NO_FATAL_FAILURE(executeUnderTest(_graphAndTensors));

        auto wrapper = _graphAndTensors.createGraphWrapper();
        const auto& tensorAttrMap = wrapper.getTensorMap();

        for(auto uid : _graphAndTensors.outputTensorUids)
        {
            auto& actualTensor = *_graphAndTensors.tensorMap.at(uid);
            auto& expectedTensor = *_referenceOutputTensors.at(uid);

            auto* attrs = tensorAttrMap.at(uid);
            auto dataType = attrs->data_type();

            // Tolerance lookup is the per-operation default only. The TOML
            // override and bundle-metadata levels of the three-level chain
            // (RFC 0011 §4.3) are a separate story and intentionally not wired
            // here.
            float atol = 0.0f;
            float rtol = 0.0f;
            resolveTolerances(wrapper, dataType, atol, rtol);

            auto compareFunc = [&](auto typeTag) {
                using T = decltype(typeTag);
                return compareTensors<T>(expectedTensor, actualTensor, atol, rtol);
            };

            auto result = std::visit(
                hipdnn_data_sdk::utilities::Visitor{
                    compareFunc,
                    [](int) -> ComparisonResult {
                        ComparisonResult r;
                        r.passed = false;
                        return r;
                    }},
                hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));

            if(!result.passed)
            {
                std::string tensorName;
                if(attrs->name() != nullptr)
                {
                    tensorName = attrs->name()->str();
                }
                std::vector<int64_t> shape;
                if(attrs->dims() != nullptr)
                {
                    for(flatbuffers::uoffset_t i = 0; i < attrs->dims()->size(); ++i)
                    {
                        shape.push_back(attrs->dims()->Get(i));
                    }
                }
                auto dtypeStr = dataTypeToShortString(dataType);

                FAIL() << formatComparisonFailure(
                    _bundlePath, uid, tensorName, shape, dtypeStr, result);
            }
        }
    }

private:
    static std::unordered_map<int64_t, void*> deviceVariantPack(
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors)
    {
        std::unordered_map<int64_t, void*> pack;
        for(auto& [uid, tensor] : graphAndTensors.tensorMap)
        {
            pack[uid] = tensor->rawDeviceData();
        }
        return pack;
    }

    static void resolveTolerances(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
        float& atol,
        float& rtol)
    {
        const float defaultTolerance = deriveDefaultTolerance(wrapper, dataType);
        atol = defaultTolerance;
        rtol = defaultTolerance;
    }

    template <typename T>
    static float toleranceForNodeAttributes(
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
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

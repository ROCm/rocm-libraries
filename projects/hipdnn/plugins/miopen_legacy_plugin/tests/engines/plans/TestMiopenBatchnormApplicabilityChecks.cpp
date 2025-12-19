// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenBatchnormApplicabilityChecks.hpp"

using namespace miopen_legacy_plugin;

namespace
{

// ============================================================================
// Test Helper Utilities
// ============================================================================

class TensorAttributesBuilder
{
public:
    TensorAttributesBuilder(int64_t id)
        : _id(id)
        , _name("tensor_" + std::to_string(id))
    {
    }

    TensorAttributesBuilder& withDims(std::vector<int64_t> dims)
    {
        _dims = std::move(dims);
        return *this;
    }

    TensorAttributesBuilder& withStrides(std::vector<int64_t> strides)
    {
        _strides = std::move(strides);
        return *this;
    }

    TensorAttributesBuilder& withDataType(hipdnn_sdk::data_objects::DataType type)
    {
        _dataType = type;
        return *this;
    }

    TensorAttributesBuilder& withName(std::string name)
    {
        _name = std::move(name);
        return *this;
    }

    TensorAttributesBuilder& asVirtual()
    {
        _isVirtual = true;
        return *this;
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>
        build(flatbuffers::FlatBufferBuilder& builder)
    {
        return hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, _id, _name.c_str(), _dataType, &_strides, &_dims, _isVirtual);
    }

private:
    int64_t _id;
    std::string _name;
    hipdnn_sdk::data_objects::DataType _dataType{hipdnn_sdk::data_objects::DataType::FLOAT};
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    bool _isVirtual{false};
};

class TensorMapBuilder
{
public:
    TensorMapBuilder& addTensor(TensorAttributesBuilder tensorBuilder)
    {
        _tensorBuilders.push_back(std::move(tensorBuilder));
        return *this;
    }

    std::pair<flatbuffers::FlatBufferBuilder,
              std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>>
        build()
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorOffsets;
        tensorOffsets.reserve(_tensorBuilders.size());

        for(auto& tensorBuilder : _tensorBuilders)
        {
            tensorOffsets.push_back(tensorBuilder.build(builder));
        }

        auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(
            builder,
            "test_graph",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            hipdnn_sdk::data_objects::DataType::HALF,
            hipdnn_sdk::data_objects::DataType::BFLOAT16,
            &tensorOffsets,
            nullptr);

        builder.Finish(graphOffset);

        const auto* graph = hipdnn_sdk::data_objects::GetGraph(builder.GetBufferPointer());
        std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*> tensorMap;

        if(graph->tensors() != nullptr)
        {
            for(const auto* tensorAttr : *graph->tensors())
            {
                tensorMap[tensorAttr->uid()] = tensorAttr;
            }
        }

        return {std::move(builder), std::move(tensorMap)};
    }

private:
    std::vector<TensorAttributesBuilder> _tensorBuilders;
};

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class TestMiopenBatchnormApplicabilityChecks : public ::testing::Test
{
protected:
    // Common setup if needed
};

// ============================================================================
// validators::validateDimensionCount Tests
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateDimensionCountAccepts4DTensor)
{
    EXPECT_NO_THROW({ validators::validateDimensionCount(4); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateDimensionCountAccepts5DTensor)
{
    EXPECT_NO_THROW({ validators::validateDimensionCount(5); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateDimensionCountThrowsFor3DTensor)
{
    EXPECT_THROW({ validators::validateDimensionCount(3); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateDimensionCountThrowsFor6DTensor)
{
    EXPECT_THROW({ validators::validateDimensionCount(6); }, hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// validators::validateConsistentDimensions Tests
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateConsistentDimensionsAcceptsSameDimensions)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({2, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({2, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .build();

    std::vector<BatchnormTensorDescriptor> tensors;
    for(const auto& [id, attr] : tensorMap)
    {
        tensors.emplace_back(attr);
    }

    EXPECT_NO_THROW({ validators::validateConsistentDimensions(tensors); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateConsistentDimensionsThrowsForMixed4DAnd5D)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({2, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({2, 3, 16, 224, 224})
                                                   .withStrides({2408448, 802816, 50176, 224, 1}))
                                    .build();

    std::vector<BatchnormTensorDescriptor> tensors;
    for(const auto& [id, attr] : tensorMap)
    {
        tensors.emplace_back(attr);
    }

    EXPECT_THROW(
        { validators::validateConsistentDimensions(tensors); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateConsistentDimensionsAcceptsEmptyVector)
{
    std::vector<BatchnormTensorDescriptor> tensors;
    EXPECT_NO_THROW({ validators::validateConsistentDimensions(tensors); });
}

// ============================================================================
// validators::validatePackedTensors Tests
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidatePackedTensorsAcceptsPackedTensor)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({2, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .build();

    std::vector<BatchnormTensorDescriptor> tensors;
    for(const auto& [id, attr] : tensorMap)
    {
        tensors.emplace_back(attr);
    }

    EXPECT_NO_THROW({ validators::validatePackedTensors(tensors); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidatePackedTensorsThrowsForNonPackedTensor)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({2, 3, 224, 224})
                                                   .withStrides({200000, 60000, 250, 1}))
                                    .build();

    std::vector<BatchnormTensorDescriptor> tensors;
    for(const auto& [id, attr] : tensorMap)
    {
        tensors.emplace_back(attr);
    }

    EXPECT_THROW(
        { validators::validatePackedTensors(tensors); }, hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// validators::validateSupportedLayout Tests
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateSupportedLayoutAcceptsNchwLayout)
{
    std::vector<int64_t> nchwOrder = {3, 2, 1, 0}; // NCHW
    EXPECT_NO_THROW({ validators::validateSupportedLayout(nchwOrder, 4); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateSupportedLayoutAcceptsNhwcLayout)
{
    std::vector<int64_t> nhwcOrder = {3, 0, 2, 1}; // NHWC
    EXPECT_NO_THROW({ validators::validateSupportedLayout(nhwcOrder, 4); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, ValidateSupportedLayoutThrowsForUnsupported4DLayout)
{
    std::vector<int64_t> invalidOrder = {0, 1, 2, 3}; // Invalid stride order
    EXPECT_THROW(
        { validators::validateSupportedLayout(invalidOrder, 4); },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Component Validator Tests - checkTensorLayoutsAndDimsSupported
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorLayoutsAndDimsAcceptsValid4DTensors)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .build();

    EXPECT_NO_THROW({ checkTensorLayoutsAndDimsSupported(tensorMap); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorLayoutsAndDimsAcceptsValid5DTensors)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({1, 3, 16, 224, 224})
                                                   .withStrides({2408448, 802816, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({1, 3, 16, 224, 224})
                                                   .withStrides({2408448, 802816, 50176, 224, 1}))
                                    .build();

    EXPECT_NO_THROW({ checkTensorLayoutsAndDimsSupported(tensorMap); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorLayoutsAndDimsRejectsMixedDimensions)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({1, 3, 16, 224, 224})
                                                   .withStrides({2408448, 802816, 50176, 224, 1}))
                                    .build();

    EXPECT_THROW(
        { checkTensorLayoutsAndDimsSupported(tensorMap); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorLayoutsAndDimsRejectsNonPackedTensors)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({200000, 60000, 250, 1})) // Non-packed
              .build();

    EXPECT_THROW(
        { checkTensorLayoutsAndDimsSupported(tensorMap); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorLayoutsAndDimsRejectsMixedLayouts)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1})) // NCHW
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 1, 672, 3})) // NHWC
                                    .build();

    EXPECT_THROW(
        { checkTensorLayoutsAndDimsSupported(tensorMap); }, hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Component Validator Tests - checkTensorDataTypesSupported
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorDataTypesAcceptsValidFloatTypes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(3)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(4)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(5)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3, 4};
    std::vector<int64_t> statTensorIds = {5};

    EXPECT_NO_THROW(
        { checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorDataTypesAcceptsValidHalfTypes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::HALF))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::HALF))
              .addTensor(TensorAttributesBuilder(3)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(4)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(5)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3, 4};
    std::vector<int64_t> statTensorIds = {5};

    EXPECT_NO_THROW(
        { checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorDataTypesRejectsInvalidIoType)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::UINT8))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::UINT8))
              .addTensor(TensorAttributesBuilder(3)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        { checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorDataTypesRejectsInvalidAffineType)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::HALF))
              .build();

    std::vector<int64_t> ioTensorIds = {1};
    std::vector<int64_t> affineTensorIds = {2};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        { checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorDataTypesRejectsInconsistentIoTypes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::HALF))
              .addTensor(TensorAttributesBuilder(3)
                             .withDims({1, 3, 1, 1})
                             .withStrides({3, 1, 1, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::FLOAT))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        { checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Component Validator Tests - checkTensorShapesSupported
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorShapesAcceptsValidInferenceShapes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({2, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({2, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3, 4};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_NO_THROW({
        checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
    });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorShapesAcceptsValidTrainingShapes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({2, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({2, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_NO_THROW({
        checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
    });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorShapesRejectsInconsistentIoShapes)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 112, 112})
                             .withStrides({37632, 12544, 112, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {3};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        {
            checkTensorShapesSupported(
                ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
        },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorShapesRejectsInvalidDerivedShape)
{
    auto [builder, tensorMap] = TensorMapBuilder()
                                    .addTensor(TensorAttributesBuilder(1)
                                                   .withDims({1, 3, 224, 224})
                                                   .withStrides({150528, 50176, 224, 1}))
                                    .addTensor(TensorAttributesBuilder(2)
                                                   .withDims({1, 5, 1, 1}) // Wrong channel count
                                                   .withStrides({5, 1, 1, 1}))
                                    .build();

    std::vector<int64_t> ioTensorIds = {1};
    std::vector<int64_t> affineTensorIds = {2};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        {
            checkTensorShapesSupported(
                ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
        },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorShapesRejectsInsufficientSpatialForTraining)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(
                  TensorAttributesBuilder(1).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(2).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    std::vector<int64_t> ioTensorIds = {1, 2};
    std::vector<int64_t> affineTensorIds = {};
    std::vector<int64_t> statTensorIds = {};

    EXPECT_THROW(
        {
            checkTensorShapesSupported(
                ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
        },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Activation Mode Validation Tests
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationAcceptsIdentity)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::IDENTITY,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationAcceptsRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationAcceptsClippedRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::Optional<double>(6.0), // relu_upper_clip = 6.0 for clipped ReLU
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationAcceptsClamp)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::Optional<double>(0.0), // relu_lower_clip = 0.0
        flatbuffers::Optional<double>(6.0), // relu_upper_clip = 6.0 for CLAMP
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, BwdActivationAcceptsRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW({ checkBatchnormBwdActivationModeSupported(*attr); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationThrowsForLeakyRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::Optional<double>(0.01), // relu_lower_clip_slope = 0.01 for leaky ReLU
        flatbuffers::nullopt, // axis_tensor_uid
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormFwdActivationModeSupported(*attr); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, BwdActivationThrowsForLeakyRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::Optional<double>(0.01), // relu_lower_clip_slope = 0.01 for leaky ReLU
        flatbuffers::nullopt, // axis_tensor_uid
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormBwdActivationModeSupported(*attr); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationThrowsForSigmoid)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormFwdActivationModeSupported(*attr); }, hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FwdActivationThrowsForTanh)
{
    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormFwdActivationModeSupported(*attr); }, hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Integration Tests - Full Configuration Validation
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorConfigAcceptsValidInference)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorConfigAcceptsValidTraining)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorConfigAcceptsValidBackward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_NO_THROW({ checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); });
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TensorConfigAcceptsValidFusedBackwardActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    EXPECT_NO_THROW({
        checkBatchnormTensorConfigSupported(
            *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
    });
}

// ============================================================================
// Unhappy Path Tests - Inference
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, InferenceRejectsInvalidIoDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_sdk::data_objects::DataType::UINT8); // Invalid IO data type
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, InferenceRejectsNonPackedTensor)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({200000, 60000, 250, 1})) // Non-packed
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(5).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnInfAttr = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(attrBuilder,
                                                                                  1, // x
                                                                                  5, // mean
                                                                                  6, // inv_variance
                                                                                  3, // scale
                                                                                  4, // bias
                                                                                  2 // y
    );
    attrBuilder.Finish(bnInfAttr);

    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>(
            attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, InferenceRejectsInvalidLayout)
{
    // Invalid stride order (not NCHW or NHWC)
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({1, 224, 672, 150528})) // Invalid order
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 224, 224})
                             .withStrides({1, 224, 672, 150528}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(5).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnInfAttr = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(attrBuilder,
                                                                                  1, // x
                                                                                  5, // mean
                                                                                  6, // inv_variance
                                                                                  3, // scale
                                                                                  4, // bias
                                                                                  2 // y
    );
    attrBuilder.Finish(bnInfAttr);

    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>(
            attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, InferenceRejectsMixedDimensions)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1}))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 16, 224, 224}) // 5D instead of 4D
                             .withStrides({2408448, 802816, 50176, 224, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(5).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnInfAttr = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(attrBuilder,
                                                                                  1, // x
                                                                                  5, // mean
                                                                                  6, // inv_variance
                                                                                  3, // scale
                                                                                  4, // bias
                                                                                  2 // y
    );
    attrBuilder.Finish(bnInfAttr);

    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>(
            attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Unhappy Path Tests - Forward Training
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, TrainingRejectsInvalidIoDataType)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 14, 14})
                             .withStrides({588, 196, 14, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::UINT8)) // Invalid
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 14, 14})
                             .withStrides({588, 196, 14, 1})
                             .withDataType(hipdnn_sdk::data_objects::DataType::UINT8))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(7).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnAttr = hipdnn_sdk::data_objects::CreateBatchnormAttributes(
        attrBuilder,
        1, // x
        3, // scale
        4, // bias
        0, // epsilon (not used in validation)
        0, // peer_stats
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2, // y
        flatbuffers::Optional<int64_t>(6), // mean
        flatbuffers::Optional<int64_t>(7), // inv_variance
        flatbuffers::nullopt,
        flatbuffers::nullopt);
    attrBuilder.Finish(bnAttr);

    const auto* attrs = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(
        attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TrainingRejectsInvalidAffineDataType)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 14, 14})
                             .withStrides({588, 196, 14, 1}))
              .addTensor(TensorAttributesBuilder(2)
                             .withDims({1, 3, 14, 14})
                             .withStrides({588, 196, 14, 1}))
              .addTensor(
                  TensorAttributesBuilder(3)
                      .withDims({1, 3, 1, 1})
                      .withStrides({3, 1, 1, 1})
                      .withDataType(
                          hipdnn_sdk::data_objects::DataType::HALF)) // Invalid - must be FLOAT
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(7).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnAttr = hipdnn_sdk::data_objects::CreateBatchnormAttributes(
        attrBuilder,
        1, // x
        3, // scale
        4, // bias
        0, // epsilon
        0, // peer_stats
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2, // y
        flatbuffers::Optional<int64_t>(6), // mean
        flatbuffers::Optional<int64_t>(7), // inv_variance
        flatbuffers::nullopt,
        flatbuffers::nullopt);
    attrBuilder.Finish(bnAttr);

    const auto* attrs = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(
        attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, TrainingRejectsInsufficientSpatialDimensions)
{
    // Batch size * spatial size must be > 1
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 1, 1}) // Spatial = 1x1, batch = 1
                             .withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(2).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(7).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnAttr = hipdnn_sdk::data_objects::CreateBatchnormAttributes(
        attrBuilder,
        1, // x
        3, // scale
        4, // bias
        0, // epsilon
        0, // peer_stats
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2, // y
        flatbuffers::Optional<int64_t>(6), // mean
        flatbuffers::Optional<int64_t>(7), // inv_variance
        flatbuffers::nullopt,
        flatbuffers::nullopt);
    attrBuilder.Finish(bnAttr);

    const auto* attrs = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(
        attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Unhappy Path Tests - Backward
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, BackwardRejectsInvalidIoDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_sdk::data_objects::DataType::UINT8); // Invalid IO data type
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, BackwardRejectsInvalidScaleBiasDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_sdk::data_objects::DataType::FLOAT, // Valid IO
        hipdnn_sdk::data_objects::DataType::HALF); // Invalid scale/bias - must be FLOAT
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, BackwardRejectsInvalidStatDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_sdk::data_objects::DataType::FLOAT, // Valid IO
        hipdnn_sdk::data_objects::DataType::FLOAT, // Valid scale/bias
        hipdnn_sdk::data_objects::DataType::HALF); // Invalid mean/variance - must be FLOAT
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, BackwardRejectsInsufficientSpatialDimensions)
{
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(
                  TensorAttributesBuilder(1).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(2).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(5).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(6).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(7).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(8).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .build();

    flatbuffers::FlatBufferBuilder attrBuilder;
    auto bnBwdAttr = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
        attrBuilder,
        2, // dy
        1, // x
        flatbuffers::Optional<int64_t>(7), // mean
        flatbuffers::Optional<int64_t>(8), // inv_variance
        4, // scale
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats
        3, // dx
        5, // dscale
        6 // dbias
    );
    attrBuilder.Finish(bnBwdAttr);

    const auto* attrs = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>(
        attrBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*attrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

// ============================================================================
// Unhappy Path Tests - Fused Backward with Activation
// ============================================================================

TEST_F(TestMiopenBatchnormApplicabilityChecks, FusedBackwardRejectsInvalidIoDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_sdk::data_objects::DataType::UINT8); // Invalid IO data type
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    EXPECT_THROW(
        {
            checkBatchnormTensorConfigSupported(
                *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
        },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FusedBackwardRejectsNonPackedTensor)
{
    // Create a valid graph then modify one tensor to be non-packed
    auto [builder, tensorMap]
        = TensorMapBuilder()
              .addTensor(TensorAttributesBuilder(1)
                             .withDims({1, 3, 224, 224})
                             .withStrides({200000, 60000, 250, 1})) // Non-packed x
              .addTensor(
                  TensorAttributesBuilder(2).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(3).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(4).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(5).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(TensorAttributesBuilder(6)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})) // dy
              .addTensor(TensorAttributesBuilder(7)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})) // dx
              .addTensor(
                  TensorAttributesBuilder(8).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(
                  TensorAttributesBuilder(9).withDims({1, 3, 1, 1}).withStrides({3, 1, 1, 1}))
              .addTensor(TensorAttributesBuilder(10)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .asVirtual())
              .addTensor(TensorAttributesBuilder(11)
                             .withDims({1, 3, 224, 224})
                             .withStrides({150528, 50176, 224, 1})
                             .asVirtual())
              .build();

    // Need 3 separate builders since each can only have one finished root
    flatbuffers::FlatBufferBuilder bnInfBuilder;
    auto bnInfAttrOffset = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
        bnInfBuilder, 1, 4, 5, 2, 3, 10);
    bnInfBuilder.Finish(bnInfAttrOffset);
    const auto* bnInfAttrs
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>(
            bnInfBuilder.GetBufferPointer());

    flatbuffers::FlatBufferBuilder actBuilder;
    auto actAttrOffset = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        actBuilder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        10,
        flatbuffers::Optional<int64_t>(6),
        flatbuffers::nullopt,
        11);
    actBuilder.Finish(actAttrOffset);
    const auto* actAttrs = flatbuffers::GetRoot<hipdnn_sdk::data_objects::PointwiseAttributes>(
        actBuilder.GetBufferPointer());

    flatbuffers::FlatBufferBuilder bnBwdBuilder;
    auto bnBwdAttrOffset = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
        bnBwdBuilder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    bnBwdBuilder.Finish(bnBwdAttrOffset);
    const auto* bnBwdAttrs
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>(
            bnBwdBuilder.GetBufferPointer());

    EXPECT_THROW(
        { checkBatchnormTensorConfigSupported(*bnInfAttrs, *actAttrs, *bnBwdAttrs, tensorMap); },
        hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestMiopenBatchnormApplicabilityChecks, FusedBackwardRejectsInsufficientSpatialDimensions)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph(
        {3, 1, 1, 1}, {1, 3, 1, 1}); // Batch * spatial = 1 * 1 = 1 (invalid)
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    EXPECT_THROW(
        {
            checkBatchnormTensorConfigSupported(
                *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
        },
        hipdnn_plugin::HipdnnPluginException);
}

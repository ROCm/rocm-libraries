// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestBatchnormApplicability.hpp"
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_plugin;
using hipdnn_data_sdk::utilities::TensorLayout;

// --- Test Case Structs: Layer 1 (Atomic Validators) ---

struct DimensionCountTestCase
{
    std::string name;
    bool shouldPass;
    size_t numDims;

    friend std::ostream& operator<<(std::ostream& os, const DimensionCountTestCase& tc)
    {
        return os << tc.name;
    }
};

struct SupportedLayoutTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<int64_t> strideOrder;
    size_t numDims;

    friend std::ostream& operator<<(std::ostream& os, const SupportedLayoutTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorDescriptorListTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<std::vector<int64_t>> tensorDims;
    std::vector<std::vector<int64_t>> tensorStrides;

    friend std::ostream& operator<<(std::ostream& os, const TensorDescriptorListTestCase& tc)
    {
        return os << tc.name;
    }
};

struct DataTypeIsSupportedTestCase
{
    std::string name;
    bool shouldPass;
    hipdnn_data_sdk::data_objects::DataType dataType;
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedTypes;

    friend std::ostream& operator<<(std::ostream& os, const DataTypeIsSupportedTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ConsistentDataTypesTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> allowedTypes;

    friend std::ostream& operator<<(std::ostream& os, const ConsistentDataTypesTestCase& tc)
    {
        return os << tc.name;
    }
};

struct FixedDataTypeTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    hipdnn_data_sdk::data_objects::DataType expectedType;

    friend std::ostream& operator<<(std::ostream& os, const FixedDataTypeTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ConsistentShapesTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> tensorIds;
    std::vector<int64_t> referenceShape;

    friend std::ostream& operator<<(std::ostream& os, const ConsistentShapesTestCase& tc)
    {
        return os << tc.name;
    }
};

struct SpatialDimensionsTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<int64_t> ioDims;

    friend std::ostream& operator<<(std::ostream& os, const SpatialDimensionsTestCase& tc)
    {
        return os << tc.name;
    }
};

// --- Test Data Providers: Layer 1 (Atomic Validators) ---

// Only 4D and 5D tensors work with batchnorm
inline std::vector<DimensionCountTestCase> getValidateDimensionCountTestCases()
{
    return {
        // Happy paths - supported dimensions
        {"Accepts4D", true, 4},
        {"Accepts5D", true, 5},

        // Unhappy paths - unsupported dimensions
        {"Rejects3D", false, 3},
        {"Rejects6D", false, 6},
        {"Rejects2D", false, 2},
        {"Rejects1D", false, 1},
    };
}

// Only NCHW/NHWC (4D) and NCDHW/NDHWC (5D) layouts are supported
inline std::vector<SupportedLayoutTestCase> getValidateSupportedLayoutTestCases()
{
    return {
        // Happy paths - 4D supported layouts
        {"AcceptsNchw4D", true, {3, 2, 1, 0}, 4}, // NCHW stride order
        {"AcceptsNhwc4D", true, {3, 0, 2, 1}, 4}, // NHWC stride order

        // Happy paths - 5D supported layouts
        {"AcceptsNcdhw5D", true, {4, 3, 2, 1, 0}, 5}, // NCDHW stride order
        {"AcceptsNdhwc5D", true, {4, 0, 3, 2, 1}, 5}, // NDHWC stride order

        // Unhappy paths - unsupported 4D layouts
        {"RejectsInvalid4D_AllReversed", false, {0, 1, 2, 3}, 4},
        {"RejectsInvalid4D_Random", false, {2, 1, 0, 3}, 4},

        // Unhappy paths - unsupported 5D layouts
        {"RejectsInvalid5D_AllReversed", false, {0, 1, 2, 3, 4}, 5},
        {"RejectsInvalid5D_Random", false, {1, 2, 3, 4, 0}, 5},
    };
}

// All tensors must have the same number of dimensions
inline std::vector<TensorDescriptorListTestCase> getValidateConsistentDimensionsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto strides4D
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto strides5D
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);

    return {
        // Happy paths - consistent dimensions
        {"AcceptsSame4D_TwoTensors", true, {dims4D, dims4D}, {strides4D, strides4D}},
        {"AcceptsSame5D_ThreeTensors",
         true,
         {dims5D, dims5D, dims5D},
         {strides5D, strides5D, strides5D}},
        {"AcceptsEmpty", true, {}, {}},
        {"AcceptsSingleTensor", true, {dims4D}, {strides4D}},

        // Unhappy paths - inconsistent dimensions
        {"RejectsMixed4D5D", false, {dims4D, dims5D}, {strides4D, strides5D}},
        {"RejectsMixed5D4D", false, {dims5D, dims4D}, {strides5D, strides4D}},
    };
}

// All tensors must be packed (contiguous in memory)
inline std::vector<TensorDescriptorListTestCase> getValidatePackedTensorsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NHWC.strideOrder);
    auto ncdhwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);

    return {
        // Happy paths - packed tensors
        {"AcceptsPacked4D_NCHW", true, {dims4D}, {nchwStrides}},
        {"AcceptsPacked4D_NHWC", true, {dims4D}, {nhwcStrides}},
        {"AcceptsPacked5D_NCDHW", true, {dims5D}, {ncdhwStrides}},
        {"AcceptsMultiplePacked", true, {dims4D, dims4D}, {nchwStrides, nhwcStrides}},

        // Unhappy paths - non-packed tensors
        {"RejectsNonPacked_ExtraStride", false, {dims4D}, {{200000, 60000, 250, 1}}},
        {"RejectsNonPacked_Gaps", false, {dims4D}, {{151000, 50200, 225, 1}}},
        {"RejectsOneNonPacked", false, {dims4D, dims4D}, {nchwStrides, {200000, 60000, 250, 1}}},
    };
}

// All tensors must have the same layout
inline std::vector<TensorDescriptorListTestCase> getValidateConsistentLayoutsTestCases()
{
    using namespace canonical_layouts;

    auto dims4D = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto dims5D = shapes::INFERENCE_5D[0]; // {1, 3, 16, 224, 224}
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims4D, TensorLayout::NHWC.strideOrder);
    auto ncdhwStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NCDHW.strideOrder);
    auto ndhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(dims5D, TensorLayout::NDHWC.strideOrder);
    auto degenerateStrides = hipdnn_data_sdk::utilities::generateStrides(
        shapes::DEGENERATE_4D, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - consistent layouts
        {"AcceptsSameNchw", true, {dims4D, dims4D}, {nchwStrides, nchwStrides}},
        {"AcceptsSameNhwc", true, {dims4D, dims4D}, {nhwcStrides, nhwcStrides}},
        {"AcceptsSameNcdhw", true, {dims5D, dims5D}, {ncdhwStrides, ncdhwStrides}},
        {"AcceptsWithDegenerate",
         true, // Degenerate tensors (all dims=1) are layout-agnostic
         {dims4D, shapes::DEGENERATE_4D},
         {nchwStrides, degenerateStrides}},

        // Unhappy paths - inconsistent layouts
        {"RejectsMixedNchwNhwc", false, {dims4D, dims4D}, {nchwStrides, nhwcStrides}},
        {"RejectsMixedNcdhwNdhwc", false, {dims5D, dims5D}, {ncdhwStrides, ndhwcStrides}},
    };
}

inline std::vector<DataTypeIsSupportedTestCase> getValidateDataTypeIsSupportedTestCases()
{
    using DT = hipdnn_data_sdk::data_objects::DataType;
    std::unordered_set<DT> ioTypes = {DT::FLOAT, DT::HALF, DT::BFLOAT16};

    return {
        // Happy paths - supported types
        {"AcceptsFloat", true, DT::FLOAT, ioTypes},
        {"AcceptsHalf", true, DT::HALF, ioTypes},
        {"AcceptsBfloat16", true, DT::BFLOAT16, ioTypes},

        // Unhappy paths - unsupported types
        {"RejectsUint8", false, DT::UINT8, ioTypes},
        {"RejectsWithEmptyAllowedList", false, DT::FLOAT, {}}, // Edge case
    };
}

// Training requires B × S > 1 (batch × spatial product)
inline std::vector<SpatialDimensionsTestCase> getValidateSpatialDimensionsTestCases()
{
    using namespace canonical_layouts;

    return {
        // Happy paths - sufficient spatial dimensions (B × S > 1)
        {"AcceptsSufficientSpatial4D_LargeBatch",
         true,
         shapes::INFERENCE_4D[3]}, // {2, 3, 224, 224}
        {"AcceptsSufficientSpatial5D", true, shapes::INFERENCE_5D[0]}, // {1, 3, 16, 224, 224}
        {"AcceptsBatch2Spatial1", true, {2, 3, 1, 1}}, // B=2, S=1 → B×S=2 > 1
        {"AcceptsBatch1Spatial2", true, {1, 3, 2, 1}}, // B=1, S=2 → B×S=2 > 1
        {"AcceptsLargeSpatial", true, {1, 3, 512, 512}},

        // Unhappy paths - insufficient spatial dimensions (B × S ≤ 1)
        {"RejectsBatch1Spatial1_4D", false, shapes::INSUFFICIENT_SPATIAL_4D}, // {1, 3, 1, 1}
        {"RejectsBatch1Spatial1_5D", false, {1, 3, 1, 1, 1}}, // B=1, S=1 → B×S=1 ≤ 1
        {"RejectsZeroBatch", false, {0, 3, 224, 224}}, // B=0 → B×S=0 ≤ 1
        {"RejectsZeroSpatial", false, {2, 3, 0, 0}}, // S=0 → B×S=0 ≤ 1
    };
}

// Validates a group of tensors all have the same data type from allowed list
inline std::vector<ConsistentDataTypesTestCase> getValidateConsistentDataTypesTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    const auto& testDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto testStrides
        = hipdnn_data_sdk::utilities::generateStrides(testDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - consistent types
        {"AcceptsConsistentFloat",
         true,
         {createIoTensor(1, "float_tensor_a", DT::FLOAT, testDims, testStrides),
          createIoTensor(2, "float_tensor_b", DT::FLOAT, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsConsistentHalf",
         true,
         {createIoTensor(1, "half_tensor_a", DT::HALF, testDims, testStrides),
          createIoTensor(2, "half_tensor_b", DT::HALF, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsSingleTensor",
         true,
         {createIoTensor(1, "single_tensor", DT::FLOAT, testDims, testStrides)},
         {1},
         bn_type_configs::getAllowedIoTypes()},
        {"AcceptsEmptyTensorList", true, {}, {}, bn_type_configs::getAllowedIoTypes()},

        // Unhappy paths - inconsistent types or unsupported types
        {"RejectsInconsistentTypes_FloatHalf",
         false,
         {createIoTensor(1, "float_tensor", DT::FLOAT, testDims, testStrides),
          createIoTensor(2, "half_tensor", DT::HALF, testDims, testStrides)},
         {1, 2},
         bn_type_configs::getAllowedIoTypes()},
        {"RejectsUnsupportedType",
         false,
         {createIoTensor(1, "unsupported_uint8_tensor", DT::UINT8, testDims, testStrides)},
         {1},
         bn_type_configs::getAllowedIoTypes()},
    };
}

// Validates all specified tensors have a specific required data type
inline std::vector<FixedDataTypeTestCase> getValidateFixedDataTypeTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto derivedDims
        = hipdnn_data_sdk::utilities::getDerivedShape(shapes::INFERENCE_4D[2]); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - matching types
        {"AcceptsMatchingFloat",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         DT::FLOAT},
        {"AcceptsMultipleMatchingFloat",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""},
          {2, "t2", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1, 2},
         DT::FLOAT},
        {"AcceptsMatchingHalf",
         true,
         {{1, "t1", DT::HALF, derivedDims, derivedStrides, ""}},
         {1},
         DT::HALF},

        // Unhappy paths - mismatched types
        {"RejectsMismatchedType_HalfExpectFloat",
         false,
         {{1, "t1", DT::HALF, derivedDims, derivedStrides, ""}},
         {1},
         DT::FLOAT},
        {"RejectsMismatchedType_FloatExpectHalf",
         false,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         DT::HALF},
        {"RejectsOneMismatch",
         false,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""},
          {2, "t2", DT::HALF, derivedDims, derivedStrides, ""}},
         {1, 2},
         DT::FLOAT},
    };
}

// Validates all tensors have the same shape
inline std::vector<ConsistentShapesTestCase> getValidateConsistentShapesTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto canonicalDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto canonicalStrides = hipdnn_data_sdk::utilities::generateStrides(
        canonicalDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(canonicalDims); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);
    auto differentChannelsDims = shapes::DIFFERENT_CHANNELS_4D; // {1, 5, 224, 224}
    auto differentChannelsStrides = hipdnn_data_sdk::utilities::generateStrides(
        differentChannelsDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - matching shapes
        {"AcceptsMatchingShape",
         true,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""}},
         {1},
         canonicalDims},
        {"AcceptsMultipleMatchingShapes",
         true,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""},
          {2, "t2", DT::FLOAT, canonicalDims, canonicalStrides, ""}},
         {1, 2},
         canonicalDims},
        {"AcceptsDerivedShape",
         true,
         {{1, "t1", DT::FLOAT, derivedDims, derivedStrides, ""}},
         {1},
         derivedDims},

        // Unhappy paths - mismatched shapes
        {"RejectsMismatchedShape_DifferentSpatial",
         false,
         {{1, "t1", DT::FLOAT, mediumDims, mediumStrides, ""}},
         {1},
         canonicalDims},
        {"RejectsMismatchedShape_DifferentChannels",
         false,
         {{1, "t1", DT::FLOAT, differentChannelsDims, differentChannelsStrides, ""}},
         {1},
         canonicalDims},
        {"RejectsOneMismatch",
         false,
         {{1, "t1", DT::FLOAT, canonicalDims, canonicalStrides, ""},
          {2, "t2", DT::FLOAT, mediumDims, mediumStrides, ""}},
         {1, 2},
         canonicalDims},
    };
}

// --- Test Classes: Layer 1 (Atomic Validators) ---

class TestValidateDimensionCount : public ::testing::TestWithParam<DimensionCountTestCase>
{
};

TEST_P(TestValidateDimensionCount, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateDimensionCount(tc.numDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateDimensionCount(tc.numDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateDimensionCount,
                         testing::ValuesIn(getValidateDimensionCountTestCases()));

class TestValidateConsistentDimensions
    : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidateConsistentDimensions, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateConsistentDimensions(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateConsistentDimensions(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentDimensions,
                         testing::ValuesIn(getValidateConsistentDimensionsTestCases()));

class TestValidatePackedTensors : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidatePackedTensors, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validatePackedTensors(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validatePackedTensors(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidatePackedTensors,
                         testing::ValuesIn(getValidatePackedTensorsTestCases()));

class TestValidateSupportedLayout : public ::testing::TestWithParam<SupportedLayoutTestCase>
{
};

TEST_P(TestValidateSupportedLayout, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateSupportedLayout(tc.strideOrder, tc.numDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateSupportedLayout(tc.strideOrder, tc.numDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateSupportedLayout,
                         testing::ValuesIn(getValidateSupportedLayoutTestCases()));
class TestValidateConsistentLayouts : public ::testing::TestWithParam<TensorDescriptorListTestCase>
{
};

TEST_P(TestValidateConsistentLayouts, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tc.tensorDims.size());
    for(size_t i = 0; i < tc.tensorDims.size(); ++i)
    {
        flatbuffers::FlatBufferBuilder builder;
        auto tensorOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            static_cast<int64_t>(i + 1),
            ("tensor_" + std::to_string(i + 1)).c_str(),
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tc.tensorStrides[i],
            &tc.tensorDims[i]);
        builder.Finish(tensorOffset);

        const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
        tensors.emplace_back(attr);
    }

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateConsistentLayouts(tensors); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateConsistentLayouts(tensors); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentLayouts,
                         testing::ValuesIn(getValidateConsistentLayoutsTestCases()));

class TestValidateDataTypeIsSupported : public ::testing::TestWithParam<DataTypeIsSupportedTestCase>
{
};

TEST_P(TestValidateDataTypeIsSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateDataTypeIsSupported(
                tc.dataType, tc.allowedTypes, "Test error message");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateDataTypeIsSupported(
                    tc.dataType, tc.allowedTypes, "Test error message");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateDataTypeIsSupported,
                         testing::ValuesIn(getValidateDataTypeIsSupportedTestCases()));

class TestValidateConsistentDataTypes : public ::testing::TestWithParam<ConsistentDataTypesTestCase>
{
};

TEST_P(TestValidateConsistentDataTypes, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateConsistentDataTypes(
                tc.tensorIds, tensorMap, tc.allowedTypes, "Type error", "Consistency error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateConsistentDataTypes(
                    tc.tensorIds, tensorMap, tc.allowedTypes, "Type error", "Consistency error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentDataTypes,
                         testing::ValuesIn(getValidateConsistentDataTypesTestCases()));

class TestValidateFixedDataType : public ::testing::TestWithParam<FixedDataTypeTestCase>
{
};

TEST_P(TestValidateFixedDataType, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateFixedDataType(
                tc.tensorIds, tensorMap, tc.expectedType, "Type mismatch error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateFixedDataType(
                    tc.tensorIds, tensorMap, tc.expectedType, "Type mismatch error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateFixedDataType,
                         testing::ValuesIn(getValidateFixedDataTypeTestCases()));

class TestValidateConsistentShapes : public ::testing::TestWithParam<ConsistentShapesTestCase>
{
};

TEST_P(TestValidateConsistentShapes, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            validators::validateConsistentShapes(
                tc.tensorIds, tensorMap, tc.referenceShape, "Shape mismatch error");
        });
    }
    else
    {
        EXPECT_THROW(
            {
                validators::validateConsistentShapes(
                    tc.tensorIds, tensorMap, tc.referenceShape, "Shape mismatch error");
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateConsistentShapes,
                         testing::ValuesIn(getValidateConsistentShapesTestCases()));

class TestValidateSpatialDimensions : public ::testing::TestWithParam<SpatialDimensionsTestCase>
{
};

TEST_P(TestValidateSpatialDimensions, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ validators::validateSpatialDimensions(tc.ioDims); });
    }
    else
    {
        EXPECT_THROW(
            { validators::validateSpatialDimensions(tc.ioDims); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestValidateSpatialDimensions,
                         testing::ValuesIn(getValidateSpatialDimensionsTestCases()));

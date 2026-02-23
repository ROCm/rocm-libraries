// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestBatchnormApplicability.hpp"
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_plugin;
using hipdnn_data_sdk::utilities::TensorLayout;

// --- Test Case Structs: Layer 2 (Component Validators) ---

struct TensorLayoutsAndDimsTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;

    friend std::ostream& operator<<(std::ostream& os, const TensorLayoutsAndDimsTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorDataTypesComponentTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> ioTensorIds;
    std::vector<int64_t> affineTensorIds;
    std::vector<int64_t> statTensorIds;
    std::vector<int64_t> intermediateTensorIds;

    friend std::ostream& operator<<(std::ostream& os, const TensorDataTypesComponentTestCase& tc)
    {
        return os << tc.name;
    }
};

struct TensorShapesComponentTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    std::vector<int64_t> ioTensorIds;
    std::vector<int64_t> affineTensorIds;
    std::vector<int64_t> statTensorIds;
    bool isTraining;

    friend std::ostream& operator<<(std::ostream& os, const TensorShapesComponentTestCase& tc)
    {
        return os << tc.name;
    }
};

// --- Test Data Providers: Layer 2 (Component Validators) ---

inline std::vector<TensorLayoutsAndDimsTestCase> getCheckTensorLayoutsAndDimsSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

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

    return {
        // Happy paths - valid layouts and dimensions
        {"AcceptsNchw4D", true, {{1, "x", DT::FLOAT, dims4D, nchwStrides, ""}}},
        {"AcceptsNhwc4D", true, {{1, "x", DT::FLOAT, dims4D, nhwcStrides, ""}}},
        {"AcceptsNcdhw5D", true, {{1, "x", DT::FLOAT, dims5D, ncdhwStrides, ""}}},
        {"AcceptsNdhwc5D", true, {{1, "x", DT::FLOAT, dims5D, ndhwcStrides, ""}}},

        // Unhappy paths - mixed layouts
        {"RejectsMixedNchwNhwc",
         false,
         {{1, "x1", DT::FLOAT, dims4D, nchwStrides, ""},
          {2, "x2", DT::FLOAT, dims4D, nhwcStrides, ""}}},

        // Unhappy paths - mixed dimensions
        {"RejectsMixed4D5D",
         false,
         {{1, "x1", DT::FLOAT, dims4D, nchwStrides, ""},
          {2, "x2", DT::FLOAT, dims5D, ncdhwStrides, ""}}},

        // Unhappy paths - non-packed tensors
        {"RejectsNonPacked", false, {{1, "x", DT::FLOAT, dims4D, {200000, 60000, 250, 1}, ""}}},
    };
}

// IO: FLOAT/HALF/BFLOAT16, Affine: FLOAT, Stat: FLOAT
inline std::vector<TensorDataTypesComponentTestCase> getCheckTensorDataTypesSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto ioDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto ioStrides
        = hipdnn_data_sdk::utilities::generateStrides(ioDims, TensorLayout::NCHW.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims); // {1, 3, 1, 1}
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - valid IO data types (FLOAT, HALF, BFLOAT16) with FLOAT affine/stat
        {"AcceptsFloat_IO",
         true,
         {{1, "x", DT::FLOAT, ioDims, ioStrides, ""},
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""},
          {5, "activ_out", DT::FLOAT, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},
        {"AcceptsHalf_IO",
         true,
         {{1, "x", DT::HALF, ioDims, ioStrides, ""},
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""},
          {5, "activ_out", DT::HALF, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},
        {"AcceptsBfloat16_IO",
         true,
         {{1, "x", DT::BFLOAT16, ioDims, ioStrides, ""},
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""},
          {5, "activ_out", DT::BFLOAT16, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},

        // Unhappy paths - invalid data types
        {"RejectsInvalidInputDataType",
         false,
         {{1, "x", DT::UINT8, ioDims, ioStrides, ""}, // Invalid
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""},
          {5, "activ_out", DT::HALF, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},

        {"RejectsInvalidAffineType",
         false,
         {{1, "x", DT::HALF, ioDims, ioStrides, ""},
          {2, "y", DT::FLOAT, ioDims, ioStrides, ""},
          {3, "scale", DT::HALF, derivedDims, derivedStrides, ""}, // Invalid
          {4, "bias", DT::HALF, derivedDims, derivedStrides, ""}, // Invalid
          {5, "activ_out", DT::HALF, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},
        {"RejectsInvalidIntermediateDataType",
         false,
         {{1, "x", DT::HALF, ioDims, ioStrides, ""},
          {2, "y", DT::HALF, ioDims, ioStrides, ""}, // Invalid
          {3, "scale", DT::FLOAT, derivedDims, derivedStrides, ""},
          {4, "bias", DT::FLOAT, derivedDims, derivedStrides, ""},
          {5, "activ_out", DT::HALF, ioDims, ioStrides, ""}},
         {1, 5},
         {3, 4},
         {},
         {2}},
    };
}

// IO tensors match, affine/stat have derived shape
inline std::vector<TensorShapesComponentTestCase> getCheckTensorShapesSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;

    auto inferenceDims = shapes::INFERENCE_4D[2]; // {1, 3, 224, 224}
    auto inferenceStrides = hipdnn_data_sdk::utilities::generateStrides(
        inferenceDims, TensorLayout::NCHW.strideOrder);
    auto trainingDims = shapes::INFERENCE_4D[3]; // {2, 3, 224, 224}
    auto trainingStrides
        = hipdnn_data_sdk::utilities::generateStrides(trainingDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);
    auto insufficientDims = shapes::INSUFFICIENT_SPATIAL_4D; // {1, 3, 1, 1}
    auto insufficientStrides = hipdnn_data_sdk::utilities::generateStrides(
        insufficientDims, TensorLayout::NCHW.strideOrder);

    auto derivedDimsInference
        = hipdnn_data_sdk::utilities::getDerivedShape(inferenceDims); // {1, 3, 1, 1}
    auto derivedStridesInference = hipdnn_data_sdk::utilities::generateStrides(
        derivedDimsInference, TensorLayout::NCHW.strideOrder);
    auto derivedDimsTraining
        = hipdnn_data_sdk::utilities::getDerivedShape(trainingDims); // {1, 3, 1, 1}
    auto derivedStridesTraining = hipdnn_data_sdk::utilities::generateStrides(
        derivedDimsTraining, TensorLayout::NCHW.strideOrder);

    auto wrongChannelDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(
        shapes::DIFFERENT_CHANNELS_4D); // {1, 5, 1, 1}
    auto wrongChannelDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        wrongChannelDerivedDims, TensorLayout::NCHW.strideOrder);

    return {
        // Happy paths - valid shapes for inference
        {"AcceptsValidInferenceShapes",
         true,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {2, "y", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""},
          {4, "bias", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""}},
         {1, 2},
         {3, 4},
         {},
         false},

        // Happy paths - valid shapes for training with sufficient spatial
        {"AcceptsValidTrainingShapes",
         true,
         {{1, "x", DT::FLOAT, trainingDims, trainingStrides, ""},
          {2, "y", DT::FLOAT, trainingDims, trainingStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsTraining, derivedStridesTraining, ""}},
         {1, 2},
         {3},
         {},
         true},

        // Unhappy paths - inconsistent IO shapes
        {"RejectsInconsistentIoShapes",
         false,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {2, "y", DT::FLOAT, mediumDims, mediumStrides, ""},
          {3, "scale", DT::FLOAT, derivedDimsInference, derivedStridesInference, ""}},
         {1, 2},
         {3},
         {},
         false},

        // Unhappy paths - wrong derived channel count
        {"RejectsWrongDerivedChannels",
         false,
         {{1, "x", DT::FLOAT, inferenceDims, inferenceStrides, ""},
          {3, "scale", DT::FLOAT, wrongChannelDerivedDims, wrongChannelDerivedStrides, ""}},
         {1},
         {3},
         {},
         false},

        // Unhappy paths - insufficient spatial for training (B × S ≤ 1)
        {"RejectsInsufficientSpatialForTraining",
         false,
         {{1, "x", DT::FLOAT, insufficientDims, insufficientStrides, ""},
          {2, "y", DT::FLOAT, insufficientDims, insufficientStrides, ""}},
         {1, 2},
         {},
         {},
         true},
    };
}

// Note: validatePeerStatsNotPopulated is tested through integration tests below
// (RejectsBatchnormFwdTrainingWithPeerStats, RejectsBatchnormBackwardWithPeerStats)

// --- Test Classes: Layer 2 (Component Validators) ---

class TestCheckTensorLayoutsAndDimsSupported
    : public ::testing::TestWithParam<TensorLayoutsAndDimsTestCase>
{
};

TEST_P(TestCheckTensorLayoutsAndDimsSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkTensorLayoutsAndDimsSupported(tensorMap); });
    }
    else
    {
        EXPECT_THROW(
            { checkTensorLayoutsAndDimsSupported(tensorMap); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorLayoutsAndDimsSupported,
                         testing::ValuesIn(getCheckTensorLayoutsAndDimsSupportedTestCases()));

class TestCheckTensorDataTypesSupported
    : public ::testing::TestWithParam<TensorDataTypesComponentTestCase>
{
};

TEST_P(TestCheckTensorDataTypesSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkTensorDataTypesSupported(tc.ioTensorIds,
                                          tc.affineTensorIds,
                                          tc.statTensorIds,
                                          tc.intermediateTensorIds,
                                          tensorMap);
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkTensorDataTypesSupported(tc.ioTensorIds,
                                              tc.affineTensorIds,
                                              tc.statTensorIds,
                                              tc.intermediateTensorIds,
                                              tensorMap);
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorDataTypesSupported,
                         testing::ValuesIn(getCheckTensorDataTypesSupportedTestCases()));

class TestCheckTensorShapesSupported
    : public ::testing::TestWithParam<TensorShapesComponentTestCase>
{
};

TEST_P(TestCheckTensorShapesSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    auto [builder, tensorMap] = buildTensorMapFromConfigs(tc.tensorConfigs);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkTensorShapesSupported(
                tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap, tc.isTraining);
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkTensorShapesSupported(
                    tc.ioTensorIds, tc.affineTensorIds, tc.statTensorIds, tensorMap, tc.isTraining);
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckTensorShapesSupported,
                         testing::ValuesIn(getCheckTensorShapesSupportedTestCases()));

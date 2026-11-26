// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace test_conv_common
{

struct ConvTestCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> yDims;
    std::vector<int64_t> convPrePadding;
    std::vector<int64_t> convPostPadding;
    std::vector<int64_t> convStride;
    std::vector<int64_t> convDilation;
    unsigned seed;

    ConvTestCase(std::vector<int64_t>&& xDimsLocal,
                 std::vector<int64_t>&& wDimsLocal,
                 std::vector<int64_t>&& convPrePaddingLocal,
                 std::vector<int64_t>&& convPostPaddingLocal,
                 std::vector<int64_t>&& convStrideLocal,
                 std::vector<int64_t>&& convDilationLocal,
                 unsigned seedLocal)
        : xDims(std::move(xDimsLocal))
        , wDims(std::move(wDimsLocal))
        , convPrePadding(std::move(convPrePaddingLocal))
        , convPostPadding(std::move(convPostPaddingLocal))
        , convStride(std::move(convStrideLocal))
        , convDilation(std::move(convDilationLocal))
        , seed(seedLocal)
    {
        // Indices for dimensions
        // N - Batch size, always at index 0
        // C - Channels, always at index 1
        // D - Depth (for 5D tensors), always at index 2 if present
        // H - Height, always at index 2 for 4D tensors and index 3 for 5D tensors
        // W - Width, always at index 3 for 4D tensors and index 4 for 5D tensors
        constexpr int N = 0; // Batch size index

        if(xDims.size() != wDims.size())
        {
            throw std::invalid_argument("xDims and wDims must have the same number of dimensions.");
        }

        // Ensure xDims has at least 3 dimensions (N, C, and at least 1 spatial dimension)
        if(xDims.size() < 3)
        {
            throw std::invalid_argument(
                "xDims must have at least 3 dimensions (N, C, and at least 1 spatial dimension).");
        }

        // Determine the number of spatial dimensions
        auto spatialDims = xDims.size() - 2; // Exclude N and C

        // Validate that the convolution parameter vectors match the number of spatial dimensions
        if(convPrePadding.size() != spatialDims || convPostPadding.size() != spatialDims
           || convDilation.size() != spatialDims || convStride.size() != spatialDims)
        {
            throw std::invalid_argument(
                "Convolution parameter vectors must match the number of spatial dimensions.");
        }

        // Calculate output dimensions based on input dimensions and convolution parameters
        auto n = xDims[N];
        auto cOut = wDims[N];
        std::vector<int64_t> outputDims = {n, cOut};

        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto paddedInputSize = xDims[2 + i] + convPrePadding[i] + convPostPadding[i];
            auto effectiveKernelSize = (convDilation[i] * (wDims[2 + i] - 1)) + 1;
            auto dimOut = ((paddedInputSize - effectiveKernelSize) / convStride[i]) + 1;
            outputDims.push_back(dimOut);
        }

        yDims = outputDims;
    }

    friend std::ostream& operator<<(std::ostream& ss, const ConvTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(x:";
        vecToStream(ss, tc.xDims);
        ss << " w:";
        vecToStream(ss, tc.wDims);
        ss << " y:";
        vecToStream(ss, tc.yDims);
        ss << " prePad:";
        vecToStream(ss, tc.convPrePadding);
        ss << " postPad:";
        vecToStream(ss, tc.convPostPadding);
        ss << " stride:";
        vecToStream(ss, tc.convStride);
        ss << " dilation:";
        vecToStream(ss, tc.convDilation);
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<ConvTestCase> getConvTestCases3D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1
        {{1, 16, 16}, {1, 16, 1}, {0}, {0}, {1}, {1}, seed},
        // Filter 3, no padding
        {{1, 16, 16}, {1, 16, 3}, {0}, {0}, {1}, {1}, seed},
        // Padding = 1
        {{1, 16, 16}, {1, 16, 3}, {1}, {1}, {1}, {1}, seed},
        // Stride = 2
        {{1, 16, 16}, {1, 16, 3}, {1}, {1}, {2}, {1}, seed},
        // Dilation = 2
        {{1, 16, 16}, {1, 16, 3}, {2}, {2}, {1}, {2}, seed},
        // Batched convolution
        {{8, 16, 16}, {1, 16, 1}, {0}, {0}, {1}, {1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16}, {2, 8, 3}, {1}, {1}, {1}, {1}, seed},
        // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
        {{2, 32, 16}, {4, 8, 3}, {1}, {1}, {2}, {2}, seed},
    };
}

inline std::vector<ConvTestCase> getConvTestCases4D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Filter 3x3
        // No Padding
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Padding = 1
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Stride = 2
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}, seed},
        // Dilation = 2
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, seed},
        // Batched convolution
        {{8, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Non-square
        {{1, 16, 16, 8}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16}, {2, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
        {{2, 32, 16, 16}, {4, 8, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, seed},
    };
}

inline std::vector<ConvTestCase> getConvTestCases5D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1
        {{1, 16, 16, 16, 16}, {1, 16, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Filter 3x3
        // No Padding
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Padding = 1
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Stride = 2
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, seed},
        // Dilation = 2
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, seed},
        // Batched convolution
        {{8, 16, 16, 16, 16}, {1, 16, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Non-square
        {{1, 16, 16, 8, 4}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16, 16}, {2, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
        {{2, 32, 16, 16, 16}, {4, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, seed},
    };
}

// Small cases the deterministic MIOpen engine supports for fprop, dgrad and wgrad.
inline std::vector<ConvTestCase> getDeterministicConvTestCases4D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1 - basic case
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Filter 3x3 with padding - common case
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16}, {2, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
    };
}

inline std::vector<ConvTestCase> getDeterministicConvTestCases5D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1x1 - basic 5D case
        {{1, 8, 8, 8, 8}, {1, 8, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Filter 3x3x3 with padding - common 5D case
        {{1, 8, 8, 8, 8}, {1, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
    };
}

// Packed strides of `dims` in channels-last or channels-first order.
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLastLayout)
{
    return channelsLastLayout ? hipdnn_data_sdk::utilities::generateStrides(
                                    dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size()))
                              : hipdnn_data_sdk::utilities::generateStrides(dims);
}

// One 2D convolution (unit stride, no padding, no dilation) shared by the tests of the plan
// builders and of the applicability checks: x [2, 4, 8, 8] with w [4, 4, 3, 3] gives
// y [2, 4, 6, 6].
inline const std::vector<int64_t> CONV_2D_X_DIMS = {2, 4, 8, 8};
inline const std::vector<int64_t> CONV_2D_W_DIMS = {4, 4, 3, 3};
inline const std::vector<int64_t> CONV_2D_Y_DIMS = {2, 4, 6, 6};

// The 2D bf16 convolution of the mixed-layout bundle that MIOpen fails on, in backward-data roles
// (forward poses the same problem with x the shape of dx and y the shape of dy): dy and dx are
// channels-first and w is channels-last, with the exact strides of the bundle.
inline const std::vector<int64_t> MIXED_LAYOUT_2D_DY_DIMS = {2, 32, 14, 14};
inline const std::vector<int64_t> MIXED_LAYOUT_2D_DY_STRIDES = {6272, 196, 14, 1};
inline const std::vector<int64_t> MIXED_LAYOUT_2D_W_DIMS = {32, 16, 3, 3};
inline const std::vector<int64_t> MIXED_LAYOUT_2D_W_STRIDES = {144, 1, 48, 16};
inline const std::vector<int64_t> MIXED_LAYOUT_2D_DX_DIMS = {2, 16, 16, 16};
inline const std::vector<int64_t> MIXED_LAYOUT_2D_DX_STRIDES = {4096, 256, 16, 1};
inline constexpr hipdnn_flatbuffers_sdk::data_objects::DataType MIXED_LAYOUT_2D_DATA_TYPE
    = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;

// The 1D bf16 convolution of the same mixed-layout bundle, in the same roles: dy and dx are
// channels-first and w is channels-last.
inline const std::vector<int64_t> MIXED_LAYOUT_1D_DY_DIMS = {8, 32, 16};
inline const std::vector<int64_t> MIXED_LAYOUT_1D_DY_STRIDES = {512, 16, 1};
inline const std::vector<int64_t> MIXED_LAYOUT_1D_W_DIMS = {32, 16, 1};
inline const std::vector<int64_t> MIXED_LAYOUT_1D_W_STRIDES = {16, 1, 16};
inline const std::vector<int64_t> MIXED_LAYOUT_1D_DX_DIMS = {8, 16, 16};
inline const std::vector<int64_t> MIXED_LAYOUT_1D_DX_STRIDES = {256, 16, 1};

// The reason the fused applicability check gives when the bias data type differs from the data
// type of the convolution input.
inline std::string biasDataTypeReason(hipdnn_flatbuffers_sdk::data_objects::DataType bias,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType input)
{
    using hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType;
    return std::string("Bias tensor data type ") + EnumNameDataType(bias)
           + " must equal the convolution input data type " + EnumNameDataType(input);
}

// The reason for the graph of cbaGraphWithFloatBiasOnHalfGraph, which is biasDataTypeReason(FLOAT,
// HALF), as the literal (a const char*) that the table of declined graphs of the fused builder
// test holds.
inline constexpr const char* FLOAT_BIAS_ON_HALF_GRAPH_REASON
    = "Bias tensor data type FLOAT must equal the convolution input data type HALF";

// A conv -> bias -> activation graph whose convolution and bias intermediates and bias are float
// while the input, weights and final output are half: the graph the node checks of the fused
// builder let through up to the bias data type rule.
inline flatbuffers::FlatBufferBuilder cbaGraphWithFloatBiasOnHalfGraph()
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_flatbuffers_sdk::data_objects::GraphT;

    return hipdnn_test_sdk::utilities::mutateGraph(
        hipdnn_test_sdk::utilities::createValidConvFwdBiasActivGraph(), [](GraphT& graph) {
            for(auto& tensor : graph.tensors)
            {
                const bool isFloat = tensor->name == "bias" || tensor->name == "y_conv"
                                     || tensor->name == "y_bias";
                tensor->data_type = isFloat ? DataType::FLOAT : DataType::HALF;
            }
        });
}

} // namespace test_conv_common

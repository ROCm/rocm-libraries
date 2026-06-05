// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <array>
#include <sstream>

#include "harness/NodeTypeNames.hpp"

using hipdnn_frontend::graph::NodeType;
using hipdnn_integration_tests::to_string;
using hipdnn_integration_tests::
    operator<<; // NOLINT(misc-unused-using-decls) -- needed by stream insertion below; ADL cannot find it

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestNodeTypeToString, AllNodeTypesHaveNonNullString)
{
    const std::array allTypes = {
        NodeType::UNKNOWN,
        NodeType::CONVOLUTION_FPROP,
        NodeType::CONVOLUTION_DGRAD,
        NodeType::CONVOLUTION_WGRAD,
        NodeType::BATCHNORM,
        NodeType::BATCHNORM_INFERENCE,
        NodeType::BATCHNORM_BACKWARD,
        NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT,
        NodeType::POINTWISE,
        NodeType::MATMUL,
        NodeType::LAYER_NORM,
        NodeType::RMS_NORM,
        NodeType::SDPA_FWD,
        NodeType::SDPA_BWD,
        NodeType::BLOCK_SCALE_QUANTIZE,
        NodeType::BLOCK_SCALE_DEQUANTIZE,
        NodeType::CUSTOM_OP,
        NodeType::REDUCTION,
    };

    for(auto type : allTypes)
    {
        EXPECT_NE(to_string(type), nullptr) << "NodeType " << static_cast<int>(type);
    }
}

TEST(TestNodeTypeToString, SpecificMappings)
{
    EXPECT_STREQ(to_string(NodeType::UNKNOWN), "Unknown");
    EXPECT_STREQ(to_string(NodeType::CONVOLUTION_FPROP), "ConvFprop");
    EXPECT_STREQ(to_string(NodeType::CONVOLUTION_DGRAD), "ConvDgrad");
    EXPECT_STREQ(to_string(NodeType::CONVOLUTION_WGRAD), "ConvWgrad");
    EXPECT_STREQ(to_string(NodeType::BATCHNORM), "Batchnorm");
    EXPECT_STREQ(to_string(NodeType::BATCHNORM_INFERENCE), "BatchnormInference");
    EXPECT_STREQ(to_string(NodeType::POINTWISE), "Pointwise");
    EXPECT_STREQ(to_string(NodeType::MATMUL), "Matmul");
    EXPECT_STREQ(to_string(NodeType::REDUCTION), "Reduction");
}

TEST(TestNodeTypeToString, StreamInsertionMatchesToString)
{
    const std::array testTypes = {
        NodeType::CONVOLUTION_FPROP,
        NodeType::POINTWISE,
        NodeType::BATCHNORM,
    };

    for(auto type : testTypes)
    {
        std::ostringstream oss;
        oss << type;
        EXPECT_EQ(oss.str(), to_string(type));
    }
}

// NOLINTEND(readability-identifier-naming)

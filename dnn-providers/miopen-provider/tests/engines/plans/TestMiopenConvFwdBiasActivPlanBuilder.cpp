/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "tests/common/ActivationCommon.hpp"
#include "tests/common/ConvolutionCommon.hpp"
#include "tests/common/TestWorkarounds.hpp"

#include <memory>
#include <unordered_map>

using namespace miopen_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_frontend::graph;

class TestMiopenConvFwdBiasActivPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        // No #6979 SKIP here: every TEST_F in this fixture exercises non-CBA
        // graphs or mocks, so isApplicable returns false regardless of whether
        // the device has a CBA solver. The skip belongs in the GPU fixture
        // below where buildPlan is actually invoked on a CBA graph.
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

enum class TypeKey
{
    X,
    W,
    BIAS,
    Y_CONV,
    Y_BIAS,
    Y,
    BIAS_COMPUTE,
    CONV_COMPUTE,
    ACTIV_COMPUTE
};

enum class FusedOp
{
    CA,
    CBA
};

std::ostream& operator<<(std::ostream& os, TypeKey tk)
{
    std::vector<const char*> keyMap{"x",
                                    "y",
                                    "bias",
                                    "y_conv",
                                    "y_bias",
                                    "y",
                                    "bias_compute_type",
                                    "conv_compute_type",
                                    "activation_compute_type"};
    os << keyMap[static_cast<size_t>(tk)];
    return os;
}

std::ostream& operator<<(std::ostream& os, FusedOp op)
{
    std::vector<const char*> keyMap{"CA", "CBA"};
    os << keyMap[static_cast<size_t>(op)];
    return os;
}

struct ConvolutionBiasActivationTestParam
{
    FusedOp op;
    test_conv_common::ConvTestCase convTestCase;
    hipdnn_data_sdk::utilities::TensorLayout layout;
    test_activation_common::ActivTestCase activTestCase;
    hipdnn_frontend::DataType defaultDataType;
    std::unordered_map<TypeKey, hipdnn_frontend::DataType> dataTypes;
    std::vector<TypeKey> virtualTensors;
    bool isApplicable;

    std::string label;

    friend std::ostream& operator<<(std::ostream& os, const ConvolutionBiasActivationTestParam& tc)
    {
        using namespace hipdnn_data_sdk::utilities;
        os << tc.label;
        os << ", Operation:" << tc.op;
        os << ", Conv: " << tc.convTestCase;
        os << ", Layout: "
           << (tc.layout.name.empty() ? vecToString(tc.layout.strideOrder) : tc.layout.name);
        os << ", Activ: " << tc.activTestCase;
        os << ", DefaultDataType: " << tc.defaultDataType;
        os << ", DataTypes: [";
        for(const auto& [typeKey, dataType] : tc.dataTypes)
        {
            os << typeKey << ": " << dataType << ", ";
        }
        os << "], ";
        os << "Virtual tensors: " << vecToString(tc.virtualTensors);

        return os;
    }
};

class TestGpuMiopenConvFwdBiasActivPlanBuilder
    : public ::testing::TestWithParam<ConvolutionBiasActivationTestParam>
{
protected:
    void SetUp() override
    {
        // Re-enable in Windows once CK is supported
        SKIP_IF_WINDOWS();

        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();

        auto param = GetParam();
        auto& convTestCase = param.convTestCase;
        auto& activTestCase = param.activTestCase;
        auto& layout = param.layout;

        auto isVirtual = [&param](TypeKey key) {
            return std::find(param.virtualTensors.begin(), param.virtualTensors.end(), key)
                   != param.virtualTensors.end();
        };

        _graphObj.set_compute_data_type(
            hipdnn_frontend::DataType::
                FLOAT); // Consider for simplicity making this the data type too

        _graphObj.set_intermediate_data_type(param.defaultDataType);
        _graphObj.set_io_data_type(param.defaultDataType);

        auto xTensorAttrObj
            = makeTensorAttributes("x",
                                   param.dataTypes[TypeKey::X],
                                   convTestCase.xDims,
                                   generateStrides(convTestCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<TensorAttributes>(std::move(xTensorAttrObj));
        xTensorAttr->set_is_virtual(isVirtual(TypeKey::X));

        auto wTensorAttrObj
            = makeTensorAttributes("w",
                                   param.dataTypes[TypeKey::W],
                                   convTestCase.wDims,
                                   generateStrides(convTestCase.wDims, param.layout.strideOrder));
        auto wTensorAttr = std::make_shared<TensorAttributes>(std::move(wTensorAttrObj));
        wTensorAttr->set_is_virtual(isVirtual(TypeKey::W));

        ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(convTestCase.convPrePadding);
        convAttrs.set_post_padding(convTestCase.convPostPadding);
        convAttrs.set_stride(convTestCase.convStride);
        convAttrs.set_dilation(convTestCase.convDilation);
        convAttrs.set_compute_data_type(param.dataTypes[TypeKey::CONV_COMPUTE]);

        auto yConvTensorAttr = _graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yConvTensorAttr->set_data_type(param.dataTypes[TypeKey::Y_CONV]);
        yConvTensorAttr->set_dim(convTestCase.yDims);
        yConvTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yConvTensorAttr->set_is_virtual(isVirtual(TypeKey::Y_CONV));
        std::shared_ptr<TensorAttributes> yBiasTensorAttr;
        if(param.op == FusedOp::CBA)
        {
            const auto biasDims = getDerivedShape(convTestCase.yDims);

            auto biasTensorAttrObj
                = makeTensorAttributes("bias",
                                       param.dataTypes[TypeKey::BIAS],
                                       biasDims,
                                       generateStrides(biasDims, layout.strideOrder));
            auto biasTensorAttr = std::make_shared<TensorAttributes>(std::move(biasTensorAttrObj));
            biasTensorAttr->set_is_virtual(isVirtual(TypeKey::BIAS));

            PointwiseAttributes biasAttrs;
            biasAttrs.set_name("bias");
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(param.dataTypes[TypeKey::BIAS_COMPUTE]);

            yBiasTensorAttr = _graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
            yBiasTensorAttr->set_name("y_bias");
            yBiasTensorAttr->set_data_type(param.dataTypes[TypeKey::Y_BIAS]);
            yBiasTensorAttr->set_dim(convTestCase.yDims);
            yBiasTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
            yBiasTensorAttr->set_is_virtual(isVirtual(TypeKey::Y_BIAS));
        }

        PointwiseAttributes activAttrs;
        activAttrs.set_name("activation_forward");
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
        activAttrs.set_compute_data_type(param.dataTypes[TypeKey::ACTIV_COMPUTE]);
        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto yTensorAttr = _graphObj.pointwise(
            param.op == FusedOp::CBA ? yBiasTensorAttr : yConvTensorAttr, activAttrs);
        yTensorAttr->set_data_type(param.dataTypes[TypeKey::Y]);
        yTensorAttr->set_dim(convTestCase.yDims);
        yTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yTensorAttr->set_output(true);
        yTensorAttr->set_is_virtual(isVirtual(TypeKey::Y));

        auto status = _graphObj.validate();
        ASSERT_TRUE(status.is_good());

        _isApplicable = param.isApplicable;
    }

    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _handle;

    Graph _graphObj;
    bool _isApplicable;
};

test_conv_common::ConvTestCase validConvTestCase4d()
{
    return {{2, 3, 4, 4}, {1, 3, 2, 2}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, 0};
}

test_conv_common::ConvTestCase validConvTestCase5d()
{
    return {{2, 3, 4, 4, 4}, {1, 3, 2, 2, 2}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 0};
};

test_activation_common::ActivTestCase validActivTestCase()
{
    return {hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.0,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt};
};

std::vector<ConvolutionBiasActivationTestParam> testParams()
{
    using Param = ConvolutionBiasActivationTestParam;
    using DataType = hipdnn_frontend::DataType;

    std::vector<Param> params = {
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Valid NCHW"},
        Param{FusedOp::CBA,
              validConvTestCase5d(),
              TensorLayout::NCDHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Valid NCDHW"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NHWC,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Valid NHWC"},
        Param{FusedOp::CBA,
              validConvTestCase5d(),
              TensorLayout::NDHWC,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Valid NDHWC"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout{"", {0, 1, 2, 3}},
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Unsupported layout WHCN"},
        Param{FusedOp::CBA,
              validConvTestCase5d(),
              TensorLayout{"", {0, 1, 2, 3, 4}},
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Unsupported layout WHDCN"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              true,
              "Valid NCHW"},
        Param{FusedOp::CA,
              validConvTestCase5d(),
              TensorLayout::NCDHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              true,
              "Valid NCDHW"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NHWC,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              true,
              "Valid NHWC"},
        Param{FusedOp::CA,
              validConvTestCase5d(),
              TensorLayout::NDHWC,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              true,
              "Valid NDHWC"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout{"", {0, 1, 2, 3}},
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              false,
              "Unsupported layout WHCN"},
        Param{FusedOp::CA,
              validConvTestCase5d(),
              TensorLayout{"", {0, 1, 2, 3, 4}},
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              false,
              "Unsupported layout WHDCN"},

        // Virtual checks
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV},
              false,
              "Invalid, y_bias is non-virtual"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_BIAS},
              false,
              "Invalid, y_conv is non-virtual"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {},
              {},
              false,
              "Invalid, y_conv is non-virtual"},

        // Intermediate datatypes and compute type
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::BFLOAT16,
              {{TypeKey::BIAS_COMPUTE, DataType::BFLOAT16}},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Valid Bfloat16"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::BFLOAT16,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Invalid Bfloat16, bias compute_data_type should be Bfloat16"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::CONV_COMPUTE, DataType::BFLOAT16},
               {TypeKey::BIAS_COMPUTE, DataType::BFLOAT16}},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Invalid Bfloat16, conv compute_data_type should be Float"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::ACTIV_COMPUTE, DataType::BFLOAT16},
               {TypeKey::BIAS_COMPUTE, DataType::BFLOAT16}},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Invalid Bfloat16, activation compute_data_type should be float"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::BIAS_COMPUTE, DataType::BFLOAT16}, {TypeKey::Y_CONV, DataType::HALF}},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Invalid Bfloat16, y_conv intermediate tensor must be Bfloat16 or Float"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::BIAS_COMPUTE, DataType::BFLOAT16}, {TypeKey::Y_BIAS, DataType::HALF}},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Invalid Bfloat16, y_bias intermediate tensor must be Bfloat16 or Float"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::BFLOAT16,
              {{TypeKey::Y_CONV, DataType::FLOAT}},
              {TypeKey::Y_CONV},
              true,
              "Valid Bfloat16"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::CONV_COMPUTE, DataType::BFLOAT16}},
              {TypeKey::Y_CONV},
              false,
              "Invalid Bfloat16, conv compute_data_type should be Float"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::ACTIV_COMPUTE, DataType::BFLOAT16}},
              {TypeKey::Y_CONV},
              false,
              "Invalid Bfloat16, activation compute_data_type should be float"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              validActivTestCase(),
              hipdnn_frontend::DataType::FLOAT,
              {{TypeKey::Y_CONV, DataType::BFLOAT16}},
              {TypeKey::Y_CONV},
              false,
              "Invalid Bfloat16, y_conv intermediate tensor must be Float"},
        // Activation modes
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, std::nullopt, 0.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Clipped RELU is supported"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, 0.0, 1.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "CLAMP is supported"},
        Param{FusedOp::CBA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                  std::nullopt,
                  std::nullopt,
                  1.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Leaky RELU is NOT supported"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, std::nullopt, 0.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "Clipped RELU is supported"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, 0.0, 1.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              true,
              "CLAMP is supported"},
        Param{FusedOp::CA,
              validConvTestCase4d(),
              TensorLayout::NCHW,
              test_activation_common::ActivTestCase{
                  hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                  std::nullopt,
                  std::nullopt,
                  1.0},
              hipdnn_frontend::DataType::FLOAT,
              {},
              {TypeKey::Y_CONV, TypeKey::Y_BIAS},
              false,
              "Leaky RELU is NOT supported"},
    };

    return params;
}

TEST_P(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableGetWorkspaceSizeAndBuildPlan)
{
    auto [graphBuffer, serErr] = _graphObj.to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graph = GraphWrapper(graphBuffer.data(), graphBuffer.size());
    MockEngineConfig mockEngineConfig;

    // Per ROCm/rocm-libraries#6979: when this case expects an applicable engine
    // but the device's MIOpen has no CBA solver, skip rather than fail.
    // Negative cases (`_isApplicable == false`) intentionally proceed: their
    // rejection assertion passes either way.
    if(_isApplicable)
    {
        SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);
    }

    EXPECT_EQ(_planBuilder.isApplicable(*_handle, graph), _isApplicable);

    if(_isApplicable)
    {
        HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));

        HipdnnMiopenContext ctx;
        ASSERT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdBiasActivPlanBuilder,
                         testing::ValuesIn(testParams()));

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));

        HipdnnMiopenSettings settings;
        EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));

        HipdnnMiopenSettings settings;
        EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    HipdnnMiopenSettings settings;
    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForWrongNodeCountGraph)
{
    MockEngineConfig mockConfig;

    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        HipdnnMiopenContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, mockGraph, mockConfig, ctx),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        HipdnnMiopenContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, mockGraph, mockConfig, ctx),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    HipdnnMiopenContext ctx;
    MockEngineConfig mockConfig;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

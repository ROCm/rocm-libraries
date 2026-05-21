// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/GraphSignature.hpp"

namespace {

using ck_dsl_provider::GraphSignature;
using ck_dsl_provider::SignatureHash;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Helper: produce a ``ConvolutionFwdAttributes`` + matching tensor
/// map for one shape so the determinism / sensitivity tests can
/// perturb a single axis at a time.
struct ConvFixture {
    flatbuffers::FlatBufferBuilder convBuilder;
    flatbuffers::FlatBufferBuilder xBuilder;
    flatbuffers::FlatBufferBuilder wBuilder;
    flatbuffers::FlatBufferBuilder yBuilder;
    const data_objects::ConvolutionFwdAttributes* convAttr{nullptr};
    std::unordered_map<std::int64_t, const data_objects::TensorAttributes*> tensorMap;

    static const data_objects::TensorAttributes* makeTensor(
        flatbuffers::FlatBufferBuilder& builder, std::int64_t uid, const std::string& name,
        data_objects::DataType dtype, const std::vector<std::int64_t>& dims,
        const std::vector<std::int64_t>& strides) {
        auto attrOffset = data_objects::CreateTensorAttributesDirect(
            builder, uid, name.c_str(), dtype, &strides, &dims, /*virtual=*/false);
        builder.Finish(attrOffset);
        return flatbuffers::GetRoot<data_objects::TensorAttributes>(builder.GetBufferPointer());
    }

    ConvFixture(const std::vector<std::int64_t>& xDims = {8, 64, 56, 56},
                const std::vector<std::int64_t>& wDims = {64, 64, 3, 3},
                const std::vector<std::int64_t>& yDims = {8, 64, 56, 56},
                const std::vector<std::int64_t>& prePadding = {1, 1},
                const std::vector<std::int64_t>& postPadding = {1, 1},
                const std::vector<std::int64_t>& stride = {1, 1},
                const std::vector<std::int64_t>& dilation = {1, 1},
                data_objects::DataType dtype = data_objects::DataType::HALF) {
        // Strides are required by the FB schema; populate with NHWC
        // physical strides for X/Y and KRSC for W. The signature
        // function only reads ``dims`` (not strides), so these are
        // documentation only.
        std::vector<std::int64_t> xStrides{xDims[1] * xDims[2] * xDims[3], 1, xDims[3] * xDims[1],
                                           xDims[1]};
        std::vector<std::int64_t> wStrides{wDims[1] * wDims[2] * wDims[3], 1, wDims[3] * wDims[1],
                                           wDims[1]};
        std::vector<std::int64_t> yStrides{yDims[1] * yDims[2] * yDims[3], 1, yDims[3] * yDims[1],
                                           yDims[1]};

        tensorMap[1] = makeTensor(xBuilder, 1, "X", dtype, xDims, xStrides);
        tensorMap[2] = makeTensor(wBuilder, 2, "W", dtype, wDims, wStrides);
        tensorMap[3] = makeTensor(yBuilder, 3, "Y", dtype, yDims, yStrides);

        auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
            convBuilder, 1, 2, 3, &prePadding, &postPadding, &stride, &dilation,
            data_objects::ConvMode::CROSS_CORRELATION);
        convBuilder.Finish(convOffset);
        convAttr = flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(
            convBuilder.GetBufferPointer());
    }
};

TEST(TestGraphSignature, DeterministicForSameInput) {
    ConvFixture a;
    ConvFixture b;
    auto h1 = GraphSignature::computeForConvFwd("conv_implicit_gemm", *a.convAttr, a.tensorMap);
    auto h2 = GraphSignature::computeForConvFwd("conv_implicit_gemm", *b.convAttr, b.tensorMap);
    EXPECT_EQ(h1, h2);
}

TEST(TestGraphSignature, ChangesWithOpKind) {
    ConvFixture fx;
    auto h1 = GraphSignature::computeForConvFwd("conv_implicit_gemm", *fx.convAttr, fx.tensorMap);
    auto h2 = GraphSignature::computeForConvFwd("conv_other_op", *fx.convAttr, fx.tensorMap);
    EXPECT_NE(h1, h2);
}

TEST(TestGraphSignature, ChangesWithShape) {
    ConvFixture base;
    ConvFixture taller(/*xDims=*/{8, 64, 28, 56});  // halve Hi
    auto h1 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *base.convAttr, base.tensorMap);
    auto h2 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *taller.convAttr, taller.tensorMap);
    EXPECT_NE(h1, h2);
}

TEST(TestGraphSignature, ChangesWithStride) {
    ConvFixture base;
    ConvFixture strided({8, 64, 56, 56}, {64, 64, 3, 3}, {8, 64, 56, 56}, {1, 1}, {1, 1},
                        /*stride=*/{2, 2});
    auto h1 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *base.convAttr, base.tensorMap);
    auto h2 = GraphSignature::computeForConvFwd("conv_implicit_gemm", *strided.convAttr,
                                                strided.tensorMap);
    EXPECT_NE(h1, h2);
}

TEST(TestGraphSignature, ChangesWithPadding) {
    ConvFixture base;
    ConvFixture noPad({8, 64, 56, 56}, {64, 64, 3, 3}, {8, 64, 56, 56}, /*prePadding=*/{0, 0},
                      /*postPadding=*/{0, 0});
    auto h1 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *base.convAttr, base.tensorMap);
    auto h2 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *noPad.convAttr, noPad.tensorMap);
    EXPECT_NE(h1, h2);
}

TEST(TestGraphSignature, ChangesWithDtype) {
    ConvFixture half;
    ConvFixture float32({8, 64, 56, 56}, {64, 64, 3, 3}, {8, 64, 56, 56}, {1, 1}, {1, 1}, {1, 1},
                        {1, 1}, data_objects::DataType::FLOAT);
    auto h1 =
        GraphSignature::computeForConvFwd("conv_implicit_gemm", *half.convAttr, half.tensorMap);
    auto h2 = GraphSignature::computeForConvFwd("conv_implicit_gemm", *float32.convAttr,
                                                float32.tensorMap);
    EXPECT_NE(h1, h2);
}

TEST(TestGraphSignature, RejectsMissingTensor) {
    ConvFixture fx;
    decltype(fx.tensorMap) emptyMap;
    EXPECT_THROW(GraphSignature::computeForConvFwd("conv_implicit_gemm", *fx.convAttr, emptyMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

}  // namespace

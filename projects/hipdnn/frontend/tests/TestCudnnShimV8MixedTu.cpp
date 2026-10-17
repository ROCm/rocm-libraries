// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// One translation unit that, like a ported consumer, includes only the shim
// umbrella and builds a convolution through both the v9 graph API and the v0.x
// operation-graph builders.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

namespace
{

using Dims = std::vector<int64_t>;

Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

cudnn_frontend::Tensor tensor(int64_t uid, const Dims& dims)
{
    const Dims strides = packedStrides(dims);
    return cudnn_frontend::TensorBuilder()
        .setDataType(cudnn_frontend::DataType_t::FLOAT)
        .setDim(static_cast<int64_t>(dims.size()), dims.data())
        .setStride(static_cast<int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .build();
}

class TestCudnnShimV8MixedTu : public hipdnn_shim_test::ShimV8MockBackendFixture
{
};

TEST_F(TestCudnnShimV8MixedTu, GraphAndBuildersShareTu)
{
    const Dims xDims = {2, 4, 8, 8};
    const Dims wDims = {8, 4, 3, 3};
    const Dims yDims = {2, 8, 8, 8};

    cudnn_frontend::graph::Graph graph;
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    auto x = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("x")
                              .set_dim(xDims)
                              .set_stride(packedStrides(xDims))
                              .set_uid(1));
    auto w = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("w")
                              .set_dim(wDims)
                              .set_stride(packedStrides(wDims))
                              .set_uid(2));
    auto y = graph.conv_fprop(x,
                              w,
                              cudnn_frontend::graph::Conv_fprop_attributes()
                                  .set_padding({1, 1})
                                  .set_stride({1, 1})
                                  .set_dilation({1, 1}));
    ASSERT_NE(y, nullptr);
    y->set_output(true).set_uid(3);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());

    const std::array<int64_t, 2> ones = {1, 1};
    const cudnn_frontend::ConvDesc conv
        = cudnn_frontend::ConvDescBuilder()
              .setComputeType(cudnn_frontend::DataType_t::FLOAT)
              .setMathMode(cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION)
              .setSpatialDimCount(2)
              .setPrePadding(2, ones.data())
              .setPostPadding(2, ones.data())
              .setDilation(2, ones.data())
              .setSpatialStride(2, ones.data())
              .build();
    const cudnn_frontend::Operation op
        = cudnn_frontend::OperationBuilder(
              cudnn_frontend::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
              .setxDesc(tensor(1, xDims))
              .setwDesc(tensor(2, wDims))
              .setyDesc(tensor(3, yDims))
              .setcDesc(conv)
              .build();
    const cudnn_frontend::Operation* ops = &op;
    cudnn_frontend::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(1, &ops);
    const cudnn_frontend::OperationGraph opGraph = builder.build();

    EXPECT_EQ(opGraph.get_status(), CUDNN_STATUS_SUCCESS) << opGraph.get_error();
    EXPECT_EQ(opGraph.getTag(), "ConvFwd_");
    // The v9 graph and the v0.x graph each lowered one hipDNN graph.
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 2U);

    const std::string status = cudnn_frontend::to_string(CUDNN_STATUS_SUCCESS);
    EXPECT_FALSE(status.empty());
    EXPECT_EQ(cudnn_frontend::detail::get_compiled_version(),
              cudnn_frontend::detail::get_backend_version());
}

} // namespace
